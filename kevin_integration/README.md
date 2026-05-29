# Kevin Integration

This folder contains the integration of the trained AAM, inverse dynamics, system dynamics, and learned forward-dynamics torque model into the existing RRLAB Isaac Lab Mulag environment.

## Goal

Use the existing RRLAB Mulag environment and integrate:

1. Arm Adaptation Module
2. Learned inverse dynamics model
3. Learned system dynamics model for internal `dP`/`Fnet` estimation
4. Learned forward dynamics model for torque prediction
5. Final learned control pipeline

## Development rule

Do not modify `rrlab_assets` unless changing the robot asset itself.
Do not modify the original RRLAB task until the integration works separately.

## Learned Models

The trained checkpoints live in `kevin_integration/models/`:

- `aam.pth`: Arm Adaptation Module, history -> `z_arm_hat` (24-d)
- `id.pth`: inverse dynamics, `[q, qdot, qddot_cmd, dP]` history + `z_arm_hat` -> `valve_cmd` (4-d)
- `fd.pth`: forward dynamics, `[q, qdot, dP, valve_cmd]` history -> Simscape-domain output (4-d)
- `fd_scaler.pth`: scaler used by the FD checkpoint
- `sd.pth`: system dynamics observer, optional hydraulic-state feedback

The code that loads and runs these models is in:

- `kevin_integration/controllers/learned_models.py`
- `kevin_integration/controllers/pipeline.py`
- `kevin_integration/control_policy/policy.py`
- `kevin_integration/rl/adaptive_rl_controller.py`
- `kevin_integration/rl/torque_adapter.py`

History window lengths are read from checkpoint metadata when present. If a checkpoint does not expose metadata,
the fallback window is used.

## Feature Order

All learned-model vectors must use the same feature order as training:

- AAM input: `[q5, qdot5, dP4, valve4]`
- ID input: `[q5, qdot5, qddot_cmd5, dP4]`
- FD input: `[q5, qdot5, dP4, valve4]`
- SD input: `[q5, qdot5, dP4, Fnet4, valve4]`
- SD output/state: `[q5, qdot5, dP4, Fnet4]`

PPO outputs a 4-d high-level action. The adaptive RL task maps it to `qddot_cmd[5]` with:
`qddot_cmd[0:4] = action_to_qddot_scale * action`, `qddot_cmd[4] = 0`, then clamps to
`[-qddot_cmd_clip, qddot_cmd_clip]`.

## Standalone Smoke Tests

This only verifies that the frozen learned-model chain loads and produces finite inference outputs:

```bash
./rrlab.sh -p kevin_integration/scripts/run_control_policy.py \
  --device cpu \
  --steps 5 \
  --debug
```

This runner launches IsaacLab, spawns `MULAG_CFG`, finds the arm joints, builds feature vectors from live joint
state, and applies the learned command to the simulator. It uses 5 state joints and 4 commanded arm joints.

```bash
./rrlab.sh -p kevin_integration/scripts/run_mulag_learned_controller.py --num_envs 1
```

Safer first test:

```bash
./rrlab.sh -p kevin_integration/scripts/run_mulag_learned_controller.py \
  --num_envs 1 \
  --duration 10 \
  --control_mode position_delta \
  --position_scale 0.005
```

FD torque mode should start with a low clamp:

```bash
./rrlab.sh -p kevin_integration/scripts/run_mulag_learned_controller.py \
  --num_envs 1 \
  --duration 10 \
  --control_mode effort \
  --max_abs_effort 10
```

Default state joints:
- `Drehzapfen_joint`
- `Ausleger_I_joint`
- `Ausleger_II_joint`
- `Messerkopf_Schwenk_joint`
- `Messerkopf_joint`

Default command joints:
- `Drehzapfen_joint`
- `Ausleger_I_joint`
- `Ausleger_II_joint`
- `Messerkopf_Schwenk_joint`

## Adaptive RL task

The adaptive RL task is registered as:

```text
Kevin-Mulag-AdaptiveRL-JointReach-v0
```

Only the PPO policy learns. AAM, ID, FD, and SD are frozen and run in eval mode.

Intended final control chain:

```text
observation/history -> AAM -> z_arm_hat
observation + z_arm_hat -> PPO policy -> action
action -> qddot_cmd
[q, qdot, qddot_cmd, dP] + z_arm_hat -> ID -> valve_cmd
[q, qdot, dP, valve_cmd] -> FD -> fd_simscape_output
fd_simscape_output -> torque_adapter -> tau_fd_adapted
tau_fd_adapted -> fd_residual_alpha -> tau_applied_raw
tau_applied_raw -> low-pass filter -> rate limit -> final clamp -> Isaac joint effort target
Isaac simulation -> next observation/history
```

The FD output is not an Isaac joint effort. Current torque debugging showed Isaac internal torques mostly around
`1e3` to `1e5`, while FD outputs are often `1e6` to `1e7` even for zero valve and zero pressure. The effort path
therefore always runs `fd_simscape_output` through the Simscape-to-Isaac adapter before applying effort.

## Control Modes

Debug order:

1. `control_mode=position_delta`, `position_delta_source=direct_action`, `use_fd_effort=False`,
   `use_sd_feedback=False`: PPO action directly drives position deltas.
2. `control_mode=position_delta`, `position_delta_source=id_valve`, `use_fd_effort=False`,
   `use_sd_feedback=False`: AAM+ID produces valve commands; valve commands drive position deltas.
3. `control_mode=effort`, `position_delta_source=id_valve`, `use_fd_effort=True`,
   `use_sd_feedback=False`: AAM+ID+FD are active; adapted and clamped Isaac torque drives effort targets.
4. Later: effort mode with `use_sd_feedback=True` or other ROM/hydraulic feedback.

Current safe defaults are the second mode. Torque defaults are intentionally low: `max_abs_torque=5.0`,
`torque_adapter_mode=scale_bias`, `torque_adapter_preset=conservative`, `use_fd_residual_alpha=True`,
`fd_residual_alpha=0.002`, `torque_rate_limit=1.0`, and `torque_lowpass_alpha=0.2`.

## Torque Adapter

Available adapter modes:

- `scale_bias`: `tau_fd_adapted_j = scale_j * fd_simscape_j + bias_j`; default scale is
  the conservative preset `[0.0001, 0.001, 0.0005, 0.0001]`.
- `tanh_squash`: `tau_fd_adapted = max_abs_torque * tanh(fd_simscape_output / fd_torque_scale)`.
- `residual_pd`: optional simple PD torque plus a small scaled FD residual.
- `none`: direct passthrough for debugging only. Do not use it for normal effort runs.

The applied raw effort candidate is then:

```text
tau_applied_raw = fd_residual_alpha * tau_fd_adapted
```

The first safe value is `fd_residual_alpha=0.002`, based on diagnostics where conservative `tau_fd_adapted`
values around `268.84`, `1087.11`, and `2827.66` become approximately `0.54`, `2.17`, and `5.66`. This gives
the learned FD torque limited authority while keeping `max_abs_torque=5`.
With the conservative preset, the effective scale is `[2e-7, 2e-6, 1e-6, 2e-7]`.

Scale presets:

- `conservative`: `[0.0001, 0.001, 0.0005, 0.0001]`
- `moderate`: `[0.0003, 0.003, 0.001, 0.0002]`
- `aggressive`: `[0.001, 0.03, 0.003, 0.0005]`

Raw FD output must never be applied directly as Isaac effort. Keep the adapter deterministic: improve it later
with explicit Simscape/Isaac calibration data or a geometry/moment-arm mapping, not a black-box learned mapper.

## Train And Evaluate

Run commands from the repository root:

```bash
cd /home/rrlab/rrlab_mulag
```

Safe tiny training smoke:

```bash
./rrlab.sh -p kevin_integration/scripts/train_mulag_adaptive_rl.py \
  --task Kevin-Mulag-AdaptiveRL-JointReach-v0 \
  --num_envs 1 \
  --max_iterations 2 \
  --headless
```

Longer safe ID-valve position-delta run:

```bash
./rrlab.sh -p kevin_integration/scripts/train_mulag_adaptive_rl.py \
  --task Kevin-Mulag-AdaptiveRL-JointReach-v0 \
  --num_envs 4 \
  --max_iterations 500 \
  --headless
```

Train the safe default control policy. This uses the current staged controller:
AAM+ID produce `valve_cmd`, and `valve_cmd` is applied as a safe position delta.

```bash
./rrlab.sh -p kevin_integration/scripts/train_mulag_adaptive_rl.py \
  --task Kevin-Mulag-AdaptiveRL-JointReach-v0 \
  --num_envs 4 \
  --max_iterations 500 \
  --headless
```

Low-torque full-pipeline smoke:

```bash
./rrlab.sh -p kevin_integration/scripts/train_mulag_adaptive_rl.py \
  --task Kevin-Mulag-AdaptiveRL-JointReach-v0 \
  --num_envs 1 \
  --max_iterations 2 \
  --control_mode effort \
  --use_fd_effort \
  --max_abs_torque 5 \
  --torque_ramp_steps 1000 \
  --headless
```

Train the low-torque full pipeline. This enables AAM+ID+FD and applies bounded effort torque.

```bash
./rrlab.sh -p kevin_integration/scripts/train_mulag_adaptive_rl.py \
  --task Kevin-Mulag-AdaptiveRL-JointReach-v0 \
  --num_envs 4 \
  --max_iterations 500 \
  --control_mode effort \
  --use_fd_effort \
  --max_abs_torque 5 \
  --torque_ramp_steps 1000 \
  --headless
```

Find the newest checkpoints after training:

```bash
find logs/rsl_rl/mulag_adaptive_rl_joint_reach -name "model_*.pt" | sort | tail -5
```

Terminal diagnostics without TensorBoard:

```bash
./rrlab.sh -p kevin_integration/scripts/evaluate_adaptive_rl_debug.py \
  --task Kevin-Mulag-AdaptiveRL-JointReach-v0 \
  --num_envs 1 \
  --steps 500 \
  --print_interval 50 \
  --headless
```

Visualize a trained safe/default policy without headless. Replace the checkpoint path with the newest
`model_*.pt` from the training run.

```bash
./rrlab.sh -p kevin_integration/scripts/evaluate_adaptive_rl_debug.py \
  --task Kevin-Mulag-AdaptiveRL-JointReach-v0 \
  --checkpoint logs/rsl_rl/mulag_adaptive_rl_joint_reach/<run_name>/model_499.pt \
  --num_envs 1 \
  --steps 2000 \
  --print_interval 50
```

Evaluate a low-torque effort checkpoint:

```bash
./rrlab.sh -p kevin_integration/scripts/evaluate_adaptive_rl_debug.py \
  --task Kevin-Mulag-AdaptiveRL-JointReach-v0 \
  --checkpoint /path/to/model.pt \
  --num_envs 1 \
  --steps 500 \
  --print_interval 50 \
  --control_mode effort \
  --use_fd_effort \
  --max_abs_torque 5 \
  --torque_adapter_preset conservative \
  --headless
```

Visualize a trained low-torque full-pipeline policy without headless:

```bash
./rrlab.sh -p kevin_integration/scripts/evaluate_adaptive_rl_debug.py \
  --task Kevin-Mulag-AdaptiveRL-JointReach-v0 \
  --checkpoint logs/rsl_rl/mulag_adaptive_rl_joint_reach/<run_name>/model_499.pt \
  --num_envs 1 \
  --steps 2000 \
  --print_interval 50 \
  --control_mode effort \
  --use_fd_effort \
  --max_abs_torque 5
```

Safety checks for effort mode:

- Start with `max_abs_torque=5`.
- Watch `fd_simscape_max`, `tau_fd_adapted_max`, `tau_applied_raw_max`, `tau_isaac_filtered_max`, `torque_max`, and
  `torque_clamp_fraction`.
- `fd_simscape_max` may remain huge; `tau_fd_adapted_max` can still be hundreds/thousands.
- `tau_applied_raw_max` should usually land around `0.2` to `6` with `fd_residual_alpha=0.002`.
- Tune in this order: `conservative`, then `moderate`, then `aggressive` only if stable.
- If torque is too weak, increase alpha gradually: `0.002 -> 0.003 -> 0.005`. Do not jump to `0.01` yet.
- For early tests, aim for `torque_clamp_fraction < 0.5`, and
  `torque_rate_limited_fraction < 1.0`.
- `torque_clamp_fraction` near `1.0` means the adapted Isaac torque is still saturating.
- Do not increase `max_abs_torque` until clamp fraction is already low.
- Keep `use_sd_feedback=False` until AAM+ID+FD effort mode is stable.
