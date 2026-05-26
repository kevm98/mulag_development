# Kevin Integration

This folder contains the integration of the trained AAM, inverse dynamics, and learned forward-dynamics torque model into the existing RRLAB Isaac Lab Mulag environment.

## Goal

Use the existing RRLAB Mulag environment and integrate:

1. Arm Adaptation Module
2. Learned inverse dynamics model
3. Learned forward dynamics model for torque prediction
4. Final learned control pipeline

## Development rule

Do not modify `rrlab_assets` unless changing the robot asset itself.
Do not modify the original RRLAB task until the integration works separately.

## Learned models + controller scaffold

The trained checkpoints live in `kevin_integration/models/`:
- `aam.pth` (AAM): history -> `z_arm_hat` (24-d)
- `id.pth` (ID): history + `z_arm_hat` -> `valve_cmd` (4-d)
- `fd.pth` + `fd_scaler.pth` (FD): history -> `torque` (4-d)

The code scaffold to *load and run* these models is in:
- `kevin_integration/controllers/learned_models.py`
- `kevin_integration/controllers/pipeline.py`
- `kevin_integration/control_policy/policy.py`

History window lengths are read from checkpoint metadata when present. If a checkpoint does not expose that
metadata, `PipelineConfig.fallback_window` is used.

### Usage (smoke test)

This only verifies that the frozen learned-model chain loads and produces finite inference outputs:

```bash
./rrlab.sh -p kevin_integration/scripts/run_control_policy.py \
  --device cpu \
  --steps 5 \
  --debug
```

### Usage (IsaacLab Mulag runner)

This runner reuses the working standalone controller structure from `standalone/tutorials/controllers/`:
it launches IsaacLab, spawns `MULAG_CFG`, finds the arm joints, builds feature vectors from live joint state,
and applies the learned command to the simulator. It uses 5 joints for model state features and 4 joints for
the valve-derived command output.

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

FD torque mode:

```bash
./rrlab.sh -p kevin_integration/scripts/run_mulag_learned_controller.py \
  --num_envs 1 \
  --duration 10 \
  --control_mode effort \
  --max_abs_effort 1000
```

In effort mode, FD outputs 4 actuator torques. The runner expands this to a 5-joint Isaac effort vector with:
`[tau_act1, tau_act2, tau_act3, tau_act4, 0.0]`, then applies it on the 5 `state_joints`.

### Integration contract (important)

`ControlPolicy.compute_action(observation)` expects a dict with:
- `aam_x_t`: shape `(18,)` = `[q(5), qdot(5), dP(4), valve_cmd(4)]`
- `id_x_t`: shape `(19,)` = `[q(5), qdot(5), qddot(5), dP(4)]`
Optional:
- `fd_x_t`: shape `(18,)` = `[q(5), qdot(5), dP(4), valve_cmd(4)]`

These vectors must be assembled in the **same feature order as used during training**.

The current IsaacLab runner estimates `qddot` by finite differencing `qdot`. It initializes `dP` to zeros
because those hydraulic signals are not currently exposed by `MULAG_CFG`; replace them with real, estimated,
or simulated hydraulic signals before judging physical performance.

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

The first real RL task is registered as:

```text
Kevin-Mulag-AdaptiveRL-JointReach-v0
```

The control chain is:

```text
q, qdot, dP, previous valve -> AAM -> z_arm_hat
q, qdot, reference, error, previous action, previous valve, z_arm_hat -> PPO policy -> a_t
q, qdot, [a_t, 0.0], dP + z_arm_hat -> ID -> valve_cmd
q, qdot, dP, valve_cmd -> FD -> torque -> Isaac Lab
```

Only the PPO policy learns. AAM, ID, and FD are loaded in eval mode with gradients disabled.

Train a small first pass:

```bash
./rrlab.sh -p kevin_integration/scripts/train_mulag_adaptive_rl.py \
  --task Kevin-Mulag-AdaptiveRL-JointReach-v0 \
  --num_envs 8 \
  --headless
```

Play a checkpoint:

```bash
./rrlab.sh -p kevin_integration/scripts/play_mulag_adaptive_rl.py \
  --task Kevin-Mulag-AdaptiveRL-JointReach-v0 \
  --checkpoint /path/to/model.pt \
  --num_envs 1
```
