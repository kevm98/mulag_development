# Kevin Mulag Hydraulic ROM Integration

This directory contains the active IsaacLab integration for the accepted Kevin
learned-control chain on the RRLAB Mulag asset.

The trained model architectures and checkpoints are treated as fixed. The active
work is runtime integration: feature assembly, checkpoint normalization,
high-level command convention, ROM interpretation, prismatic target writing,
preflight validation, and PPO lifecycle.

## Quick Start

Run from the repository root:

```bash
cd /home/rrlab/rrlab_mulag
```

Use the normal IsaacLab/RRLAB launcher environment for this repository:

```bash
conda activate env_isaaclab
```

Default play mode, no checkpoint. This runs honest `joint_space_validation`:
stationary vehicle, scripted mechanical reference, AAM/ID/ROM active, and FD
observer active. It does not claim task-space tracking.

```bash
./rrlab.sh -p kevin_integration/scripts/play_mulag_adaptive_rl.py
```

Task-space playback requires a trained PPO checkpoint:

```bash
./rrlab.sh -p kevin_integration/scripts/play_mulag_adaptive_rl.py \
  --checkpoint logs/rsl_rl/kevin_hydraulic_rom_joint_tracking/<run_name>/model_<iteration>.pt
```

The demo uses a 300 s episode, settles under gravity before learned control,
warm-starts histories from the settled state, ramps in the learned command, and
prints one-second signal-chain telemetry plus shutdown max/RMS summaries.

Runtime diagnostic preflight only:

```bash
./rrlab.sh -p kevin_integration/scripts/train_mulag_adaptive_rl.py \
  --num_envs 1 \
  --max_iterations 0 \
  --preflight_only
```

20-iteration PPO smoke training. This runs motion preflights before PPO:

```bash
./rrlab.sh -p kevin_integration/scripts/train_mulag_adaptive_rl.py \
  --num_envs 1 \
  --max_iterations 20
```

Checkpoint playback:

```bash
./rrlab.sh -p kevin_integration/scripts/play_mulag_adaptive_rl.py \
  --checkpoint logs/rsl_rl/kevin_hydraulic_rom_joint_tracking/<run_name>/model_<iteration>.pt
```

## Active Task

Gym task:

```text
Kevin-Mulag-HydraulicROM-JointTracking-v0
```

The default play command runs `control_mode=joint_space_validation`. With a
checkpoint, play switches to `control_mode=rl_task_space_tracking`.

The train script defaults to `control_mode=rl_task_space_tracking`, enables
scripted vehicle motion, and aborts before PPO if asset transmission,
learned-pipeline, task-space observation/reward, FD observer, or wheel-command
preflight fails. Use `--skip_motion_preflight` only for deliberate debugging.

## Active Architecture

```text
runtime history
-> AAM
-> z_arm_hat

PPO/scripted qddot action + runtime state + z_arm_hat
-> ID
-> ValveCmd[4]

ValveCmd[4] + training-mean dP + cylinder/mechanical state
-> valve-conditioned causal cylinder ROM
-> DeltaExtension_simscape[4]

DeltaExtension_simscape[4]
-> nominal-sign coordinate adapter
-> DeltaJoint_isaac[4] / five 10 ms control steps
-> accumulated prismatic position targets
-> USD linkage constraints
-> arm motion
```

The PPO action is four normalized desired accelerations for the four hydraulic
families. Runtime converts action `[-1, 1]` to physical `qddot_cmd[0:4]` using
the ID checkpoint qddot feature standard deviations. PPO does not output valves
directly.

Kevin now mirrors the working RRLAB Mulag obstacle task's vehicle convention:
wheel velocity is scripted separately from PPO/ID/ROM, sampled from 3 to
8 km/h, converted with the 0.575 m wheel radius, and written to the four wheel
joints when they are present in the active articulation.

The traced RRLAB target is task-space `ee_pose` for `Messerkopf` in robot-base
frame, ordered `[x, y, z, qw, qx, qy, qz]`. There is no IK bridge in the
reference task: PPO learns the map from `ee_pose` and measured state to
`Action[4]`. Kevin keeps that contract. PPO does not control the truck or write
ValveCmd directly.

## Runtime Reports

Static and runtime contract reports are under `kevin_integration/results/`:

- `static_signal_chain_audit.md`
- `runtime_valve_command_contract.md`
- `runtime_rom_motion_contract.md`
- `prismatic_drive_authority.md`
- `full_pipeline_preflight_contract.md`

Source audits for the RRLAB obstacle vehicle/reference convention:

- `RRLAB_VEHICLE_AND_REFERENCE_AUDIT.md`
- `RRLAB_TASKSPACE_REFERENCE_TRACE.md`
- `PPO_ACTION_REFERENCE_ALIGNMENT.md`

## Active Files

See [ACTIVE_PIPELINE_FILES.md](ACTIVE_PIPELINE_FILES.md) for the file map and
[ISAACLAB_CONTROL_CONTRACT.md](ISAACLAB_CONTROL_CONTRACT.md) for the runtime
contract.

Main runtime files:

- `kevin_integration/tasks/mulag_adaptive_rl_env.py`
- `kevin_integration/tasks/mulag_adaptive_rl_env_cfg.py`
- `kevin_integration/rl/adaptive_rl_controller.py`
- `kevin_integration/rl/observation_builder.py`
- `kevin_integration/rom_cylinders/valve_conditioned_causal_rom.py`
- `kevin_integration/rom_cylinders/multirate_rom_wrapper.py`
- `kevin_integration/rom_cylinders/cylinder_coordinate_adapter.py`
- `kevin_integration/scripts/play_mulag_adaptive_rl.py`
- `kevin_integration/scripts/train_mulag_adaptive_rl.py`

## Known Limitations

- The active `exported_valve_conditioned_causal_rom/` package uses the causal
  ValveCmd-conditioned model export. The old ROM training/evaluation tree has
  been removed from the runtime checkout.
- The cylinder coordinate mapping is sign verified and nominal-reference based,
  but not second-pose validated.
- Task-space tracking requires PPO training/checkpoint playback because the
  policy is the `ee_pose -> Action[4]` bridge.
- SD feedback is disabled for the active task.
- dP uses the final ROM training mean until a verified environment pressure
  source is available.
- USD-authored joint axes, masses, inertias, and drive APIs still require
  Isaac/USD introspection in an environment with `pxr` or Isaac runtime tools.
- Do not claim full runtime success until the scripted demo has visibly shown
  cylinder motion, linkage transmission, and acceptable vehicle/base behavior in
  Isaac.

## Curriculum (Learnability Repair)

`rl_task_space_tracking` mode uses a four-stage curriculum to improve early PPO
training stability:

| Stage | Steps       | Episode Length | Action Authority | Vehicle  | Position Range  | Orientation |
|-------|-------------|----------------|------------------|----------|-----------------|-------------|
| S1    | 0–50k       | 1500 st (15s)  | 0.25×            | off      | tight           | disabled    |
| S2    | 50k–150k    | 1500 st (15s)  | 0.50×            | off      | moderate        | disabled    |
| S3    | 150k–300k   | 3000 st (30s)  | 0.75×            | off      | wide            | enabled     |
| S4    | 300k+       | 4500 st (45s)  | 1.00×            | 3–8 km/h | full            | enabled     |

PPO action is scaled by the stage-dependent authority multiplier before qddot
conversion. Command ranges expand from tight volumes near nominal to the full
workspace. Vehicle remains stationary in S1–S3 and moves at 3–8 km/h in S4.
Orientation tracking reward is gated off in S1–S2.

### PPO Config Notes

`MulagHydraulicROMTaskSpacePPORunnerCfg` overrides:

- `num_steps_per_env = 128` (was 24 in `MulagAdaptiveRLPPORunnerCfg`)
- `init_std = 0.20` (higher initial policy entropy for exploration)
- `num_mini_batches = 4` → 32 samples per minibatch with 1 env

Do not remove `empirical_normalization`; it is required for stable training.

### Running Tests

Standalone curriculum unit tests (no IsaacLab dependency):

```bash
python kevin_integration/tasks/test_curriculum_learnability_repair.py
```

43 tests covering stage boundaries, command ranges, action authority, episode
lengths, orientation gating, vehicle speed, reward formulae, observation/action
dimensions, termination diagnostics, config field integrity, reset semantics,
and boundary conditions.

Pre-existing pipeline tests:

```bash
python kevin_integration/tasks/test_preflight_observation_contract.py
python kevin_integration/rom_cylinders/test_valve_conditioned_runtime.py
python kevin_integration/rom_cylinders/test_multirate_rom_wrapper.py
python kevin_integration/rom_cylinders/test_cylinder_coordinate_adapter.py
```
