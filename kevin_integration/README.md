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

Visible scripted full-pipeline demo, no checkpoint:

```bash
./rrlab.sh -p kevin_integration/scripts/play_mulag_adaptive_rl.py
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

The default play command runs `control_mode=scripted_reference_demo`. With a
checkpoint, play switches to `control_mode=rl_policy`.

The train script defaults to `control_mode=rl_policy` and aborts before PPO if
the asset transmission preflight or learned-pipeline preflight fails. Use
`--skip_motion_preflight` only for deliberate debugging.

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

## Runtime Reports

Static and runtime contract reports are under `kevin_integration/results/`:

- `static_signal_chain_audit.md`
- `runtime_valve_command_contract.md`
- `runtime_rom_motion_contract.md`
- `prismatic_drive_authority.md`
- `full_pipeline_preflight_contract.md`

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
- SD feedback is disabled for the active task.
- dP uses the final ROM training mean until a verified environment pressure
  source is available.
- USD-authored joint axes, masses, inertias, and drive APIs still require
  Isaac/USD introspection in an environment with `pxr` or Isaac runtime tools.
- Do not claim full runtime success until the scripted demo has visibly shown
  cylinder motion, linkage transmission, and acceptable vehicle/base behavior in
  Isaac.
