# Canonical hydraulic ROM for PPO2 V3

This folder contains the command-conditioned, hydraulic-aware reduced-order cylinder
model used by the canonical PPO2-ROM-PhysX V3 workflow. The deployed ROM operates only
in cylinder space and replaces the complete runtime ID/FD actuator pair.

It does not consume joint positions, truck state, trajectory names, or PPO state.

## V3 runtime path

```text
reachable V3 tool reference
  -> nominal four-cylinder command
  -> four-action PPO residual
  -> valve command
  -> hydraulic cylinder ROM
  -> PhysX arm
```

The ROM uses six contiguous 50 ms history rows. Each row contains extension,
extension velocity, pressure difference, previous predicted valve, desired extension
delta, and desired extension velocity for all four cylinders. It predicts valve,
pressure delta, velocity delta, and extension delta. One ROM prediction is distributed
over three 60 Hz control steps.

## Canonical V3 artifacts

```text
Selected hydraulic ROM:
ROM-cylinders_pipeline/checkpoints/plant_reachable_v1_finetune_blend25/best_rom.pt

Selected smooth PPO policy:
logs/rsl_rl/ppo2_rom_residual_4cyl_joint/2026-07-21_02-09-15/model_607.pt

V3 plant-reachable trajectory library:
PPO2_ROM_PhysX/models/plant_reachable_trajectory_library_v3_large.json
```

The ROM checkpoint retains the existing filename because it is the physically
validated actuator model. V3 designates the canonical integrated workflow, trajectory
library, four-cylinder policy, and deployment supervisor.

## Run the main V3 workflow

```bash
cd /home/rrlab/rrlab_mulag
./PPO2_ROM_PhysX/visualize.sh
```

This runs park → arm track → stow → straight drive → next station for line, arc,
sine, circle, and square. The truck is locked during arm work, PPO cannot command the
wheels, and Isaac Sim draws only the cyan reference trajectory.

## Train the V3 policy with this ROM

```bash
cd /home/rrlab/rrlab_mulag
./PPO2_ROM_PhysX/train_large_v3.sh
```

## Verify the ROM and V3 integration

```bash
cd /home/rrlab/rrlab_mulag
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python -m pytest -q \
  ROM-cylinders_pipeline/tests \
  PPO2_ROM_PhysX/tests/test_command_rom_runtime.py \
  PPO2_ROM_PhysX/policy_4cyl/tests \
  PPO2_ROM_PhysX/trajectories/tests
```

## Dataset contract

- Cylinder space only.
- Six contiguous history samples per transition.
- Histories never cross run boundaries.
- Complete runs stay wholly within train, validation, or test.
- Measured valves may supervise training but are not runtime inputs.
- Zero command must preserve the stationary hold contract.
- A ROM is promoted only after full-cycle PhysX validation; offline loss alone is not
  an acceptance criterion.

See [PPO2-ROM-PhysX V3](../PPO2_ROM_PhysX/README.md) for policy observations,
trajectory dimensions, metrics, and deployment details.
