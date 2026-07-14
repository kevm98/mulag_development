# MULAG ID-FD head-down arm controller

This directory contains the maintained deterministic hydraulic-arm workflow and the residual
PPO task built above it. The truck is parked, the cutter follows a long ground-level arc from
A to B and back to A, and the mower head remains facing down. The policy may add only small
cylinder 1-3 setpoint corrections; it cannot command cylinder 4 or the vehicle.

## Validated behavior

The acceptance run uses the active ID/FD pair, the head-down reference arc, a 0.13 m/s cruise
speed, cosine acceleration and turnaround, and one complete A->B->A cycle.

| Metric | Result |
|---|---:|
| Result | PASS |
| One-way arc | 4.067 m |
| Complete A->B->A reference | 8.133 m |
| Mean tool error | 10.6 mm |
| p95 tool error | 22.2 mm |
| Final tool error | 12.9 mm |
| Ground-band occupancy | 100% |
| Minimum head-down alignment | 0.999902 |
| Maximum head tilt from world down | 0.80 deg |
| Final truck planar drift | 0.66 mm |

The verified arc is about 81 degrees, not a literal 180-degree semicircle. It is the long
continuous arc that is executable, holdable, ground-level, and compatible with the head-down
constraint on the current plant.

## Run it

From the repository root:

```bash
# Headless acceptance gate
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  ID_FD_pipline/dls_solver/tests/ground_arc_tracking_gate.py \
  --headless --speed_mps 0.13 --cycles 1

# IsaacLab GUI
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  ID_FD_pipline/dls_solver/tests/ground_arc_tracking_gate.py \
  --speed_mps 0.13 --cycles 1 --realtime
```

GUI markers are green for the full reference, yellow for A, magenta for B, red for the live
reference, and blue for the measured cutter point.

## Current control path

```text
truck-frame cutter reference
        |
        v
head-down cylinder teacher + velocity-scaled seed lead
        |
        v
EE cascade (cylinders 1-3 control position; cylinder 4 is reserved for head-down)
        |
        v
cylinder DLS with measured cylinder feedback
        |
        v
ID(target) - ID(zero target) -> paired valve response
        |
        v
FD(valve) - FD(zero-target valve) -> learned cylinder residual
        |
        v
0.8 requested cylinder increment + 0.2 learned ID-FD residual
        |
        v
goal-anchored PhysX cylinder position target
```

Important runtime settings:

- Active ID: `models/id/best_id_cylinder_control.pt`
- Active FD: `models/fd/best_fd_cylinder_strict.pt`
- Learned-model sample time: 0.05 s
- Reference blend: 0.8 direct reference / 0.2 learned residual
- Paired residual: `FD(ID(target)) - FD(ID(zero target))`
- Residual gains: `[1, 1, 1, 1]`
- Cylinder-target lead limit: 25 mm
- DLS action smoothing: 0.5
- Arc seed lead: 0.25 s scaled by current path velocity
- Head-down cylinder-4 reference: 0.588991 m
- Minimum permitted head-down alignment: 0.95
- Truck drive disabled; wheel positions held; planar fixture has a 1 mm deadband

The cutter position is measured in the truck chassis (`base_link`) frame. The articulation
root is the arm slew base and must not be used as the truck reference frame. The controlled
point includes the measured `-0.79 m` mower-tip offset from the `Messerkopf` body origin.

## Acceptance checks

```bash
# Learned pipeline rank and wiring
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  ID_FD_pipline/tests/rank_gate.py --headless --num_envs 1

# Cylinder-space ID-FD step, return, and smoothness gates
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  ID_FD_pipline/dls_solver/tests/idfd_step_gate.py --headless

# Final tool-space, ground, orientation, and parked-truck gate
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  ID_FD_pipline/dls_solver/tests/ground_arc_tracking_gate.py \
  --headless --speed_mps 0.13 --cycles 1

# Pure trajectory time-scaling tests
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python -m pytest \
  ID_FD_pipline/trajectories/tests/test_time_scaling.py -q

# Residual policy: exact zero-action baseline contract
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  ID_FD_pipline/residual_policy/tests/zero_action_contract.py \
  --headless --num_envs 1
```

Residual PPO training, playback, and checkpoint evaluation are documented in
[residual_policy/README.md](residual_policy/README.md).

## Directory layout

```text
ID_FD_pipline/
├── README.md                         current runtime and commands
├── TRAINING.md                       ID/FD training and fine-tuning provenance
├── dls_solver/
│   ├── cylinder.py                   learned composite inversion and cylinder tracking
│   ├── ee_cascade.py                 cutter-to-cylinder cascade
│   ├── kinematics.py                 stepped calibration helpers
│   └── tests/
│       ├── ground_arc_tracking_gate.py
│       └── idfd_step_gate.py
├── mdp/                              observations, rewards, and action sanitization
├── models/
│   ├── ee_jacobian_ground_truck3x4.json
│   ├── mowing_arc_head_down.json
│   ├── id/                           active control ID and source data
│   └── fd/                           FD lineage, strict data, reports, active strict FD
├── residual_policy/                  3-D residual PPO task, scripts, and contracts
├── scripts/                          retained model/data/reference builders
├── tasks/                            IsaacLab environments and active configuration
├── tests/rank_gate.py                learned-pipeline structural gate
├── training_data/                    strict four-run open-loop excitation export
└── trajectories/time_scaling.py      smooth cruise and turnaround profiles
```

See [TRAINING.md](TRAINING.md) for the complete model lineage, dataset contract, objectives,
validation results, rejected approaches, and reproduction commands.

`dls_solver/` and `mdp/` are active dependencies, not cleanup leftovers. The residual
environment calls the DLS cylinder/cascade controllers every scored step and inherits the
ID/FD task environment, which imports MDP action sanitization, observation/reward contracts,
and pipeline reporting used by the retained deterministic gates.
