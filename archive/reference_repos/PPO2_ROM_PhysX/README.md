# PPO2-ROM-PhysX

PPO2 is the four-cylinder ROM-only hydraulic arm controller. The deterministic
trajectory generator supplies the nominal cylinder target, and PPO applies a bounded
residual to all four arm cylinders while the truck is parked. 
```text
reachable tool reference
  -> four-cylinder seed
  -> 42-observation / 4-action PPO residual
  -> command-conditioned hydraulic ROM
  -> PhysX arm

PARK -> ARM_TRACK -> ARM_STOW -> STRAIGHT_DRIVE -> PARK
```

The policy cannot command the wheels, truck root, parking fixture, or station changes.
The deterministic supervisor enables PPO only in `ARM_TRACK`.

## Current PPO2 checkpoint

```text
logs/rsl_rl/ppo2_rom_residual_4cyl_joint/2026-07-20_21-47-18/model_167.pt
```

This is the project-selected PPO2 implementation. Its policy interface is:

- observation dimension: 42;
- action dimension: 4;
- cylinders 1–3 residual authority: at most ±2.00 mm;
- cylinder 4 residual authority in deployment: ±0.25 mm;
- tool-path speed request: 0.18 m/s;
- exact shapes: line, arc, sine, circle, and square.



## Visualize PPO2

From the repository root:

```bash
cd /home/rrlab/rrlab_mulag
./PPO2_ROM_PhysX/visualize.sh
```

The command loads `model_167.pt` and runs the complete five-station workflow:

1. park and lock the truck;
2. track the station trajectory with the four-cylinder PPO policy;
3. stow the arm;
4. drive the truck in a straight line to the next station;
5. park and repeat for line, arc, sine, circle, and square.

The Matplotlib window displays live reference-versus-measured mower-head tracking and
the truck route. The Isaac viewport displays only the cyan active reference; it does
not draw a measured-tool trail. The reference is rebuilt in world coordinates after
every relocation.

Metrics are written to:

```text
outputs/ppo2_rom_residual_4cyl_joint/model_167_stop_and_work_metrics.json
```

Additional supervisor arguments may be appended, for example:

```bash
./PPO2_ROM_PhysX/visualize.sh --stations 2
```

## Canonical artifacts

```text
Four-cylinder PPO2 policy:
logs/rsl_rl/ppo2_rom_residual_4cyl_joint/2026-07-20_21-47-18/model_167.pt

Hydraulic ROM:
ROM-cylinders_pipeline/checkpoints/plant_reachable_v1_finetune_blend25/best_rom.pt

Reachable trajectory library:
PPO2_ROM_PhysX/models/plant_reachable_trajectory_library_v1.json
```

The previous 41-observation/3-action actor is not a PPO2 deployment option anymore.
Its environment modules and checkpoint remain only as transfer/rollback dependencies
for the four-cylinder implementation.

## Exact trajectories

| Shape | Geometry | Effective speed |
|---|---|---:|
| Line | 2.50 m | 0.180 m/s |
| Arc | 1.80 m wide | 0.153 m/s |
| Sine | two cycles, 2.20 m × 0.50 m | 0.135 m/s |
| Circle | 0.80 m diameter | 0.126 m/s |
| Square | sharp 0.60 m × 0.60 m | 0.090 m/s |

Future paths must pass the measured plant workspace gate. The policy is locally
command-conditioned rather than shape-labelled, but PPO cannot make an unreachable
reference reachable.

## Policy contract

The 42 observations contain tool error and velocity, local reference velocity,
acceleration and preview, all four cylinder states and target errors, ROM command and
increment, previous residual actions, clearance, direction, head alignment, and head
alignment rate. There is no shape label.

The reward uses the original tool reference and penalizes tracking error, residual
magnitude and rate, measured tool acceleration, head deviation, clearance deviation,
and valve saturation. Cylinder 4 has an additional magnitude/rate penalty and stronger
filtering to prevent tool-tilt shortcuts.

Implementation and training details are in
[residual_policy_4cyl/README.md](residual_policy_4cyl/README.md).

## Verification

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python -m pytest -q \
  PPO2_ROM_PhysX/residual_policy_4cyl/tests \
  PPO2_ROM_PhysX/residual_policy/tests \
  PPO2_ROM_PhysX/tests \
  PPO2_ROM_PhysX/trajectories/tests
```

## Diagrams

- [Simple pipeline](docs/ppo2_rom_physx_pipeline_simple.png)
- [Detailed hydraulic pipeline](docs/ppo2_rom_physx_pipeline_detailed.png)
