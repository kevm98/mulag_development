# PPO2_A — nominal ROM V3 Policy

PPO2_A is the chosen nominal ROM V3 workflow. It replaces the complete
runtime ID/FD actuator pair with one command-conditioned hydraulic ROM and applies a
bounded PPO Policy correction to all four arm cylinders. The policy acts only while the truck
is parked.

```text
reachable tool reference
  -> four-cylinder trajectory seed
  -> 42-observation / 4-action PPO Policy correction
  -> command-conditioned hydraulic ROM
  -> PhysX arm

PARK -> ARM_TRACK -> ARM_STOW -> STRAIGHT_DRIVE -> PARK
```

The deterministic supervisor owns parking, wheel commands, straight relocation, and
station changes. PPO cannot move the truck and is enabled only in `ARM_TRACK`.

## Canonical V3 artifacts

```text
Selected smooth PPO2 policy:
logs/rsl_rl/ppo2_rom_residual_4cyl_joint/2026-07-21_02-09-15/model_607.pt

Hydraulic command ROM:
ROM-cylinders_pipeline/checkpoints/plant_reachable_v1_finetune_blend25/best_rom.pt

Reachable V3 trajectory library:
PPO2_ROM_PhysX/models/plant_reachable_trajectory_library_v3_large.json
```

The selected policy has 42 observations and four actions. Cylinders 1–3 have at most
±2.00 mm residual authority; cylinder 4 has ±0.25 mm in deployment and stronger
filtering. The observation contains no trajectory-shape label, so the policy responds
to reference, plant, cylinder, ROM, clearance, and head-alignment state rather than a
named shape.

## Main V3 workflow

Run the complete GUI workflow:

```bash
cd /home/rrlab/rrlab_mulag
./PPO2_ROM_PhysX/visualize.sh
```

This executes park → track → stow → straight drive → park for line, arc, sine,
circle, and square. Matplotlib shows live tool and truck tracking. Isaac Sim draws
only the cyan reference trajectory. Each station reports mean error, p95 error, and
p95 tool acceleration.

The immutable route begins at the settled station-zero chassis pose. ARM_TRACK and
ARM_STOW preserve the exact plant timing used to train the checkpoint: chassis correction
runs once per policy-rate interval and root velocity is not rewritten. PPO is disabled
during DRIVE; only there does a mechanical transport fixture pin the settled arm rest pose
while the chassis follows a straight quintic pose profile. The fixture is removed before
PARK, ROM warm-up, preposition, and ARM_TRACK.
PARK removes transport momentum once, latches the new station, performs the normal ROM
warm-up and arm preposition, and then applies one reset-style station latch before PPO
resumes. Normal Isaac rendering cadence is retained, so the visualizer no longer waits on
suppressed render intervals.

The repaired two-station regression matched a standalone arc after transport: mean/p95
error 0.0201/0.0351 m, acceleration p95 28.13 m/s², and terminal chassis speed
0.0147 m/s. Before the station latch those values were 0.0258/0.0522 m,
45.89 m/s², and 0.1004 m/s. A five-station six-second-per-shape run completed all
line, arc, sine, circle, and square segments with no termination and at most 1.36 mm
sampled chassis drift. Metrics are in
[`../outputs/ppo2_a_policy_v3/regression_recovery_all_shapes.json`](../outputs/ppo2_a_policy_v3/regression_recovery_all_shapes.json).
The transport-specific regression measured zero arm-joint displacement, 3.8 mm maximum
world-Z suspension excursion, 0.295 m minimum clearance, and 0.006 mm route error. Arm
motion is gated directly by joint displacement; chassis-frame and world-frame tool motion
remain diagnostics because both include suspension movement. See
[`../outputs/ppo2_a_policy_v3/transport_mechanical_fixture_tracking8s.json`](../outputs/ppo2_a_policy_v3/transport_mechanical_fixture_tracking8s.json).

Run fewer stations when checking a change:

```bash
./PPO2_ROM_PhysX/visualize.sh --stations 2
```

Workflow metrics are written to:

```text
outputs/ppo2_rom_residual_4cyl_joint/model_607_v3_smooth_stop_and_work_metrics.json
```

## Train the V3 policy

The canonical launcher uses 128 environments, all four PPO actions, the V3 large
trajectory library, per-environment promotion probes, and automatic termination at
the configured training limit or promotion criterion.

```bash
cd /home/rrlab/rrlab_mulag
./PPO2_ROM_PhysX/train_large_v3.sh
```

Training outputs are stored under:

```text
logs/rsl_rl/ppo2_a_policy_v3/
```

## V3 trajectories

| Shape | Geometry | Effective speed |
|---|---|---:|
| Line | 2.50 m | 0.180 m/s |
| Arc | 2.20 × 0.50 m | 0.153 m/s |
| Sine | two cycles, 3.20 × 0.90 m | 0.135 m/s |
| Circle | 1.10 m diameter | 0.126 m/s |
| Square | sharp 0.90 × 0.90 m | 0.090 m/s |

Future trajectories must first pass the measured plant-workspace and full-cycle ROM
gates. PPO can correct reachable local tracking error but cannot make an unreachable
reference reachable.

## Verify the V3 contract

```bash
cd /home/rrlab/rrlab_mulag
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python -m pytest -q \
  PPO2_ROM_PhysX/policy_4cyl/tests \
  PPO2_ROM_PhysX/policy/tests \
  PPO2_ROM_PhysX/tests \
  PPO2_ROM_PhysX/trajectories/tests
```

## Diagrams

- [Simple pipeline](docs/images/ppo2_rom_physx_pipeline_simple.png)
- [Advanced hydraulic pipeline](docs/images/ppo2_rom_physx_pipeline_advanced.png)
