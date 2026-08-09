# PPO1_A — nominal ID-FD V3 Policy

PPO1_A is the chosen nominal ID/FD V3 workflow. It keeps the learned
`ID(target) - ID(zero)` and `FD(valve) - FD(zero)` actuator backend while matching
the maintained ROM V3 policy, trajectory, logging, promotion, and park-work-relocate
contracts.

```text
reachable V3 tool reference
  -> nominal four-cylinder seed
  -> 42-observation / 4-action PPO Policy correction
  -> ID target/zero pair
  -> FD target/zero pair
  -> PhysX arm

PARK -> ARM_TRACK -> ARM_STOW -> STRAIGHT_DRIVE -> PARK
```

The deterministic supervisor owns the wheels and station changes. PPO acts only while
the truck is parked.

## PPO1_A V3 training contract

- 128 parallel environments.
- Four PPO cylinder outputs.
- Cylinders 1–3 residual authority: ±2.00 mm.
- Cylinder 4 residual authority: ±0.25 mm.
- Balanced line, arc, sine, circle, and square phases.
- V3 large plant-reachable trajectory library.
- Tool tracking, p95 tracking, p95 acceleration, alignment, drift, and residual logs.
- Finite safety events remain metrics and penalties during collection; non-finite state
  still terminates immediately.
- Per-environment promotion probe every two shape rounds.
- Automatic stop after two consecutive promotion passes or completion of 640 updates.

The previous three-action `model_1500.pt` is used only as checkpoint initialization for
cylinders 1–3; it is not a maintained workflow or deployment policy.
Cylinder 4 starts at exactly zero through the independent pitch branch. The optimizer
is fresh and all four outputs then train jointly.

## Train PPO1_A

```bash
cd /home/rrlab/rrlab_mulag
./PPO1_ID_FD_PhysX/train_v3.sh
```

Logs and checkpoints are written under:

```text
logs/rsl_rl/ppo1_a_policy_v3/<timestamp>/
```

The iteration output uses the same useful fields as ROM V3:

```text
tracking/mean_error_m
tracking/p95_error_m
smoothness/p95_tool_acceleration_mps2
safety/min_head_alignment
safety/max_truck_drift_m
policy/mean_abs_residual_m
policy/cylinder4_mean_abs_residual_m
```

## Evaluate a completed checkpoint

The completed 128-environment paired probes select `model_575.pt` as the best
balanced checkpoint. It satisfied completion, acceleration, per-shape regression,
alignment, and truck-drift safety checks. Against its matched zero-Policy baseline:

| Metric | Baseline | `model_575.pt` |
|---|---:|---:|
| Mean tool error | 0.02837 m | 0.02829 m |
| P95 tool error | 0.04982 m | 0.04950 m |
| P95 tool acceleration | 31.18 m/s² | 30.97 m/s² |
| Minimum paired completion | — | 95.31% |

The gain is modest and does not meet the deliberately strict 2% automatic-promotion
threshold, but later checkpoints lost completion consistency. Therefore model 575 is
the visualization candidate; it is not being claimed as a large tracking breakthrough.

```bash
cd /home/rrlab/rrlab_mulag
CANDIDATE=logs/rsl_rl/id_fd_residual_v3/2026-07-21_20-56-31/model_575.pt

/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  PPO1_ID_FD_PhysX/policy/scripts/evaluate_v3.py \
  --headless --device cuda:0 --checkpoint "$CANDIDATE" \
  --num_envs 128 --scored_steps 120 \
  --out outputs/id_fd_policy_v3/model_575_evaluation.json
```

Evaluation requires completed cycles, improved mean and p95 tracking, bounded p95 tool
acceleration, at least 0.98 head alignment, and less than 2 mm retained truck drift.
By default the command writes readings even if the promotion gate fails. Add
`--strict_gate` when a failed gate should return exit code 2.

## Visualize park-work-relocate

The launcher now defaults to the selected checkpoint:

```bash
cd /home/rrlab/rrlab_mulag
./PPO1_ID_FD_PhysX/visualize_v3.sh
```

To inspect another compatible checkpoint, pass it as the first argument.

This executes line → arc → sine → circle → square over five stations with live tool
and truck plots. Each completed station reports mean error, p95 error, and p95 tool
acceleration. The Isaac viewport draws the current reference trajectory in cyan only;
the measured orange tool trail remains confined to the Matplotlib tracking plot.

The immutable route begins at the settled station-zero chassis pose—the same physical
state from which the first arm trajectory starts. ARM_TRACK and ARM_STOW use the original
trained plant contract: the wheel fixture and chassis correction run once per policy-rate
interval and articulation velocity is not overwritten. PPO is disabled only for DRIVE,
where a mechanical transport fixture pins the settled arm rest pose while the chassis
follows a straight quintic profile with zero endpoint velocity and acceleration. The fixture
holds all five commanded arm joints, all ten passive hydraulic linkage/constraint joints,
and one settled articulation-root pose. A single root reference is translated along the
route; suspension measurements are not fed back into arm orientation. It is never active
during ARM_STOW, PARK, or ARM_TRACK. PARK removes transport momentum once, latches the new chassis
pose, completes normal arm preposition/settling, and performs one final reset-style station
latch before PPO resumes. Because nominal ID-FD reactions are stronger than ROM reactions,
PPO1 additionally anchors the actual chassis heading and renders only after the parked-pose
correction; the arm-slew articulation root is never mistaken for the truck frame.

The same isolation contract is used by PPO1_A and PPO2_A. A normal-duration PPO1_A
line → stow → 1 m drive → park → arc regression run completed with 0.0209/0.0302 m
line mean/p95 error and 0.0203/0.0322 m arc mean/p95 error; maximum transport path
error was 0.004 mm. No Policy retraining is required for this supervisor repair.
The transport fixture regression gate requires the complete 15-joint arm chain to remain
fixed, at least 20 mm tool clearance, no more than 1 mm world-Z or truck-relative tool
motion, and straight-route accuracy during DRIVE. After a complete 29.74 s line cycle,
the latest 2 m transport regression measured 0.0 mm route error, 0.010° heading error,
zero arm-chain displacement, and 0.0 mm tool motion in both frames. The following arc
entered ARM_TRACK normally with less than 0.2 mm truck drift.

## Canonical inputs

```text
Compatibility initializer (not a deployable PPO1_A policy):
logs/rsl_rl/id_fd_residual/2026-07-15_16-24-23/model_1500.pt

ID model:
PPO1_ID_FD_PhysX/models/id/best_id_cylinder_control.pt

FD model:
PPO1_ID_FD_PhysX/models/fd/best_fd_cylinder_strict.pt

V3 trajectory library:
PPO2_ROM_PhysX/models/plant_reachable_trajectory_library_v3_large.json
```

The ID and FD models do not need retraining for this policy update. Their training and
artifact lineage remain documented in [TRAINING.md](TRAINING.md).

## Verification completed before long training

- All five zero-residual V3 shape preflights passed at 128 environments.
- Four-action checkpoint transfer passed with an exactly zero cylinder-4 output.
- One line and one arc PPO update completed at roughly 535–570 steps/s.
- The deployed line smoke test completed without termination and reported smoothness
  metrics.

## Diagrams

- [Simple ID-FD pipeline](docs/images/id_fd_pipeline_simple.png)
- [Advanced ID-FD pipeline](docs/images/id_fd_pipeline_advanced.png)
