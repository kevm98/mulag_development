# PPO Action Reference Alignment

Date: 2026-06-11

## Architecture

`RRLAB-Obstacle-Avoidance-Mulag-v0` supplies `ee_pose` to the policy and uses PPO as the task-space-to-action bridge. There is no IK controller in that task.

Kevin follows the same high-level contract:

```text
desired Messerkopf ee_pose
+ measured robot state
-> PPO Action[4]
-> desired hydraulic-family qddot[4]
-> AAM
-> ID
-> ValveCmd[4]
-> valve-conditioned causal ROM
-> prismatic cylinder targets
-> linkage
-> measured Messerkopf motion
```

No-checkpoint playback remains `joint_space_validation`: it validates AAM -> ID -> ROM -> cylinders -> linkage with a mechanical joint reference and makes no task-space tracking claim.

## PPO Action

Dimension: `4`.

Meaning: normalized desired qddot command for the four hydraulic families. It is not wheel speed, Cartesian displacement, ValveCmd, cylinder extension, torque, or effort.

The ID qddot scale comes from the ID checkpoint scaler entries for `qddot1..qddot4`.

## Task-Space Observation

Active PPO task-space observation order is 73D:

| Feature | Dim |
|---|---:|
| `ee_pose_command` | 7 |
| `current_ee_pose` | 7 |
| `ee_position_error` | 3 |
| `ee_orientation_error` | 1 |
| `mechanical_q` | 5 |
| `mechanical_qdot` | 5 |
| `cylinder_q` | 4 |
| `cylinder_qdot` | 4 |
| `previous_action` | 4 |
| `previous_ValveCmd` | 4 |
| `z_arm_hat` | 24 |
| `fd_tau_hat` | 4 |
| `vehicle_forward_speed_kmh` | 1 |

Joint-space validation keeps the legacy 62D observation because it is a scripted actuator-chain check, not PPO task-space training.

## Rewards

`rl_task_space_tracking` uses:

- end-effector position tracking: `exp(-4 * ||ee_position_error||)`
- end-effector orientation tracking: `0.25 * exp(-2 * ee_orientation_error)`
- action-rate penalty: `rew_action_rate = 0.05`
- action-magnitude penalty: `rew_action_magnitude = 0.02`
- ValveCmd saturation penalty: `rew_action_saturation = 0.5`
- cylinder target clamp penalty: scale `1.0`
- FD predicted-load penalty: `rew_fd_load = 0.01`
- unsafe-state penalty: `rew_termination = 5.0`

Joint-space reference rewards are retained only for `joint_space_validation`.

## FD Role

FD runs as an observer/load estimator. It adds `fd_tau_hat[4]` to PPO observations, reward telemetry, and safety logs. It does not write efforts while ROM position-target actuation is active.

Hard guard:

```text
ROM position-target actuation and FD effort actuation cannot command the same hydraulic joints simultaneously.
```

## Vehicle

Vehicle speed is scripted independently:

```text
speed_kmh ~ Uniform(3, 8)
wheel_velocity_rad_s = (speed_kmh / 3.6) / 0.575
```

Kevin samples the speed once per environment reset and all four wheel joints
receive the same positive velocity target. PPO never controls wheel velocity.

## Curriculum (Learnability Repair)

Four-stage curriculum based on total environment steps:

| Stage | Steps         | Ep. Length (steps) | Action Authority | Vehicle          | Position Range          | Orientation |
|-------|---------------|---------------------|------------------|------------------|--------------------------|-------------|
| S1    | 0 – 50k       | 1500 (15 s)         | 0.25             | stationary       | tight around nominal     | disabled    |
| S2    | 50k – 150k    | 1500 (15 s)         | 0.50             | stationary       | moderate                 | disabled    |
| S3    | 150k – 300k   | 3000 (30 s)         | 0.75             | stationary       | wide                     | enabled     |
| S4    | 300k+         | 4500 (45 s)         | 1.00             | 3 – 8 km/h       | full                     | enabled     |

**Action authority**: PPO action is multiplied by the stage-dependent scalar before qddot conversion in `_prepare_high_level_action()`. This reduces effective action magnitude in early stages, preventing unstable early exploration from saturating valves or violating joint limits.

**Command ranges** (`_sample_task_space_command()`): x/y/z command sampling uses `curriculum_command_{x,y,z}_range[stage-1]`. S1 starts with tiny volumes near nominal; S4 expands to the full reachable workspace.

**Orientation**: S1–S2 set a flat-facing identity-like quaternion `[0,0,1,0]` (no orientation reward). S3–S4 set the identity quaternion `[1,0,0,0]` and enable orientation tracking reward.

**Vehicle**: S1–S3 wheel velocity is zeroed (stationary). S4 samples `speed_kmh ~ Uniform(3, 8)` matching the original uncurriculed behavior.

**Episode length**: Stage-aware timeout via `_get_dones()` multiplies `curriculum_episode_length_s[stage-1] / dt`. Early stages are shorter to collect more reset events per wall-clock hour and avoid wasted simulation on a policy that has not yet stabilised.
