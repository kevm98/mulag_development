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
