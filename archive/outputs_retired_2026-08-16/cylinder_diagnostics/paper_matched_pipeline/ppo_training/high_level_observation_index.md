# High-Level PPO Observation Index

## Architecture

PPO observes task-space robot/EE/goal state only. Cylinder internals, ID/FD
variables, valve commands, and delta_x values are excluded.

## Observation (30D)

| Name | Indices | Dim | Description |
|------|---------|-----|-------------|
| ee_pos_b | [0:3] | 3 | EE position in vehicle base frame |
| target_pos_b | [3:6] | 3 | Command target in base frame |
| ee_pos_error_b | [6:9] | 3 | target - ee (base frame) |
| ee_error_norm | [9:10] | 1 | Euclidean distance to target |
| joint_pos | [10:15] | 5 | Arm joint positions (rad) |
| joint_vel | [15:20] | 5 | Arm joint velocities (rad/s) |
| prev_hl_action | [20:23] | 3 | Previous PPO action (EE delta) |
| unimog_pose | [23:27] | 4 | Base x, y, sin(yaw), cos(yaw) |
| unimog_pose_dot | [27:30] | 3 | Base vx, vy, omega_z |

## Excluded (Legacy 62D obs)

These are now internal to the low-level controller:

- cylinder_ext[4] - cylinder extension (simscape)
- cylinder_vel[4] - cylinder velocity
- prev_delta_x_des_H[4] - previous desired cylinder delta
- prev_delta_x_fd_H[4] - FD-predicted delta
- prev_delta_x_cmd_H[4] - commanded delta
- prev_delta_x_meas_H[4] - measured delta
- prev_guard_flags[4] - guard activation flags
- ee_quat_w[4] - EE quaternion (world)

## Frame Consistency

All position vectors are in the vehicle base frame:
- ee_pos_b = quat_apply_inverse(root_quat_w, ee_world - root_world)
- target_pos_b = _command_pos_w[:3] (stored in base frame)
- ee_pos_error_b = target_pos_b - ee_pos_b

No world/base frame mixing.
