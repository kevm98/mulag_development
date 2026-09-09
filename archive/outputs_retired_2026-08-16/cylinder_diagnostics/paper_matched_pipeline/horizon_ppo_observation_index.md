# Horizon PPO Observation Index

Observation dimension: `62`

All TCP task-space position fields are in the Unimog/base frame. The EE
quaternion is the normalized world quaternion.

| Slice | Field | Size |
|---|---|---:|
| `0:3` | `ee_pos_b` | 3 |
| `3:7` | `ee_quat_w` | 4 |
| `7:10` | `target_pos_b` | 3 |
| `10:13` | `ee_pos_error_b` | 3 |
| `13:18` | `joint_pos` | 5 |
| `18:23` | `joint_vel` | 5 |
| `23:27` | `cylinder_ext` | 4 |
| `27:31` | `cylinder_vel` | 4 |
| `31:35` | `prev_ppo_action` | 4 |
| `35:39` | `prev_delta_x_des_H` | 4 |
| `39:43` | `prev_delta_x_fd_H` | 4 |
| `43:47` | `prev_delta_x_cmd_H` | 4 |
| `47:51` | `prev_delta_x_meas_H` | 4 |
| `51:55` | `prev_guard_flags` | 4 |
| `55:59` | `unimog_pose` | 4 |
| `59:62` | `unimog_pose_dot` | 3 |

Excluded by design:

- camera images
- height maps
- ray sensors
- AAM latent features
- FD load estimates
- teacher-student manager action state
