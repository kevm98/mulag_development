# RRLAB Task-Space Reference Trace

Date: 2026-06-11

Scope: source trace for `RRLAB-Obstacle-Avoidance-Mulag-v0`. This is a static audit only; IsaacLab was not launched.

## Registration

- File: `extensions/rrlab_tasks/rrlab_tasks/manager_based/manipulation/obstacle_avoidance/config/mulag/__init__.py`
- Gym ID: `RRLAB-Obstacle-Avoidance-Mulag-v0`
- Entry point: `isaaclab.envs:ManagerBasedRLEnv`
- Env cfg: `MulagObstacleAvoidanceEnvCfg`

## End Effector

- Body name: `Messerkopf`
- Config path: `config/mulag/joint_pos_env_cfg.py`
- Assignment: `self.commands.ee_pose.body_name = "Messerkopf"`
- Body index resolution: `CustomUniformPoseCommand.__init__()` calls `self.robot.find_bodies(cfg.body_name)[0][0]`.

## Reference Frame And Pose Ordering

- Command name: `ee_pose`
- Source class: `mdp/commands.py::CustomUniformPoseCommand`
- Command tensor: `pose_command_b`
- Shape/order: `[x, y, z, qw, qx, qy, qz]`
- Position units: meters
- Quaternion convention: IsaacLab WXYZ
- Frame exposed to policy/reward: robot base frame
- World debug target: `target_pos_w`, `target_quat_w`

`CustomUniformPoseCommand._update_command()` computes:

1. robot root pose in world;
2. robot x in the environment frame;
3. target x as `robot_x_env + sampled_x_offset`;
4. fixed sampled y, z, roll, pitch, yaw;
5. target pose in world by adding the environment origin;
6. command in robot base frame with `subtract_frame_transforms`.

## Trajectory Generator

- File: `obstacle_avoidance_env_cfg.py`
- Class: `CommandsCfg`
- Config: `CustomUniformPoseCommandCfg`
- Resampling time: `(10.0, 12.0)` seconds
- Ranges:
  - x offset: `(6.5, 7.5)` m
  - y: `(0.7, 2.2)` m
  - z: `(0.5, 0.5)` m
  - roll: `(0.0, 0.0)` rad
  - pitch: set by Mulag cfg to `(pi, pi)`
  - yaw: `(pi, pi)`

The target is piecewise sampled, not a smooth analytic IK trajectory. It tracks forward with the vehicle through the x update.

## Update Frequency

- Obstacle env `__post_init__`: `sim.dt = 1 / 120`, `decimation = 2`
- Policy/control step: `1 / 60` s
- Command manager updates every environment step and resamples according to the 10-12 s range.

## Vehicle Motion

- File: `mdp/randomization.py`
- Class: `random_driving_speed`
- Wheel radius: `0.575` m
- Speed range: `(3.0, 8.0)` km/h
- Conversion: `(speed_kmh / 3.6) / 0.575`
- Writer: `asset.write_joint_velocity_to_sim(...)`
- Joints:
  - `Wheel_Front_Left_joint`
  - `Wheel_Front_Right_joint`
  - `Wheel_Rear_Left_joint`
  - `Wheel_Rear_Right_joint`
- Timing: interval event at `(0.1, 0.1)` s; resamples only while `episode_length_buf <= 10`, then reuses the stored wheel targets.
- Steering: no active steering command found in the traced task.
- Wheel signs: all four receive the same positive angular velocity in source.

## Action Term

- File: `config/mulag/joint_pos_env_cfg.py`
- Action: `mdp.JointPositionActionCfg`
- Joint order:
  - `Drehzapfen_joint`
  - `Ausleger_I_joint`
  - `Ausleger_II_joint`
  - `Messerkopf_Schwenk_joint`
  - `Messerkopf_joint`
- `use_default_offset=True`

## Rewards

The obstacle task rewards task-space tracking against `ee_pose`, plus terrain/contact/workspace penalties.

Important pose-tracking terms in `obstacle_avoidance_env_cfg.py` and `mdp/rewards.py`:

- `position_command_error`
- `position_command_error_tanh`
- `orientation_command_error_yaw`
- `orientation_command_error_yaw_tanh`
- `orientation_command_error_pitch`
- `orientation_command_error_pitch_tanh`

The position reward combines the base-frame command with the root pose to get desired world position, then compares `Messerkopf` body world position.

## IK/Controller Search Result

No reusable IK implementation was found in the RRLAB obstacle task source.

Search terms included:

- `DifferentialIKController`
- `differential_ik`
- `inverse_kinematics`
- `jacobian`
- body/joint Jacobian APIs

The traced RRLAB obstacle task is model-free at the arm-controller level: it observes/rewards the task-space target and PPO emits joint-position actions. There is no source IK bridge because PPO is the intended bridge from `ee_pose` to arm action.

## Kevin Implication

Kevin should not implement:

```text
ee_pose -> IK -> mechanical q_ref
```

for this architecture. Kevin should expose `ee_pose` in PPO observations/rewards and let PPO output `Action[4]` upstream of AAM/ID/ROM.
