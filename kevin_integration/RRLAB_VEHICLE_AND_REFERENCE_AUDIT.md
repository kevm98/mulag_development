# RRLAB Vehicle And Reference Audit

Date: 2026-06-11

Scope: source-only audit of the working RRLAB Mulag obstacle task and Kevin hydraulic ROM task. No IsaacLab, Conda, USD, PPO training, ROM weights, AAM/ID models, rewards, or coordinate mapping were run or changed.

## Source Files

- `extensions/rrlab_tasks/rrlab_tasks/manager_based/manipulation/obstacle_avoidance/config/mulag/__init__.py`
- `extensions/rrlab_tasks/rrlab_tasks/manager_based/manipulation/obstacle_avoidance/config/mulag/joint_pos_env_cfg.py`
- `extensions/rrlab_tasks/rrlab_tasks/manager_based/manipulation/obstacle_avoidance/obstacle_avoidance_env_cfg.py`
- `extensions/rrlab_tasks/rrlab_tasks/manager_based/manipulation/obstacle_avoidance/mdp/commands.py`
- `extensions/rrlab_tasks/rrlab_tasks/manager_based/manipulation/obstacle_avoidance/mdp/randomization.py`
- `extensions/rrlab_tasks/rrlab_tasks/manager_based/manipulation/obstacle_avoidance/mdp/rewards.py`
- `kevin_integration/tasks/mulag_adaptive_rl_env.py`
- `kevin_integration/tasks/mulag_adaptive_rl_env_cfg.py`

## Findings

1. `RRLAB-Obstacle-Avoidance-Mulag-v0` is registered as a `ManagerBasedRLEnv` using `MulagObstacleAvoidanceEnvCfg`.

2. The active Mulag obstacle arm action is an IsaacLab `JointPositionActionCfg` named `arm_action`.

3. The obstacle policy action controls five arm joints:
   `Drehzapfen_joint`, `Ausleger_I_joint`, `Ausleger_II_joint`, `Messerkopf_Schwenk_joint`, `Messerkopf_joint`.

4. The obstacle task does command vehicle forward motion, but not through PPO. `EventCfg.randomize_driving_speed` calls `mdp.random_driving_speed` every 0.1 s.

5. The vehicle command writes velocity targets to:
   `Wheel_Front_Left_joint`, `Wheel_Front_Right_joint`, `Wheel_Rear_Left_joint`, `Wheel_Rear_Right_joint`.

6. The speed is variable scripted. It samples `3.0` to `8.0` km/h while `episode_length_buf <= 10`, then reuses the stored target.

7. The wheel conversion is:
   `omega_rad_s = (speed_kmh / 3.6) / 0.575`.
   This gives approximately `1.4492753623` rad/s at 3 km/h and `3.8647342995` rad/s at 8 km/h.

8. No active obstacle task source was found that gives PPO a vehicle speed, steering, or root-velocity action.

9. The obstacle reference is task space, not joint space. `CommandsCfg.ee_pose` is a `CustomUniformPoseCommandCfg` with body `Messerkopf`.

10. The task-space command ranges are:
    x offset `6.5` to `7.5` m, y `0.7` to `2.2` m, z `0.5` m, roll `0`, pitch `pi`, yaw `pi`, resampling time `10` to `12` s.

11. `CustomUniformPoseCommand._update_command()` tracks the moving vehicle in x. It computes target x as current Unimog x in the environment frame plus the sampled x offset, then converts the target pose into the robot base frame.

12. Obstacle rewards consume the task-space command by comparing the `Messerkopf` body pose against `ee_pose`, especially position, yaw, and pitch tracking terms.

13. Kevin currently uses a 4D normalized desired-qddot action, then ID produces `ValveCmd[4]`, then the valve-conditioned ROM writes four cylinder position targets.

14. Kevin observations and rewards are joint-space: `q`, `qdot`, `q_ref`, `qdot_ref`, q errors, previous action/valve, and `z_arm_hat`.

15. The obstacle task-space reference is now exposed to Kevin PPO observations
    and rewards. PPO is the controller bridge: it maps `ee_pose` plus measured
    robot state to Kevin `Action[4]`, which then enters AAM/ID/ROM as desired
    hydraulic-family qddot.

## Kevin Changes Made From Proven Source

- Added `kevin_integration/tasks/rrlab_reference_conventions.py` with the obstacle task joint order, wheel names, pose-command ranges, and speed conversion constants.
- Updated Kevin config to use the obstacle arm joint order from that convention table.
- Added `vehicle_command_mode = "rrlab_obstacle_scripted_wheel_velocity"`.
- Added Kevin wheel velocity command logic separate from PPO/ID/ROM.
- Added startup/runtime/debug telemetry for vehicle command mode, resolved wheel joint IDs, sampled speed, and wheel angular targets.
- Disabled the old synthetic joint-reference marker by default so it no longer appears to be a real world-frame end-effector target.
- Added Kevin task-space PPO observation/reward contract so no source IK bridge
  is required.

## Boundary

The implemented vehicle alignment is source-equivalent to the RRLAB obstacle
task. The task-space reference alignment is implemented as PPO observation and
reward context, not as direct conversion to joint references or valves.
