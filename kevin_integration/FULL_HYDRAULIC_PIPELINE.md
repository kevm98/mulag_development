# Full Hydraulic Pipeline

Date: 2026-06-11

## High-Level Control

Task-space tracking uses PPO, not IK:

```text
Messerkopf ee_pose [x, y, z, qw, qx, qy, qz]
+ measured state
-> PPO Action[4]
```

The RRLAB obstacle task uses the same idea: `ee_pose` is an observation/reward command and PPO learns the mapping to arm action.

## Low-Level Learned Chain

```text
Action[4]
-> raw qddot command in ID training units
-> AAM history
-> z_arm_hat[24]
-> ID
-> ValveCmd[4]
-> valve-conditioned causal ROM, dt 0.05 s
-> DeltaExtension[4]
-> coordinate mapping
-> four prismatic position targets at Isaac dt 0.01 s
-> USD linkage
-> Messerkopf motion
```

The only actuator writer in ROM mode is:

```text
ROM -> robot.set_joint_position_target(... four cylinder joints ...)
```

## FD Observer

FD is active as:

```text
[q5, qdot5, dP4, ValveCmd4] history
-> FD
-> fd_tau_hat[4]
```

FD output is used for policy observation, load telemetry, safety fractions, and a small auxiliary reward. FD effort actuation is disabled in ROM position-target mode.

## Runtime Modes

- `joint_space_validation`: no checkpoint, stationary vehicle by default, scripted mechanical-joint reference, AAM/ID/ROM/FD observer active.
- `rl_task_space_tracking`: PPO training or checkpoint playback, RRLAB-style scripted vehicle motion, `ee_pose` command in observation/reward, PPO maps task-space state to `Action[4]`.

No mode claims task-space tracking without a trained PPO policy.
