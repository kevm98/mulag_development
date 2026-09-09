# Teacher-Student Reference Audit

Reference path:
`teacher_student/rrlab-mulag_teacher_student`

## Scope Used

The teacher-student tree was used as a reference only. No manager-based action
semantics, heavy terrain, camera/ray sensors, reset randomizers, or unsafe reset
poses were copied into PPO1.

## Useful Ideas Borrowed

- Compact proprioception: arm joint position, arm joint velocity, previous action,
  Unimog pose, and Unimog pose rate are useful low-dimensional policy context.
- Task reward shape: TCP position tracking, progress, close-target tanh bonus,
  yaw/pitch tolerance, TCP-ahead bonus, action-rate penalty, and joint-velocity
  penalty are useful PPO terms.
- Collision/workspace terms: workspace/cab guard activation is useful as a PPO
  penalty signal, but PPO1 implements it as a lightweight guard flag rather than
  copying the obstacle-avoidance sensor stack.

## Explicitly Not Copied

- Generated/heavy terrain and terrain curriculum.
- Height-map observations.
- RTX camera observations and camera-enabled train/play defaults.
- Ray-caster distance sensors.
- Obstacle avoidance manager action semantics.
- Reset randomizers for root pose, joint offsets, link masses, friction, actuator
  gains, or driving speed.
- Collision rewards that require teacher-student contact/ray sensor wiring.
- Per-step velocity PID or direct PPO-to-valve/PhysX target control.

## Selected PPO1 Pipeline Preserved

```text
action[0:4]
-> delta_x_des_H
-> closed-loop ID/FD rollout
-> delta_fd_H
-> delta_cmd_H = fd_gain[c] * delta_fd_H
-> smooth PhysX ramp
-> PhysX
```

Selected controller: `current_horizon_idfd_ff`

- `fd_gain_per_cylinder = [2.0, 3.0, 3.0, 3.0]`
- `tracking_gain = 0.0`
- `horizon_steps = 10`
- `horizon_action_scale_m = 0.01`

## PPO1 Changes Motivated By The Audit

- Persistent reset hold target is maintained inside the horizon wrapper.
- Zero action maps to the persistent hold target, not to an unconstrained fresh
  current-position target.
- Horizon observation is now a 62D compact state vector with task, proprioception,
  previous horizon deltas, guard flags, and Unimog pose/rate.
- Horizon reward now includes tracking/progress/close-target terms, yaw, TCP-ahead,
  action-rate, joint-velocity, hydraulic-consistency, and guard penalties.
- Visual defaults stay black ground, white lane lines, sparse gray math-box grid,
  no heavy terrain, no cameras, and no ray sensors.
