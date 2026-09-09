# Teacher-Student Adaptation Report

## Controller Kept

PPO1 still uses the selected horizon IDFD feed-forward controller:

```text
action[0:4]
-> delta_x_des_H = action * 0.01
-> closed-loop ID/FD rollout
-> delta_fd_H
-> delta_cmd_H = [2.0, 3.0, 3.0, 3.0] * delta_fd_H
-> smooth PhysX ramp over 10 steps
-> PhysX
```

`tracking_gain` remains `0.0`. Diagnostic controllers such as
`position_ramp_pi`, `horizon_idfd_ff_pi`, per-step velocity PID, direct
PPO-to-valve, and direct PPO-to-PhysX-target control are not used.

## Adapted Into PPO1

- Persistent horizon cylinder hold target.
- Zero-action hold behavior.
- Cylinder soft-limit clamp and workspace/cab guard flags.
- Compact 62D horizon observation.
- Horizon reward with tracking, progress, close-target, yaw, TCP-ahead,
  smoothness, joint-velocity, hydraulic-consistency, and guard penalties.
- Headless reset-hold and collision-proxy audit scripts.
- README commands for laptop, lab GPU, smoke, play, and audits.

## Not Adapted

- Heavy generated terrain.
- Cameras.
- Ray sensors and height maps.
- Teacher-student reset randomizers.
- Manager-based action semantics.
- Unsafe randomized poses.

## Verification

- `py_compile` passed for train/play, horizon wrapper, PPO1 env/config,
  observation/reward adapters, and both audit scripts.
- Reset/hold audit ran headless for 1.0 s:
  - `max_ee_drift_m = 0.277092`
  - `max_tcp_base_frame_drift_m = 0.059856`
  - `max_guard_fraction = 0.000000`
  - cylinder hold target delta stayed `0.000000`
  - finite/in-limit cylinder targets: `True`
  - cab proxy clear: `True`
  - pass: `False`, because drift remains above the requested `0.05 m` gate.
- Collision proxy audit passed:
  - cab proxy clear: `True`
  - finite/in-limit cylinder targets: `True`
  - root height ok: `True`
  - guard fraction: `0.000000`
- PPO smoke passed:
  - command: `train_upper_level_ppo_horizon_idfd.py --num_envs 1 --max_iterations 1`
  - actor/critic input dimension: `62`
  - action dimension: `4`
  - RSL-RL completed iteration `0/1` and closed cleanly.

Remaining blocker: the parked reset pose is cab-clear and no longer activates the
workspace guard, but it still settles by about 6 cm in the base frame over 1 s.
That is close to, but above, the requested reset-hold gate.
