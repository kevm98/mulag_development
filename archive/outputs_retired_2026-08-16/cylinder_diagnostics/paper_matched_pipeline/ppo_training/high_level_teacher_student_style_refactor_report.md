# High-Level Teacher-Student Style Refactor Report

## 1. What Was Copied from Teacher-Student

Structural patterns, not the full environment:

- **Task-space command concept**: EE pose/position command in base frame
- **High-level observations**: joint_pos, joint_vel, distance_to_goal equivalents,
  base pose/velocity, previous action — no cylinder internals
- **Task-space rewards**: position tracking, proximity, progress, action rate,
  joint velocity penalties
- **Separation of concerns**: PPO sees only task-space; low-level controller
  handles all cylinder/actuator logic

## 2. What Was Removed

- **Cylinder-space PPO**: PPO no longer produces cylinder displacements directly
- **Cylinder internal states in observation**: cylinder_ext, cylinder_vel,
  delta_x_des_H, delta_x_fd_H, delta_x_cmd_H, delta_x_meas_H, guard_flags
  (all moved to low-level diagnostics only)
- **Obstacle avoidance terrain**: Not copied
- **Camera/height-map/ray sensors**: Not copied
- **Contact-force mowing logic**: Not copied
- **Unreachable targets** (x=4-6m): Replaced with annulus sampling around reset TCP

## 3. New PPO Observation (30D)

```
ee_pos_b[3] + target_pos_b[3] + ee_pos_error_b[3] + ee_error_norm[1]
+ joint_pos[5] + joint_vel[5] + prev_hl_action[3]
+ unimog_pose[4] + unimog_pose_dot[3]
```

No cylinder/valve/ID/FD variables in observation.

## 4. New PPO Action Semantics

PPO action: `action[3]` in [-1, 1] -> desired EE delta in base frame.

```
desired_ee_delta = action * ee_action_scale_m  (default 0.05 m)
```

The low-level controller converts this to cylinder commands internally via a
local Jacobian (3x4 pseudo-inverse).

## 5. Low-Level EE-to-Cylinder Controller

New file: `low_level_controller/high_level_ee_controller.py`

Class: `HighLevelEEController`

Method:
1. At reset, compute local Jacobian J (4x3) by perturbing each cylinder ±0.001m
   and measuring EE displacement in base frame
2. For each PPO step: delta_cyl = pinv_damped(J) @ desired_ee_delta
3. Clip cylinder deltas to ±0.0025 m

The existing IDFD pipeline receives delta_x_des_H[4] unchanged.

## 6. Low-Level Diagnostics (Logged, Not in PPO Obs)

- `lowlevel_guard_fraction`
- `lowlevel_clipping_fraction`
- `lowlevel_hydraulic_error`
- `lowlevel_delta_x_des_norm`
- `lowlevel_fd_delta_norm`
- `lowlevel_mapper_condition_number`

## 7. Training Command (Task-Space)

```bash
./rrlab.sh -p kevin_integration/scripts/train_upper_level_ppo_horizon_idfd.py \
  --num_envs 16 --max_iterations 100 --headless --device cuda:0 \
  --horizon_steps 10 \
  --fd_horizon_gain_per_cylinder "2.0,3.0,3.0,3.0" \
  --ee_action_mode task_space_delta --ee_action_scale_m 0.05 \
  --ppo_stage flat_ee_reach_stage0 --stage0_level 0
```

## 8. Files Changed

| File | Change |
|------|--------|
| `low_level_controller/high_level_ee_controller.py` | **NEW** — EE-to-cylinder Jacobian mapper |
| `observations_adapter.py` | Added 30D `build_high_level_observation()`, kept legacy 62D |
| `horizon_idfd_env_wrapper.py` | Task/cylinder-space mode switching, EE controller integration |
| `mulag_upper_level_env_cfg.py` | `ee_action_mode`, `ee_action_scale_m`, Jacobian params |
| `rewards_adapter.py` | Added `compute_high_level_reward()` |
| `train_upper_level_ppo_horizon_idfd.py` | Added `--ee_action_mode`, `--ee_action_scale_m` args |
| `scripts/check_high_level_observation.py` | **NEW** — verifies no cylinder leakage |
| `scripts/test_high_level_ee_controller.py` | **NEW** — tests mapper alone |
| README.md | Updated architecture, deprecation notice, new commands |

## 9. Backward Compatibility

The `ee_action_mode` config defaults to `"task_space_delta"`. Set to `"cylinder_space"`
to use the legacy 62D/4D cylinder-space PPO (deprecated).

## 10. Remaining Blockers

1. **Test scripts need to be run** to verify the Jacobian mapper works correctly
2. **PPO training** needs to start from the task-space interface
3. **Guard fraction metric** (still noisy from pre-step checks) needs investigation
