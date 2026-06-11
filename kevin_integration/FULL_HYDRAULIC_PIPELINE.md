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

## Curriculum (Learnability Repair)

The `rl_task_space_tracking` mode activates a four-stage curriculum designed to
improve early training stability. The curriculum is governed by config fields on
`MulagAdaptiveRLJointReachEnvCfg` and overridden by
`MulagHydraulicROMTaskSpaceEnvCfg`.

| Stage | Steps         | Episode Length | Action Authority | Vehicle Motion | Command Range | Orientation Reward |
|-------|---------------|----------------|------------------|----------------|---------------|-------------------|
| S1    | 0 — 50k       | 15 s / 1500 st | 0.25×            | off            | tight         | disabled          |
| S2    | 50k — 150k    | 15 s / 1500 st | 0.50×            | off            | moderate      | disabled          |
| S3    | 150k — 300k   | 30 s / 3000 st | 0.75×            | off            | wide          | enabled           |
| S4    | 300k+         | 45 s / 4500 st | 1.00×            | 3–8 km/h       | full          | enabled           |

**Code touchpoints:**

- `_curriculum_stage()` — returns `[1,2,3,4]` based on `total_env_steps` vs `curriculum_stage_steps`
- `_prepare_high_level_action()` — multiplies `self._current_action` by `self._action_authority_multiplier`
- `_sample_task_space_command()` — selects x/y/z range from stage-indexed config tuple
- `_apply_rrlab_vehicle_command()` — zeroes wheel velocity when `stage <= curriculum_vehicle_stationary_stages`
- `_get_dones()` — applies stage-aware episode length timeout
- `_get_task_space_rewards()` — conditionally skips orientation reward when `curriculum_orientation_enabled[stage-1]` is `False`
- `_reset_idx()` — resets per-reason termination booleans, offending indices, and all episode-sum tracking tensors

**Termination diagnostics** (tracked per episode, logged to `extras["log"]`):

- `termination_joint_nonfinite`, `termination_lower_limit`, `termination_upper_limit`, `termination_velocity`, `termination_control_nonfinite` — booleans indicating which reason fired
- `termination_offending_joint`, `termination_offending_channel` — index of first violation
- `termination_count_joint_nonfinite`, `termination_count_lower_limit`, `termination_count_upper_limit`, `termination_count_velocity`, `termination_count_control_nonfinite` — cross-episode accumulators
- `termination_count_total` — total sum

**Reward tracking** (episode sums logged):

- `episode_pos_tracking_sum`, `episode_orient_tracking_sum`, `episode_progress_reward_sum`
- `episode_action_rate_penalty_sum`, `episode_action_mag_penalty_sum`, `episode_valve_sat_penalty_sum`
- `episode_clamp_penalty_sum`, `episode_fd_load_penalty_sum`, `episode_unsafe_penalty_sum`

**Contract assertions** (`_validate_control_mode()` for `rl_task_space_tracking`):

- 4 command joints required (cylinders)
- Observation dimension = 73 (`TASK_SPACE_OBSERVATION_DIM`)
- SD observer disabled (`use_sd_observer = False`)
- SD feedback disabled (`use_sd_feedback = False`)
- ID valve source active (`position_delta_source = "id_valve"`)
- FD effort actuation disabled (`use_fd_effort = False`)

## Tests

43 standalone unit tests in `kevin_integration/tasks/test_curriculum_learnability_repair.py`:

```bash
python kevin_integration/tasks/test_curriculum_learnability_repair.py
```

Covers stage boundaries, command range expansion, action authority values, episode length steps, orientation gating, vehicle stationary/moving, reward formulae, observation/action dimensions, termination diagnostics structure, extras["log"] keys, config field integrity, progress sign consistency, reset semantics, and boundary conditions.
