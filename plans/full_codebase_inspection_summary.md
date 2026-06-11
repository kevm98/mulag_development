# Full Codebase Inspection: Kevin RL Pipeline Audit

**Branch:** `integration/taskspace-fd-hydraulic-pipeline`
**Date:** 2026-06-11
**Files Scanned:** 20+ source files across RRLAB working task and Kevin task

---

## 1. Git Worktree State

| Property | Value |
|----------|-------|
| **Branch** | `integration/taskspace-fd-hydraulic-pipeline` |
| **Modified files** | `kevin_integration/TASKSPACE_RECONCILIATION_REPORT.md`, `kevin_integration/tasks/mulag_adaptive_rl_env.py` |
| **Untracked** | `kevin_integration/tasks/test_preflight_observation_contract.py`, `plans/corrected_addendum.md`, `plans/integration_codebase_inspection_report.md` |
| **Working tree** | Clean (no staged/unstaged divergence) |

The diff on [`mulag_adaptive_rl_env.py`](kevin_integration/tasks/mulag_adaptive_rl_env.py) shows the addition of `_learned_preflight_scripted_action_active` flag and refactored preflight mode switching (lines 157, 826, 882, 2796, 2828).

---

## 2. Working RRLAB Task: ObstacleAvoidanceCfg

**Framework:** [`ManagerBasedRLEnv`](https://isaac-sim.github.io/IsaacLab) (Manager-based paradigm)
**File:** [`obstacle_avoidance_env_cfg.py`](extensions/rrlab_tasks/rrlab_tasks/manager_based/manipulation/obstacle_avoidance/obstacle_avoidance_env_cfg.py)

### Environment Parameters

| Parameter | Value |
|-----------|-------|
| `num_envs` | 169 |
| `env_spacing` | 20 |
| `decimation` | 2 |
| `sim.dt` | 1/120 = 0.00833s |
| `step_dt` | 2 × 0.00833 = 0.0167s |
| `episode_length_s` | 34.0s |
| `max_episode_length` | 34.0 / 0.0167 ≈ **2040 steps** |

### Action Space

- **Type:** [`JointPositionAction`](https://isaac-sim.github.io/IsaacLab) (position targets)
- **Output:** Joint position commands for 5 arm joints + 4 wheel joints + 2 steering joints = **11 total**
- **Normalization:** [`JointPositionNormalizedProcessedActionToLimitsAction`](extensions/rrlab_tasks/rrlab_tasks/manager_based/manipulation/obstacle_avoidance/mdp/actions.py) maps normalized [-1,1] to joint limits

### Observation Space (PolicyCfg: 5-joint subset)

| Observation Term | Shape | Frames | Notes |
|-----------------|-------|--------|-------|
| `height_map` | (21,41,3) | 3-frame stack | Ray-caster grid, CNN-ready |
| `tcp_height` | (1,3) | 3-frame | End-effector height sensors |
| `distance_data` | (1, 37) | 1 | Lidar 1-ch, ±90°, 5° res |
| `joint_pos` | (5,3) | 3-frame | 5 arm joints |
| `joint_vel` | (5,3) | 3-frame | |
| `distance_to_goal` | (2,3) | 3-frame | TCP pos in base frame minus command |
| `normalized_contact_force` | (2,1) | 1 | tanh(force/10000) → [0,1] |
| `contact_time` | (2,1) | 1 | Contact sensor duration |
| `normalized_joint_effort` | (5,1) | 1 | applied_torque / effort_limits → [-1,1] |
| `last_action` | (11,1) | 1 | Previous step actions |
| `normalized_unimog_pose` | (4,3) | 3-frame | (x_norm, y_norm, sin(yaw), cos(yaw)) |
| `custom_unimog_pose_dot` | (3,3) | 3-frame | (x_dot, y_dot, yaw_dot) |
| `normalized_depth_image` | (100,100,1) | 1 | Tiled camera, range-normalized |

**`concatenate_terms=False`** — observations NOT concatenated (used with actor-critic that handles dicts)
**`enable_corruption=False`** — no observation noise

### Reward Configuration (18 terms, total raw magnitude ≈ 300+)

| Reward Term | Weight | Type | Notes |
|-------------|--------|------|-------|
| `end_effector_height_to_terrain` | **-40** | Linear | Target 0.05m, punishes deviation |
| `end_effector_height_to_terrain_tanh` | **+20** | Tanh kernel | Reward within target band |
| `end_effector_height_to_terrain_abs` | **-40** | Absolute | Threshold 0.02 |
| `end_effector_height_to_terrain_abs_tanh` | **+20** | Tanh kernel | std=0.02 |
| `distance_to_obstacle` | **-30** | Gaussian | safe_distance=1.5, std=0.2 |
| `undesired_collision` | **+100** | Binary | threshold=100N, lateral |
| `end_effector_position_tracking` | **-12** | Linear | offset=0.4m |
| `end_effector_position_tracking_fine_grained` | **+30** | Tanh kernel | std=0.4m |
| `end_effector_yaw_tracking` | **-15** | Linear | offset_deg=30 |
| `end_effector_yaw_tracking_fine_grained` | **+15** | Tanh kernel | std_deg=30 |
| `end_effector_pitch_tracking` | **-15** | Linear | offset_deg=4 |
| `end_effector_pitch_tracking_fine_grained` | **+15** | Tanh kernel | std_deg=4 |
| `tcp_ahead_of_arm_base` | **+2** | Binary | threshold=-0.1 |
| `tcp_hovering_above_terrain` | **-30** | Timed penalty | 0.15m, hysteresis 0.05, 3s |
| `tcp_off_working_space` | **-30** | Timed penalty | -1.7, hysteresis 0.2, 3s |
| `tcp_off_working_space_tanh` | **-30** | Tanh | threshold=-1.7, std=0.2 |
| `finish_on_limited_contacts` | **+10** | Dense scoring | 0.2*progress + 0.4*contact_quality + 0.4*contact_score + bonus |
| `action_rate` | **-0.006** (effective) | L2 | weight=-0.06 × decimation=2 → -0.03/step? |
| `joint_vel` | **-0.7** | L2 | Per-joint velocity penalty |

**Key:** The RRLAB reward uses a **mixture of linear penalties and tanh-shaped rewards** with large magnitudes (up to ±100), providing strong gradient signals across diverse failure/success modes.

### Termination Conditions

| Condition | Threshold | Type |
|-----------|-----------|------|
| `unimog_end_of_lane` | x < -16m (time_out=True) | Timeout + boundary |
| `time_out` | episode_length ≥ max | Basic timeout |
| `unimog_flying` | roll/pitch > 0.05 rad | Safety |
| `undesired_collision_ausleger_2` | > 1000 N | Safety |
| `success` | time_out=True | Flags success on timeout |

### PPO Runner: [`MulagObstacleAvoidancePPORunnerCfg`](extensions/rrlab_tasks/rrlab_tasks/manager_based/manipulation/obstacle_avoidance/config/mulag/agents/rsl_rl_ppo_cfg.py)

| Parameter | Value |
|-----------|-------|
| `num_steps_per_env` | **24** |
| `max_iterations` | 1000 |
| `save_interval` | 50 |
| `experiment_name` | `reach_mulag` |
| `empirical_normalization` | **False** |
| `policy` / `critic` | **[64, 64]** ELU |
| `init_noise_std` | **1.0** |
| `clip` | 0.2 |
| `entropy_coef` | 0.001 |
| `num_learning_epochs` | **8** |
| `num_mini_batches` | **4** |
| `learning_rate` | **1e-3** (adaptive) |
| `gamma` | 0.99 |
| `lam` | 0.95 |
| `desired_kl` | 0.01 |
| `max_grad_norm` | 1.0 |

---

## 3. Failing Kevin Task: MulagHydraulicROMTaskSpaceEnvCfg

**Framework:** [`DirectRLEnv`](https://isaac-sim.github.io/IsaacLab) (Direct paradigm)
**Files:** [`mulag_adaptive_rl_env_cfg.py`](kevin_integration/tasks/mulag_adaptive_rl_env_cfg.py), [`mulag_adaptive_rl_env.py`](kevin_integration/tasks/mulag_adaptive_rl_env.py)

### Architecture (Learned Neural Pipeline)

```
RL Action (4d) → expand_action_to_id (5d) → clamp_and_sanitize
  → AAM (Action Activation Model): qddot_cmd (4d) 
  → ID (Inverse Dynamics): valve_cmd (4d)
  → Valve → FD (Force Disturbance) observer: tau_hat (4d)
  → SD (State Disturbance) observer: dP_hat, fnet_hat
  → ROM position-target (hydraulic simulation) → joint_pos_target (4d)
  → Isaac Sim PD controller
```

### Environment Parameters

| Parameter | Value |
|-----------|-------|
| `num_envs` | **1** (default, configurable) |
| `decimation` | **1** |
| `sim.dt` | 1/100 = 0.01s |
| `step_dt` | 1 × 0.01 = **0.01s** |
| `episode_length_s` | **45.0s** |
| `max_episode_length` | 45.0 / 0.01 = **4500 steps** |
| `action_space` | **4** (hydraulic valve commands) |
| `observation_space` | `TASK_SPACE_OBSERVATION_DIM` |

### Control Mode Resolution

From [`_task_control_mode()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:398):
- `control_mode = "rl_task_space_tracking"` → returns `"rl_task_space_tracking"`
- `cfg.control_mode` takes priority; falls back to `cfg.task_control_mode`

From [`_low_level_control_mode()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:403):
- Since `control_mode = "rl_task_space_tracking"` (not "position_delta" or "effort")
- Falls back to `cfg.low_level_control_mode` → defaults to **"position_delta"**

### Action Space (4d)

RL outputs 4 continuous valve commands for 4 hydraulic cylinders. Processing in [`_prepare_high_level_action()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:883):
1. For `rl_task_space_tracking` mode: `_update_task_space_state()` called
2. Action = `clamp_and_sanitize(actions, 1.0)` → normalized to [-1, 1]
3. `self._current_action = cfg.action_scale * action` (action_scale defaults to 1.0)
4. `build_qddot_cmd()` converts action to qddot_cmd

Then in [`_pre_physics_step()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:1070):
- Since `use_final_causal_cylinder_rom=True`: routes to [`_pre_physics_step_final_causal_rom()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:1187)
- This runs: ID→valve_cmd, FD observer, SD observer, SD feedback, ROM position-target prediction
- Final output: `_rom_cylinder_pos_target` (4d joint position targets)
- In [`_apply_action()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:1531): `robot.set_joint_position_target(_rom_cylinder_pos_target, command_joint_ids)`

### Observation Space (Task Space Mode)

From [`build_task_space_rl_obs()`](kevin_integration/rl/observation_builder.py:73):

| Component | Dim | Notes |
|-----------|-----|-------|
| `ee_pose_command` | 7 | (x, y, z, qx, qy, qz, qw) in base frame |
| `current_ee_pose` | 7 | Current end-effector pose |
| `ee_position_error` | 3 | (dx, dy, dz) |
| `ee_orientation_error` | 1 | Quaternion error magnitude |
| `mechanical_q` | 5 | Joint positions (5 arm joints) |
| `mechanical_qdot` | 5 | Joint velocities |
| `cylinder_q` | 4 | Cylinder joint positions |
| `cylinder_qdot` | 4 | Cylinder joint velocities |
| `prev_action` | 4 | Previous RL action |
| `prev_valve_cmd` | 4 | Previous valve command |
| `z_arm_hat` | 24 | AAM internal state (24-dim) |
| `fd_tau_hat` | 4 | FD torque estimate |
| `vehicle_speed_kmh` | 1 | Commanded vehicle speed |

**Total:** 73 dims (controlled via `TASK_SPACE_OBSERVATION_DIM`)

### Reward Configuration (Task-Space Mode)

From [`_get_rewards()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:1603):

When `_task_control_mode() == "rl_task_space_tracking"`:

| Reward Component | Formula | Effective Weight | Notes |
|-----------------|---------|-----------------|-------|
| **pos_tracking** | `exp(-4.0 * ||ee_pos_error||)` | **+1.0** (unweighted) | Max 1.0 when perfectly tracking |
| **orient_tracking** | `0.25 * exp(-2.0 * orient_error)` | **+0.25** max | Max 0.25 when perfectly aligned |
| **action_rate_penalty** | `-rew_action_rate * mean(action_delta²)` | -0.05 × mean(action_delta²) | From `cfg.rew_action_rate` |
| **action_mag_penalty** | `-rew_action_magnitude * mean(action²)` | -0.02 × mean(action²) | From `cfg.rew_action_magnitude` |
| **valve_sat_penalty** | `-rew_action_saturation * sat_fraction` | -0.5 × sat_fraction | Saturation at `action_saturation_threshold` |
| **clamp_penalty** | `-rom_target_clamp_fraction` | **-1.0** per clamped output | ROM position targets hitting limits |
| **fd_load_penalty** | `-rew_fd_load * mean(fd_excess²)` | **0.0** (default: rew_fd_load=0) | Currently disabled |
| **unsafe_penalty** | `-rew_termination * terminated` | -5.0 if terminated | From `cfg.rew_termination` |

**CRITICAL FINDING:** The task-space reward does **NOT** use `_reward_weights_for_control_mode()` or `JointReachRewardWeights` at all. It's a completely separate reward function that:
1. **Has no q_tracking reward** (rew_q_tracking=0.0 in cfg is ignored in this code path)
2. **Has no progress reward** (no `joint_reach_reward()` call)
3. **Has no alive bonus** (no `rew_alive`)
4. **Has no settle_tracking reward** (use_settle_tracking_reward=False is irrelevant in this path)
5. **Maximum achievable reward per step:** ~1.25 (pos_tracking 1.0 + orient_tracking 0.25) minus penalties
6. **Typical reward when tracking poorly:** ~0.0 to -2.0 (all penalties active, tracking rewards near zero)

For reference, the joint-space reward path (used by `MulagAdaptiveRLJointReachEnvCfg`) goes through `_reward_weights_for_control_mode()` → [`joint_reach_reward()`](kevin_integration/rl/reward_terms.py:167) which uses `JointReachRewardWeights`:

| Weight | Default Value | Notes |
|--------|--------------|-------|
| `q_tracking` | 3.0 | -weight × joint_weights × q_error² |
| `qdot` | 0.25 | -weight × |qdot|² |
| `progress` | 2.0 | Δerror_mean |
| `action_rate` | 0.05 | -mean(action_delta²) |
| `action_magnitude` | 0.02 | -mean(action²) |
| `action_saturation` | 0.5 | -mean(max(|action|-threshold,0)²) |
| `valve_effort` | 0.01 | -|valve|² |
| `torque_effort` | 1e-6 | -|torque|² |
| `alive` | 0.1 | +0.1 per step alive |
| `termination` | 5.0 | -5.0 on termination |
| `settle_tracking` | 2.0 | Only in settle phase |
| `joint1_settle_tracking` | 3.0 | Only in settle phase |

**`_reward_weights_for_control_mode()`** (line 1724):
- For `position_delta` + `not use_fd_effort`: returns `replace(self._reward_weights, torque_effort=0.0)`
- Otherwise: returns `self._reward_weights` unchanged

**This method is NOT called in task-space mode**, only in the joint-space branch.

### Termination Conditions

From [`_get_dones()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:2363):

**`terminated`** = `safety_terminated | (~finite_control)`

Where:
- **`safety_terminated`** = [`joint_safety_termination()`](kevin_integration/rl/reward_terms.py:255):
  - `~finite(q, qdot)` — NaN/Inf check
  - `lower_violation` = any(q < lower_limits - joint_limit_margin)
  - `upper_violation` = any(q > upper_limits + joint_limit_margin)
  - `velocity_violation` = any(|qdot| > max_abs_joint_velocity)

- **`finite_control`** checks (23 tensors checked for finiteness):
  - `_current_torque`, `_current_fd_simscape_output`, `_current_tau_fd_adapted`, `_fd_tau_hat`, `_fd_tau_hat_normalized`
  - `_fd_soft_limit_fraction`, `_fd_hard_limit_fraction`, `_fd_normalized_abs_max`
  - `_current_tau_applied_raw`, `_current_tau_isaac_filtered`, `_current_tau_isaac_clamped`
  - `_current_valve_cmd`, `_current_qddot_cmd`, `_z_arm_hat`, `_dP`, `_fnet`
  - `_rom_cylinder_pos_target`, `_rom_pressure`, `_rom_xdot_pred`, `_rom_dx_pred`
  - `_final_rom_delta_extension_50ms/kinematic_50ms/correction_50ms/10ms`
  - `_final_rom_delta_isaac_50ms/10ms`
  - `_final_rom_extension`, `_final_rom_extension_velocity`

**`time_out`** = `episode_length_buf >= max_episode_length - 1`
**Note:** The task-space mode has **NO early termination for "success"** — no target-reached logic, no timeout-success flag. Only safety violations and NaN/Inf trigger early termination.

### Reset Logic

From [`_reset_idx()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:2427):

1. Records `reset_due_to_timeout`, `reset_due_to_termination`
2. Calls `super()._reset_idx(env_ids)` — DirectRLEnv reset
3. Resets joint positions to defaults, resets all internal state tensors to zero
4. Calls `_sample_joint_targets(env_ids, q_ref_center)` — samples new joint-space targets
5. Calls `_sample_task_space_command(env_ids)` — samples new EE pose command (x ~ U(6.5,7.5), y ~ U(0.7,2.2), z=0.5, quat=[0,0,1,0])
6. Resets all ROM state, controller state, pressure/torque state
7. Resets `_learned_control_initialized[env_ids] = False` — triggers re-initialization on next step

---

## 4. PPO Runner Comparison

| Parameter | RRLAB (Working) | Kevin Task-Space (Failing) |
|-----------|----------------|---------------------------|
| **`num_steps_per_env`** | 24 | 24 |
| **`max_iterations`** | 1000 | 1000 |
| **`save_interval`** | 50 | 50 |
| **`empirical_normalization`** | **False** | **True** |
| **Policy/Critic** | **[64,64]** ELU | **[128,128]** ELU |
| **`init_noise_std`** | **1.0** | **0.5** |
| **`num_learning_epochs`** | **8** | **5** |
| **`num_mini_batches`** | 4 | 4 |
| **`learning_rate`** | **1e-3** (adaptive) | **3e-4** (adaptive) |
| **`gamma`** | 0.99 | 0.99 |
| **`lam`** | 0.95 | 0.95 |
| **`desired_kl`** | 0.01 | 0.01 |
| **`max_grad_norm`** | 1.0 | 1.0 |
| **Num Envs** | 169 | 1 |
| **Step dt** | 0.0167s | 0.01s |
| **Episode Length** | 2040 steps (34s) | **4500 steps (45s)** |

**Key differences:**
1. **`empirical_normalization=True`** in Kevin — observsation normalization can suppress signal with 1 env
2. **`init_noise_std=0.5`** vs 1.0 — lower initial exploration noise
3. **`num_learning_epochs=5`** vs 8 — fewer updates per batch
4. **`lr=3e-4`** vs 1e-3 — 3× lower learning rate
5. **1 env vs 169** — dramatically less diverse experience per iteration
6. **4500 steps vs 2040** — 2.2× longer episodes

---

## 5. Reward Path Trace (Task-Space Mode)

```
_get_rewards()
  └── _task_control_mode() == "rl_task_space_tracking" → TRUE
      ├── _update_task_space_state()
      │   ├── ee_pos_w = robot.data.body_link_state_w[:, ee_body_id, :3]
      │   ├── ee_quat_w = robot.data.body_link_state_w[:, ee_body_id, 3:7]
      │   ├── subtract_frame_transforms() → ee_pos_b, ee_quat_b (body frame)
      │   ├── _current_ee_pose[:3] = ee_pos_b
      │   ├── _current_ee_pose[3:] = ee_quat_b
      │   ├── _ee_position_error = command[:3] - ee_pos_b
      │   └── _ee_orientation_error = quat_error_magnitude(ee_quat_b, command[3:])
      ├── pos_tracking = exp(-4.0 * ||ee_position_error||)       ← [0, 1]
      ├── orient_tracking = 0.25 * exp(-2.0 * orient_error)      ← [0, 0.25]
      ├── action_rate_penalty = -0.05 * mean(action_delta²)       ← [-0.2, 0]
      ├── action_mag_penalty = -0.02 * mean(current_action²)      ← [-0.08, 0]
      ├── valve_sat_penalty = -0.5 * sat_fraction                 ← [-0.5, 0]
      ├── clamp_penalty = -rom_target_clamp_fraction               ← [-1.0, 0]
      ├── fd_load_penalty = 0.0 (rew_fd_load=0)                   ← 0
      └── unsafe_penalty = -5.0 * terminated                      ← {-5.0, 0}
      
      reward = pos_tracking + orient_tracking + penalties
      
      MAX possible reward: ~1.25 (perfect tracking, no penalties)
      TYPICAL reward (poor tracking): -0.5 to -2.0
      Reported mean: -1756 — this implies most episodes end in termination (unsafe_penalty=-5)
```

### Root Cause Analysis for -1756 Mean Reward

At 4500 steps with mean episode length 3011:
- If each step gives ~0 reward on average when tracking poorly (~-0.5), total = 3011 × -0.5 = -1505
- Plus termination penalties: each episode ends with unsafe_penalty=-5.0
- If episodes frequently terminate early (safety violations), each termination adds -5 to accumulated reward
- With frequent early terminations, the cumulative penalty compounds

**The dominant factor for -1756:** The task-space reward has a max of ~1.25 per step, and when tracking is poor (likely with an untrained policy), rewards hover near 0 or slightly negative. With 3011 step episodes, the accumulated reward sums to around -1756, which translates to ~-0.58 average per step — consistent with the penalties dominating the small tracking rewards.

---

## 6. Termination Path Trace

```
_get_dones()
  ├── q, qdot = _read_joint_state()
  ├── safety_terminated = joint_safety_termination(q, qdot, lower_limits, upper_limits,
  │       joint_limit_margin, max_abs_joint_velocity)
  │   ├── ~finite(q) | ~finite(qdot) → NaN in joint state
  │   ├── q < lower_limits - margin → joint limit violation
  │   ├── q > upper_limits + margin → joint limit violation
  │   └── |qdot| > max_abs_joint_velocity → velocity safety
  │
  ├── finite_control = (23 tensor finiteness checks ANDed together)
  │   └── ANY NaN/Inf in the learned pipeline triggers termination
  │
  ├── terminated = safety_terminated | (~finite_control)
  ├── time_out = episode_length_buf >= max_episode_length - 1
  │
  └── return terminated, time_out
```

### Common Early Termination Reasons

1. **ROM position targets hitting limits** → `_rom_target_clamp_fraction` may indicate clamped outputs
2. **NaN propagation through learned pipeline** → AAM/ID/FD/SD can produce NaNs
3. **Joint limit violations** → ROM pushing cylinders beyond soft limits
4. **Velocity violations** → ROM or hydraulic simulation producing aggressive motion

---

## 7. Comparison Summary: Key Structural Differences

| Aspect | RRLAB (Working) | Kevin Task-Space (Failing) |
|--------|----------------|---------------------------|
| **RL Paradigm** | ManagerBasedRLEnv (events, commands, sensors) | DirectRLEnv (manual step logic) |
| **Action type** | Joint position (11 DoF: arm+wheels+steering) | Valve command (4DoF: hydraulic cylinders) |
| **Physical actuation** | Direct joint position → PD control | Learned pipeline → ROM position targets → PD |
| **Observations** | Rich: cameras, lidar, height maps, contact sensors | Learned neural state (AAM hidden, FD, SD) + task space |
| **Reward density** | 18 terms, mixture ±100 magnitude, shaping signal | 7 terms, max +1.25/step, predominantly penalties |
| **Termination** | Multiple: lane end, flying, collision, timeouts | Safety: joint limits, NaN/Inf, no "success" termination |
| **Environments** | 169 parallel | 1 (default) |
| **Episode length** | 34s, 2040 steps | 45s, 4500 steps |
| **PPO network** | [64,64], init_std=1.0 | [128,128], init_std=0.5 |
| **Observation norm** | False | True |
| **Learn rate** | 1e-3 | 3e-4 |
| **Training epochs** | 8 | 5 |

### Most Likely Failure Sources (Ranked)

1. **Reward starvation:** Task-space reward is fundamentally weak (~1.25 max per step, dominated by penalties). With poor initial tracking, the policy sees mostly negative rewards, making it hard to learn positive behaviors.

2. **Frequent safety terminations:** The 23-tensor finite_control check combined with joint safety limits means any NaN from the learned pipeline (AAM/ID/FD/SD) immediately terminates the episode, giving the policy no chance to learn recovery behaviors.

3. **Single environment + long episodes:** 1 env × 4500 steps × 24 steps/batch = only ~10-11 transitions per PPO update. Combined with `empirical_normalization=True`, the statistics are extremely noisy.

4. **No exploration-friendly curriculum:** Unlike the RRLAB task which has command resampling and randomization events, the Kevin task has only q_ref curriculum (amplitude 0.25→1.0 over 20k steps) and no action-space curriculum.

5. **ROM clamping penalties dominant:** If the ROM position targets frequently hit limits, the `clamp_penalty` (-1 per clamped dimension) can easily dominate the small tracking rewards.
