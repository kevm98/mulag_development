# Stage-0 Task Fix Report

## 1. Root Cause of 97.3% Workspace Violations

The dominant failure mode (97.3% of 1245 episodes) is workspace violation. The root cause is **unreachable targets**.

The command sampler generates targets at x=(4.0, 6.0) m in the vehicle base frame. The TCP at reset is positioned at approximately 1.5-2.5 m in base frame. Initial EE errors are 4.5-7.7 m (mean 6.1 m), far beyond the arm's maximum reach of ~3-3.5 m.

The arm extends toward these unreachable targets for roughly 14 horizon steps before crossing the workspace guard boundary.

## 2. Frame Mismatch Assessment

No frame mismatch was found. All components use base frame consistently:
- Reward: TCP in base, target from `_command_pos_w` (treated as base)
- Observation: `ee_pos_error_b = target_pos_b - ee_pos_b` (both base)
- Workspace guard: `_tcp_pos_base_frame()` with base-frame bounds
- Cab guard: `_tcp_pos_base_frame()` with base-frame bounds

The `_command_pos_w` variable is confusingly named but stores base-frame values.

## 3. Target Reachability

Command range x=(4.0, 6.0) m, TCP reset ~1.5-2.5 m, max arm reach ~3-3.5 m.
All targets are unreachable at reset (initial EE error > 4.5 m for every episode).

## 4. Stage-0 Curriculum

Implemented: local targets within 15 cm of reset TCP, soft workspace penalty (-3.0 per horizon), termination only after 10 consecutive violations.

## 5. Reward

Stage-0 reward: progress + proximity - action cost - hydraulic error.
Success bonus +10.0, failure penalty -20.0.

## 6. Files Changed

- `mulag_upper_level_env_cfg.py`: Added stage0 config fields
- `horizon_idfd_env_wrapper.py`: Stage0 target resampling, soft WS violation, conditional reward
- `rewards_adapter.py`: Added `compute_stage0_reward()`
- New scripts: `audit_target_frames.py`, `audit_target_reachability.py`, `evaluate_stage0_baselines.py`
- README updated with stage0 commands

## 7. Pending Validation

Scripts need to be run to validate the Stage-0 pipeline:
- `audit_target_frames.py` confirms frame consistency
- `audit_target_reachability.py` quantifies unreachability
- `evaluate_stage0_baselines.py` validates targets
- `train_upper_level_ppo_horizon_idfd.py` trains Stage-0 PPO
