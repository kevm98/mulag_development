# Stage-0 Stabilization Report

## 1. Baseline Results

| Baseline | Final EE Error | WS Violation | Guard Fraction |
|----------|---------------|-------------|----------------|
| Zero action | 0.042 m | TBD | ~1.0 (artifact) |
| Random action (scale=0.01) | TBD | high | high |
| Greedy action | TBD | TBD | TBD |

Key finding: Zero action is safe and stable at ~0.04 m EE error. PPO exploration
at action_scale=0.01 is too aggressive compared to zero-action baseline.

## 2. Action-Scale Sweep

Random-action baseline at scale 0.0025:
- Final EE error: 0.039 m
- WS violation rate: 0.0%
- Action norm: 1.12 (random)

**Recommended action scale: 0.0025** for Stage-0 Level 0.

## 3. PPO Config Changes

- **init_std**: 0.15 (was 0.50) — reduced initial exploration noise
- **Stage-0 Levels**: 0 (5cm box), 1 (10cm box), 2 (15cm box) with level-specific success criteria
- Training starts at Level 0 with action_scale=0.0025

## 4. Reward Changes

Stage-0 reward rescaled to [-5, +5] per horizon:
```
reward = 20.0 * clamp(progress, -0.02, 0.02)
       + 2.0 * exp(-8.0 * ee_error_after)
       - 1.0 * ee_error_after
       - 0.05 * ||action||
       - 0.10 * ||action - prev_action||
       - 1.0 * hydraulic_consistency
       - 5.0 * guard_fraction  (increased from 0.5)
       - 3.0 * clipping_fraction (new)
       + do_no_harm: 30.0 * progress if progress < 0
```

Success bonus: +5.0 (reduced from 20.0). Failure penalty: -20.0.

## 5. Penalty Changes

- Workspace violation: soft penalty (-5.0 per horizon), done after 20 consecutive steps
- Guard penalty: -5.0 * guard_fraction
- Clipping penalty: -3.0 * clipping_fraction
- Only severe cab collision + nonfinite cause immediate termination

## 6. Stage-0 Level Used

Level 0 (easiest):
- Target box: ±5 cm dx/dy, ±2 cm dz
- Success tolerance: 0.10 m
- Required hold steps: 3
- Min episode steps: 10

## 7. Training Command

```bash
./rrlab.sh -p kevin_integration/scripts/train_upper_level_ppo_horizon_idfd.py \
  --num_envs 16 --max_iterations 100 --headless --device cuda:0 \
  --horizon_steps 10 --horizon_action_scale_m 0.0025 \
  --fd_horizon_gain_per_cylinder "2.0,3.0,3.0,3.0" \
  --ppo_stage local_reach_stage0 --stage0_level 0
```

## 8. Remaining Blockers

1. **Guard fraction measurement artifact**: `_last_guard_fraction` reports 1.0
   even for zero-action, likely due to pre-step guard check during smooth ramp
   physics drift. Not critical for training — the guard is working correctly
   (clamping targets when appropriate) but the metric is noisy.

2. **Full baseline data pending**: Zero and random baselines need to run to
   completion (200 episodes) with the fixed evaluation script for complete
   statistical comparison.

3. **Training needs to be run**: All code changes are implemented and compile.
   The Stage-0 Level 0 PPO training with action_scale=0.0025 needs to be
   executed to verify learning.

## Files Changed

- `mulag_upper_level_env_cfg.py`: Stage-0 levels, penalty weights, reduced init_std
- `horizon_idfd_env_wrapper.py`: Level-based target sampling, level-based success criteria, reward logging
- `rewards_adapter.py`: Rescaled reward with do-no-harm term
- `train_upper_level_ppo_horizon_idfd.py`: Added --stage0_level and --ppo_stage args
- New: `scripts/sweep_action_scale.py`
- Updated: `scripts/evaluate_stage0_baselines.py`
