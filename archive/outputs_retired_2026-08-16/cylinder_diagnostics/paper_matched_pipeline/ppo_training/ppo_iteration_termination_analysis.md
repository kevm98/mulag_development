# PPO Iteration Termination Analysis

This report is updated online from completed horizon episodes.

## Reward Trend
- mean_episode_return: 162.837
- first_20pct_mean_episode_return: 157.572
- last_20pct_mean_episode_return: 208.015

## Episode Length Trend
- mean_episode_length: 297.938
- first_20pct_mean_episode_length: 289.000
- last_20pct_mean_episode_length: 300.000

## Termination Cause Histogram
- success: 1 (0.031)
- timeout: 31 (0.969)
- cab_guard_violation: 0 (0.000)
- workspace_violation: 6 (0.188)
- collision: 0 (0.000)
- joint_limit: 0 (0.000)
- cylinder_limit: 0 (0.000)
- ground_safety: 0 (0.000)
- NaN/nonfinite: 0 (0.000)
- huge_transform: 0 (0.000)
- base_flying: 0 (0.000)
- reset_hold_failure: 0 (0.000)
- unknown_done: 0 (0.000)

## Rates
- success_rate: 0.031250
- failure_rate: 0.000000
- timeout_rate: 0.968750
- early_termination_rate: 0.000000
- first_20pct_early_failure_rate: 0.000000
- last_20pct_early_failure_rate: 0.000000

## Final EE Error Trend
- mean_final_ee_error: 0.223569
- mean_ee_error_improvement: 2.852429
- first_20pct_mean_final_ee_error: 0.244597
- last_20pct_mean_final_ee_error: 0.163705
- first_20pct_mean_ee_error_improvement: 5.684071
- last_20pct_mean_ee_error_improvement: -0.028273

## Verdict
- inconclusive

## Required Reward/Termination Fixes
- Failure terminations now receive a strong negative reward.
- Strict success requires low final EE error, sustained hold, no active guard, and a minimum episode length.
- Cab, workspace, cylinder, joint, nonfinite, and base-flying events are logged separately.
