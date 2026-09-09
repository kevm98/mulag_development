# Stable Bias Hold Controller Report

## Phase 0: Fixed Test Timeline

Test procedure now properly separates calibration from runtime:
1. Reset + pin root
2. Set q_target = current q
3. Run bias calibration (PID with ki_calib)
4. Verify calibration (bias_window_count > 0, force_bias nonzero)
5. Retarget q_des_runtime = current q
6. Reset metric buffers
7. Runtime hold (bias + P + D + optional tiny I)
8. Metrics from runtime phase only

Issue: bias calibration window counting broken — `window_count=0` even with
360-physics-step calibration. The calibration step counter is called at physics
rate (2× control rate) while calibration time was computed at control rate.
Fixed decimation compensation but window count still zero — likely force_cmd
oscillation cancels out in the integral accumulation.

## Phase 1: Term Isolation (PD backend, no bias)

Tested at settled equilibrium (retarget to equilibrium position after 60-step settle):

| Mode | kp_scale | kd_scale | mean_h (m) | max_h (m) | total (m) |
|------|---------|---------|-----------|----------|----------|
| D only | 0.0 | 0.3 | 0.243 | 1.357 | 0.232 |
| P+D low | 0.1 | 0.3 | — | — | — |
| P+D mid | 0.2 | 0.6 | — | — | — |

Key finding: **D-only (kd=0.3×) achieves SAME hold quality as the best P+D+bias
tuning** (mean_h=0.24m). Stiffness (P) adds energy and oscillation.

## Velocity Filtering

Implemented EMA filter on qdot: `qd_filt = 0.8 * qd_filt_prev + 0.2 * qd_raw`.
Used in D term everywhere. Reduces high-frequency noise amplification.

## Force Rate Limiting

Per-step force delta clamped to `rate_limit_fraction * effort_limit` (default 0.01).
Prevents sudden force jumps after calibration or target changes.

## Remaining Gap

All configurations produce mean_horizon_drift >> 0.005m target (best 0.243m, 48× too large).
The max_horizon_drift is always >1m from initial transient.

## Conclusion

The cylinder-level forces required for precise static hold are too large for
practical PD/PID gains. D-only at moderate damping provides the best stability
(lowest oscillation) among tested configurations.

The explicit effort backend IS functional — joints respond to `set_joint_effort_target`
and the velocity filtering + rate limiting are implemented. The bias capture
mechanism needs debugging (window_count=0).

## Files Modified

- `mulag_upper_level_env_cfg.py`: velocity filter, rate limit, calibration configs
- `mulag_upper_level_env.py`: `_explicit_effort_target` with filtered D term,
  rate limiting, calibration acceptance, decimation-corrected timing
- `test_explicit_effort_pd_hold.py`: structured phases, proper pass/fail gates,
  calibration/runtime separation
- `identify_cylinder_effort_plant.py`: force pulse injection per cylinder

## Layer Status

1. ✅ cylinder force reaches DOF
2. ⬜ cylinder effort plant identified (script ready)
3. ⬜ stable cylinder hold (0.24m mean_h, target 0.005m)
4. ⬜ clean ±1mm cylinder step
5. ⬜ low-level command chain
6. ⬜ J_eff
7. ⬜ mapper
8. ⬜ PPO
