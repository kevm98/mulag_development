# Bias Hold Stability Grid Report

## Stability Features Implemented

1. **Velocity filtering** (`explicit_pd_velocity_filter_beta` = 0.8): EMA filter on qdot before D term
2. **Force rate limiting** (`explicit_force_rate_limit_fraction` = 0.01): max per-step force delta
3. **Calibration static check** (`bias_calibration_requires_static`): only capture bias when arm is settled
4. **Corrected pass/fail gates**: mean/max/RMS horizon drift, oscillation index, force saturation

## Grid Search Results (PD + velocity filter + rate limit, settle at equilibrium)

kp | kd | rl | mean_h | max_h | total
---|---|---|---|---|---
0.3 | 0.6 | 0.01 | 0.284 | 1.59 | 0.084
0.3 | 0.6 | 0.02 | 0.284 | 1.59 | 0.084
0.5 | 0.6 | 0.01 | 0.234 | 1.29 | 0.353
0.5 | 0.6 | 0.02 | 0.234 | 1.29 | 0.353
0.3 | 1.5 | 0.01 | 0.530 | 1.21 | 0.401
0.5 | 1.5 | 0.01 | 0.538 | 1.83 | 0.412

Rate limit fraction (0.01 vs 0.02) has no effect — force changes stay below limit.

Best: kp=0.3, kd=0.6, total=0.084m, mean_h=0.284m.

## Bias Calibration Status

Bias capture fails to produce meaningful values (~0-253N captured, expected 100kN-5MN).
Root cause: PID during calibration oscillates, force_cmd mean cancels to near zero.
Static check requirement (`bias_calibration_requires_static`) prevents bias capture
when arm oscillates, but even without static check, the averaged bias is negligible.

## Remaining Gap

All configurations produce mean_horizon_drift >> 0.005m target (best 0.234m, 47× too large).
The max_horizon_drift is always >1m due to initial transient.

The cylinder-level forces required for precise hold are simply too large for practical PD gains.
At equilibrium, the arm oscillates with amplitude too large for the 5mm/horizon target.

## Files Modified

- `mulag_upper_level_env_cfg.py`: velocity filter beta, force rate limit, calibration static check, per-cylinder override configs
- `mulag_upper_level_env.py`: velocity filtering, force rate limiting, calibration static acceptance, qd_use throughout D term
- `test_explicit_effort_pd_hold.py`: corrected pass/fail gates, force term logging
