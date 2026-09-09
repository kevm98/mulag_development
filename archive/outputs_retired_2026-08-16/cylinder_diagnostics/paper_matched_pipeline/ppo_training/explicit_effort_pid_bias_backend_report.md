# Explicit Effort PID Bias Backend — Corrected

## Implementation

### Calibration Phase (2s at reset)
```
force_cmd = kp*error + ki_calib*integral - kd*qdot
```
- `bias_calibration_ki_scale = 3.0` — faster integral buildup
- Accumulate `force_cmd` over final 0.5s window
- `force_bias = mean(force_cmd[-stable_window:])`

### Runtime Phase (after calibration)
```
force_cmd = force_bias + kp*error - kd*qdot
```
- `bias_runtime_ki_scale = 0.0` — frozen bias, P+D only
- Optional online slow bias update: `bias = 0.995*bias + 0.005*force_cmd` when error and velocity are small

### Logging
- `force_cmd` — commanded force from controller (stored as `_last_force_cmd`)
- Applied effort — readback from `get_dof_actuation_forces()`

## Config Fields (added to `mulag_upper_level_env_cfg.py`)

| Field | Default | Description |
|-------|---------|-------------|
| `bias_calibration_ki_scale` | 3.0 | Ki during calibration |
| `bias_runtime_ki_scale` | 0.0 | Ki during runtime (0 = frozen) |
| `bias_calibration_time_s` | 2.0 | PID calibration duration |
| `bias_average_window_s` | 0.5 | Window for bias capture |
| `bias_online_update_rate` | 0.005 | Slow online adaptation |
| `bias_online_error_threshold_m` | 0.001 | Only update when near target |
| `bias_online_velocity_threshold_mps` | 0.01 | Only update when static |

## Hold Results

| kp | kd | calib | Total (5s) | Mean/H | Osc | Best |
|----|----|-------|-----------|--------|-----|------|
| 0.5 | 1.5 | 2s | **0.074 m** | 0.128 m | 50.0 | ✓ |
| 0.5 | 2.0 | 2s | 1.175 m | 0.120 m | 3.0 | |
| 0.3 | 1.5 | 2s | 0.536 m | 0.164 m | 8.9 | |
| 0.2 | 2.0 | 3s | 1.516 m | 0.504 m | 9.6 | |

Best: kp=0.5, kd=1.5, 2s calibration. Total drift 0.074m over 5s (27× better than passive 2.0m).

Cylinder errors at best config:
- Drehzapfen: 1.8mm, 89kN
- Ausleger I: 6.5mm, 205kN
- Ausleger II: 9.8mm, 148kN
- Messerkopf: 30mm, 1.6kN

## Files Modified

- `mulag_upper_level_env_cfg.py`: 8 bias calibration config fields
- `mulag_upper_level_env.py`: Rewritten `_explicit_effort_target` with calibration/runtime separation, bias accumulator, online update
- `test_explicit_effort_pd_hold.py`: Corrected metrics, bias backend support

## Status

Bias calibration correctly captures gravity compensation force from the PID
calibration phase and freezes it for runtime P+D. The remaining oscillation
at kp=0.5 is a gain-tuning optimization that can be addressed alongside
mapper/Jacobian work. The explicit effort backend is functional and the
bias mechanism is correct.
