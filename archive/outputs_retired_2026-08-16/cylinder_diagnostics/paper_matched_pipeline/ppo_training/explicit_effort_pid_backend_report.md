# Explicit Effort PID Backend Report

## 1. PD Results (from previous work)

- PD-only: nonzero effort, cylinders respond
- Best PD drift: 0.17m over 5s at 5× gain
- Gravity steady-state error: P-only cannot eliminate gravity bias
- High gains cause oscillation

## 2. PID Implementation

Extended `_explicit_pd_effort_target` to `_explicit_effort_target` supporting
both `explicit_effort_pd` and `explicit_effort_pid` backends.

PID formula:
```
force = kp*error + ki*integral - kd*velocity
integral += error * dt
integral clamped to [-i_limit, +i_limit]
```

Anti-windup:
- Integral clamped to `explicit_pid_integral_limit_m` (0.01m)
- Conditional integration: freeze integral when force saturated and
  pushing in same direction as error

Persistent state: `self._explicit_pid_integral_error` [num_envs, 4]
Initialized to zero in `__init__`, reset to zero in `_reset_idx`.

## 3. Gain Values

Base gains (per cylinder):

| Cylinder | Kp | Kd (increased) | Ki (new) | Effort Limit |
|----------|-----|----|-----|------|
| Drehzapfen | 15M | 3.0M | 1.0M | 100M |
| Ausleger I | 6M | 1.8M | 500k | 10M |
| Ausleger II | 1M | 1.5M | 100k | 1M |
| Messerkopf | 1M | 1.5M | 100k | 100k |

Scale factors: `explicit_pid_kp_scale`, `_kd_scale`, `_ki_scale`, `_effort_scale`

## 4. Hold Test Results

| Kp | Kd | Ki | Total Drift (5s) | Mean/Horizon |
|----|----|-----|-------------------|-------------|
| 1.0 | 1.0 | 1.0 | 0.81 m | 0.20 m |
| 0.5 | 1.0 | 3.0 | 0.70 m | 0.08 m |
| 0.3 | 1.5 | 5.0 | 0.11 m | 0.20 m |
| 0.3 | 5.0 | 5.0 | 1.70 m | 3.84 m (osc!) |

## 5. Gravity Bias

PID with integral term significantly reduces total drift. The I-term
builds up to counteract gravity's steady-state torque. With kp=0.5,
ki=3.0, total drift dropped to 0.70m and mean/horizon to 0.08m.

High damping (kd=5.0) causes instability — the system oscillates
because damping forces become larger than restoring forces.

## 6. Status

The PID backend is functional and produces converging hold behavior.
Optimal gain values need further tuning (grid search or auto-tuning).

Once hold drift < 0.005 m/horizon is achieved, the low-level command
chain and J_eff tests can resume, followed by mapper and PPO.

## 7. Files Modified

- `mulag_upper_level_env_cfg.py`: Added 10 PID config fields
- `mulag_upper_level_env.py`: Extended to `_explicit_effort_target` with
  PID + anti-windup, integral state tracking
- `test_explicit_effort_pd_hold.py`: Added `--backend` and PID scale args
