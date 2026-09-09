# Stage 3: prediction_plus_feedback_delta

**Purpose**: Paper-matched practical servo — ID FF + PID + FD prediction + feedback delta correction.

**Config**: servo_mode: `id_ff_plus_cylinder_velocity_pid` and `cylinder_velocity_pid` (PID-only variant), fd_usage: `prediction_plus_feedback_delta`

## PID-only variant (cylinder 0, kp=6/ki=0.5/kd=0.1, correction_gain=0.5)

| v_desired | rmse | sign_acc | atten |
|-----------|------|----------|-------|
| -0.01 | 0.029 | 0.58 | 1.30 |
| -0.005 | 0.029 | 0.57 | 3.75 |
| +0.005 | 0.031 | 0.42 | 3.75 |
| +0.01 | 0.034 | 0.42 | 1.87 |

## PID-only variant (cylinder 0, higher velocity, kp=15/ki=3/kd=0.3, correction_gain=2.0)

| v_desired | rmse | sign_acc | atten |
|-----------|------|----------|-------|
| -0.02 | 0.037 | 0.53 | 0.85 |
| -0.01 | 0.032 | 0.60 | 2.14 |
| +0.01 | 0.038 | 0.40 | 2.14 |
| +0.02 | 0.044 | 0.40 | 1.07 |

## PID-only variant (cylinder 1, kp=10/ki=2/kd=0.2, correction_gain=1.0)

| v_desired | rmse | sign_acc | atten |
|-----------|------|----------|-------|
| -0.01 | 0.072 | 0.38 | 4.92 |
| -0.005 | 0.079 | 0.33 | 11.04 |
| +0.005 | 0.078 | 0.68 | 11.04 |
| +0.01 | 0.079 | 0.68 | 5.52 |

## Verdict

**PID feedback delta partially works** but can't overcome gravity bias. Cylinder 1 (Ausleger_I) shows 5-11x overshoot due to gravity load on the boom. Cylinder 0 (Drehzapfen, less gravity-loaded) shows better attenuation (1-4x) but still poor sign accuracy (~0.5).

**Higher gains don't help**: increasing kp from 6 to 15 and correction_gain from 0.5 to 2.0 doesn't improve sign accuracy — the issue is directional, not magnitude.

**Root cause chain**:
1. ID model dead for velocity targets → no feedforward
2. PID + feedback delta produces usable motion magnitude
3. But gravity/load bias overwhelms directional control, especially for Ausleger_I
4. Target clipping from PPO action adapter injects spurious motion into non-commanded cylinders

**Next step**: Fix the debug velocity path to bypass PPO action adapter clipping. Then retest with PID-only servo at higher feedback delta gains.
