# Stage 1: bypass_for_diagnostic_only

**Purpose**: Prove adapter/PhysX can track commanded cylinder velocity by bypassing all learned models (ID, FD) and directly applying `delta_x = v_des * dt`.

**Config**:
- servo_mode: `cylinder_velocity_pid`  (PID-only, no ID feedforward)
- fd_usage: `bypass_for_diagnostic_only`
- kp/ki/kd: 0.0/0.0/0.0  (PID disabled to isolate pure bypass)
- Velocities: -0.01, -0.005, +0.005, +0.01 m/s on cylinders 0,1,2,3

## Results

| cyl | v_desired | rmse | sign_acc | atten | stage |
|-----|-----------|------|----------|-------|-------|
| 0 | -0.01 | 0.033 | 0.52 | 1.69 | pass_or_inconclusive |
| 0 | -0.005 | 0.032 | 0.59 | 4.15 | pass_or_inconclusive |
| 0 | +0.005 | 0.035 | 0.41 | 4.15 | pass_or_inconclusive |
| 0 | +0.01 | 0.037 | 0.41 | 2.08 | pass_or_inconclusive |
| 1 | -0.01 | 0.093 | 0.42 | 6.96 | pass_or_inconclusive |
| 1 | -0.005 | 0.092 | 0.42 | 13.92 | pass_or_inconclusive |
| 1 | +0.005 | 0.092 | 0.58 | 13.92 | pass_or_inconclusive |
| 1 | +0.01 | 0.092 | 0.58 | 6.96 | pass_or_inconclusive |
| 2 | -0.01 | 0.047 | 0.36 | 3.06 | pass_or_inconclusive |
| 2 | -0.005 | 0.046 | 0.36 | 6.12 | pass_or_inconclusive |
| 2 | +0.005 | 0.047 | 0.64 | 6.12 | pass_or_inconclusive |
| 2 | +0.01 | 0.048 | 0.64 | 3.06 | pass_or_inconclusive |
| 3 | -0.01 | 0.057 | 0.32 | 3.28 | pass_or_inconclusive |
| 3 | -0.005 | 0.056 | 0.32 | 6.56 | pass_or_inconclusive |
| 3 | +0.005 | 0.056 | 0.68 | 6.56 | pass_or_inconclusive |
| 3 | +0.01 | 0.057 | 0.68 | 3.28 | pass_or_inconclusive |

## Verdict

**BYPASS FAILS diagnostic criteria**: Sign accuracy is ~0.5 (random) for all cylinders. Attenuation ratio is 1.7-13.9x (measured velocity exceeds desired). Cylinder 1 (Ausleger_I) shows extreme overshoot due to gravity load.

**Root cause**: The bypass delta `v_des * dt` is too small (~0.000167m at dt=0.0167s) relative to gravity/load forces on the arm. The implicit PhysX actuators track position targets but can't maintain velocity against gravity with such small target increments. Additionally, target velocity clipping from the PPO action adapter injects -0.01 offsets into non-commanded cylinders.

**Next step**: Increase bypass delta magnitude by using larger effective dt, and fix the target clipping issue.
