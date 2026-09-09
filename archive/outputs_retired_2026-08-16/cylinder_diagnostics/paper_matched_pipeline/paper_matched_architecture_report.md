# Paper-Matched Architecture Design Report

**Paper**: [High Precision Hydraulic Excavator Control for Heavy-Duty Grading](https://arxiv.org/pdf/2605.09465)
**Authors**: Lennart Werner, Pol Eyschen, Sean Costello, Andrei Cramariuc, Marco Hutter (ETH Zurich / Hexagon AB)
**Date**: 2026-07-01

---

## 1. What the Paper Does

The paper presents a two-layer control architecture for autonomous excavator grading:

```
Upper level:  MPC path tracking controller
               -> commands desired joint velocities [rad/s]

Mapping:      joint-to-cylinder kinematic mapping
               -> converts joint velocities to cylinder velocities [m/s]

Lower level:  Hydraulic joint velocity controller
               -> hydraulic FF model (LUT) maps desired cylinder velocity -> valve command
               -> parallel PID corrects residual velocity tracking error
               -> FF + PID summed, sent to valves
```

The lower-level controller is calibrated per machine before any upper-level policy runs. For LS architectures a simple 1D LUT suffices; for NFC a novel 2D pressure-adaptive LUT is fit from a compact calibration dataset (~20 min).

Key results: RMSE 1.4-1.8 cm in heavy grading, 2.6x better than commercial system.

---

## 2. What We Copy

| Paper Element | Our Implementation |
|---|---|
| Two-layer architecture (path tracking + hydraulic servo) | Upper PPO/task-space + lower cylinder velocity servo |
| Low-level cylinder velocity servo | `_apply_cylinder_velocity_servo()` in `mulag_upper_level_env.py` |
| Feedforward + PID parallel structure | ID as FF map + PID residual |
| Calibration before upper-level policy | ID/FD fine-tuning + servo tuning before PPO training |
| Transient response identification | `tune_cylinder_velocity_servo.py` step-response tests |
| Iterative test-isolate-fix-retest | `_stage_failure()` decision rules in tuning script |

---

## 3. What We Adapt

| Paper Element | Adaptation to Our Learned Pipeline |
|---|---|
| MPC path tracking | PPO policy or task-space PD controller |
| Hydraulic LUT (LS/NFC) | Learned neural ID feedforward model (HistoryGRU or MLP) |
| Cylinder velocity tracking | Cylinder-space servo with ID feedforward + PID feedback |
| Plant model | ID -> FD -> adapter -> PhysX chain |
| Calibration (20 min, real machine) | IsaacLab/PhysX cylinder-space calibration + ID/FD fine-tuning on collected simulation data |
| Pilot pressure / CAN commands | Normalized valve commands [-1, 1] through FD to cylinder position targets |

---

## 4. What Is Out of Scope

- Soil/grading interaction (our task is spatial tracking)
- Full surface MPC planning
- Real excavator NFC/LS pressure model (we have learned ID/FD)
- Direct valve PPO (explicitly forbidden)
- USD edits unless actuator audit proves necessary
- Commercial controller comparison
- Real-world deployment calibration

---

## 5. Pipeline Topology

```
Observation
    |
    v
Upper-level PPO / PD controller
    |
    v
desired EE velocity OR desired cylinder velocity
    |
    v
Joint-to-cylinder kinematic mapping (DLS / empirical Jacobian)
    |
    v
v_cyl_desired [m/s]  (target cylinder velocity)
    |
    v
Lower-level cylinder velocity servo:
    |
    +---> ID feedforward: valve_ff = ID(v_des, ext, ext_vel, history)
    |
    +---> PID feedback:    valve_pid = Kp*e + Ki*int_e + Kd*de
    |
    v
valve_total = clamp(valve_ff + valve_pid, limits)
    |
    v
FD prediction/calibration: delta_x = FD(valve_total, ext, ext_vel, history)
    |
    v
adapter: cylinder-space delta -> PhysX joint position targets
    |
    v
PhysX implicit actuators -> measured cylinder velocity
    |
    +---> feedback to servo error computation
```

---

## 6. Configuration Interface

```python
# Low-level servo mode
low_level_servo_mode = "none"  # "none" | "cylinder_velocity_pid" | "id_ff_plus_cylinder_velocity_pid"

# PID gains (per-cylinder)
cyl_vel_kp = [6.0, 6.0, 6.0, 6.0]
cyl_vel_ki = [0.0, 0.0, 0.0, 0.0]
cyl_vel_kd = [0.0, 0.0, 0.0, 0.0]
cyl_vel_integral_limit = 0.05
cyl_vel_cmd_limit = 0.05  # m/s
valve_cmd_limit = 1.0
servo_deadband_mps = 1.0e-4
servo_rate_limit = 0.0  # 0 = no limit

# FD usage in servo
fd_servo_usage = "prediction_only"
# "prediction_only"              -> delta_x = FD(valve_total)
# "prediction_plus_feedback_delta" -> delta_x = FD(valve_total) + Kx * (v_des - v_meas) * dt
# "bypass_for_diagnostic_only"   -> delta_x = v_des * dt  (diagnostic only)

servo_tracking_delta_gain = 0.0  # Kx when prediction_plus_feedback_delta
```

---

## 7. State Buffers

```python
_cyl_vel_desired[envs, 4]        # desired cylinder velocity [m/s]
_cyl_vel_measured[envs, 4]       # measured cylinder velocity [m/s]
_cyl_vel_error[envs, 4]         # e = v_des - v_meas
_cyl_vel_error_integral[envs, 4] # clamped integral term
_cyl_vel_error_prev[envs, 4]    # previous error for derivative
_valve_ff[envs, 4]              # ID feedforward component
_valve_pid[envs, 4]             # PID feedback component
_valve_total[envs, 4]           # clipped FF + PID sum
_fd_delta_raw[envs, 4]          # raw FD delta extension
_fd_delta_corrected[envs, 4]    # after gain/blend/correction
_servo_tracking_rmse[envs, 4]   # per-cylinder tracking RMSE
```

---

## 8. Control Law (already implemented)

```
e = v_des - v_meas
int_e = clamp(int_e + e * dt, ±integral_limit)
de = (e - e_prev) / dt

valve_ff = ID(v_des, x_current, xdot_current, history)
valve_pid = Kp*e + Ki*int_e + Kd*de
valve_total = clamp(valve_ff + valve_pid, ±valve_limit)

fd_delta = FD(valve_total, x, xdot, history)
delta_x_cmd = fd_delta + (servo_tracking_delta_gain * e * dt)
```

---

## 9. FD Servo Usage Modes (new)

| Mode | Formula | Status |
|---|---|---|
| `prediction_only` | `delta_x = FD(valve_total)` | Default for normal operation |
| `prediction_plus_feedback_delta` | `delta_x = FD(valve_total) + Kx * e * dt` | Used when FD alone is too weak |
| `bypass_for_diagnostic_only` | `delta_x = v_des * dt` | Diagnostic — verifies PhysX/adapter can track |

---

## 10. Pass Conditions

1. **Low-level servo**: sign_accuracy > 0.9, velocity ratio 0.5-1.5 for moderate commands, no instability
2. **PhysX actuator audit**: cylinder joints can track direct position targets at control rate
3. **ID feedforward**: valve_ff has correct sign and usable magnitude for all velocity commands
4. **FD model**: delta moves cylinder in correct direction
5. **Upper-level**: PPO/PD can drive EE to target with servo engaged

