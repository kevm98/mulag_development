# Gravity Compensation for Hydraulic Actuators: Research Summary

## Werner et al. (2025) Approach

**Two architectures handled differently**:

**LS (Load Sensing)**:
- Hydraulically stiff — no explicit gravity compensation needed
- Simple 1D LUT: desired_flow -> valve_command
- Gravity load creates pressure demand, LS pump adjusts displacement automatically

**NFC (Negative Flow Control)**:
- Hydraulically soft — load-dependent flow
- **Inertia compensation**: subtract inertia forces from pressure-derived force
- Prevents positive feedback loop (command -> acceleration -> increased force -> increased command)
- **Pressure-adaptive 2D LUT**: maps (desired_flow, load_pressure) -> valve_command
- Gravity compensation is implicit in the pressure-adaptive mapping

## Why Our Simulation Differs

Implicit position-target actuators (PhysX drives):
- Position targets -> spring-damper force -> joint motion
- No hydraulic flow/pressure feedback
- Gravity produces **position error** proportional to τ_gravity / Kp
- Small position deltas excite underdamped oscillation

## Implemented Solution: Position-Ramp PI + Anti-Gravity Integral

```
x_cmd += v_des * dt           # integrate velocity into position ramp
e_pos = x_cmd - x_meas         # position error
e_pos_int += e_pos * dt        # integral accumulates anti-gravity bias
dx_cmd = Kp*e_pos + Ki*e_pos_int # PI control
```

**Preserving integral on neutral hold**: The integral term holds the anti-gravity bias memory. When the cylinder is commanded to hold, the integral maintains the bias to counteract gravity.

## Results

With kp=3.0, ki=1.0 on cylinder 0:
- Mean velocity tracks command direction correctly
- Attenuation ratio ~1.0-1.6 (close to desired)
- Sign accuracy ~0.7 for gravity-assisted, ~0.3 against gravity
- RMSE ~0.03-0.08 m/s

## Remaining Limitations

1. **Oscillatory tracking**: Implicit actuators inherently oscillate at the control rate
2. **Poor negative sign accuracy**: Cylinder can't fight gravity in negative direction as well as positive
3. **No load-adaptive mapping**: Without hydraulic model, gravity compensation is reactive (integral) rather than feedforward

## Possible Improvements

1. **Gravity feedforward**: Compute gravity torque from robot model, convert to position offset
2. **Higher stiffness**: Increase implicit actuator stiffness for boom cylinder
3. **Damping injection**: Add velocity feedback damping term
4. **Variable gains**: Per-cylinder PI gains based on gravity load
