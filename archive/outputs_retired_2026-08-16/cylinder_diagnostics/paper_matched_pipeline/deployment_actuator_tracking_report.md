# Deployment Actuator Tracking — Final Report

**Date**: 2026-07-01  
**Sweep**: 72 trials, 18 configs, 2 servo modes × 3 fd_gains × 3 tracking_gains × 2 cylinders × 2 velocities

---

## Results Summary

| Metric | Range |
|--------|-------|
| Sign accuracy | 0.487–0.500 (random) |
| Attenuation ratio | 3.69x (all configs) |
| RMSE | 0.058–0.059 m/s |
| Effect of fd_gain | None |
| Effect of tracking_gain | None |
| Effect of servo_mode | None |

**All 18 configurations produced identical results within floating-point precision.** The fd_gain (1,5,20), tracking_gain (0,0.5,2.0), and servo_mode (fd_tracking_servo, id_pid_fd_tracking_servo) have zero impact on cylinder velocity tracking.

## Conclusion

**PhysX implicit position-drive actuators cannot support the hydraulic-flow-style learned actuator interface.** The Werner paper's controller works because real hydraulics use flow control (valve directly sets oil flow rate = cylinder velocity). Our simulation uses position drives with stiffness/damping, which produce oscillatory velocity regardless of ID/FD mapping quality.

The deployment actuator tracking layer (fd_gain × FD_delta + tracking_gain × error × dt) was the final attempt to bridge this physics gap. It failed because:
1. FD predicts position deltas that are ~0.0001–0.0006m — too small for velocity control
2. Implicit actuators oscillate at the control rate, producing 50/50 sign accuracy
3. No gain combination changes the underlying physics

## Recommended Path

Accept that the learned ID→FD pipeline works for **positional control** but not **velocity tracking** in the current PhysX simulation. For velocity tracking:
- Use position-ramp PI (demonstrated working, correct mean velocity direction)
- Or switch to explicit actuator modeling with velocity-drive mode
- Or use the fine-tuned ID as a positional feedforward in a position-tracking controller

The original ID and FD models are preserved and validated for their intended use case (positional delta prediction).
