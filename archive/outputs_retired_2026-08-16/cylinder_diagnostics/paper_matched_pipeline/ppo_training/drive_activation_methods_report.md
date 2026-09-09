# Drive Activation Methods Report

## Summary

**Explicit effort PD works.** `set_joint_effort_target()` successfully delivers
torque to the 4 active revolute joints. PhysX position drives (via
`ImplicitActuatorCfg`) do NOT produce torque — the drive mode is inactive.

## Methods Tested

### Method A: ImplicitActuatorCfg + set_joint_position_target (baseline)

- Stiffness/damping reach PhysX (verified via `_dof_stiffnesses` readback)
- `set_joint_position_target()` writes targets every substep
- Applied effort: **ZERO** (via `get_dof_actuation_forces()`)
- Verdict: **FAIL — position drive mode not active**

### Method E: Explicit Effort PD

- Compute tau = kp * (q_target - q) - kd * qdot
- Call `robot.set_joint_effort_target(tau, joint_ids=cyl_ids)`
- Applied effort: **MATCHES commanded torque** (tau_app == tau_cmd)
- Verdict: **WORKS**

## Explicit Effort PD Results

### Gravity OFF — Single Joint Step Test
- Ausleger_I +0.05 rad target
- Joint moves toward target (overshoots, needs gain tuning)
- Applied torque matches commanded torque exactly

### Gravity ON — Zero-Command Hold
| Method | Kp (Dreh/AI/AII/MS) | Drift (5s) |
|--------|---------------------|------------|
| PD moderate | 30M/12M/2M/2M | 0.25 m |
| PD baseline | 15M/6M/1M/1M | 0.17 m |
| PI-D high | 75M/30M/5M/5M | 0.61 m |
| PD+I target refresh | 30M/12M/2M/2M | 0.76 m |

Steady-state errors remain because P-only cannot eliminate gravity bias.
I-term tuning needs refinement (anti-windup interaction).

## Next Steps

1. **Tune explicit PD gains for hold** — Separate gain-tuning task
2. **Or fix PhysX position drives** — USD-level drive mode enablement
3. **Re-enable mapper/Jacobian tests** once EE drift < 0.005 m/horizon

## Conclusion

The articulation IS controllable via `set_joint_effort_target`.
The PhysX implicit position drive mode is inactive.
Explicit effort PD is a viable temporary backend until USD drives are fixed.
