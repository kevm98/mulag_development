# PhysX Drive Activation Fix Report

## 1. Evidence: Original Position Drive Produced Zero Effort

- Stiffness values reach PhysX (confirmed via `_dof_stiffnesses` readback at 200×)
- `set_joint_position_target()` writes targets every substep (confirmed via debug prints)
- Position errors exist (Ausleger_I: -0.118 rad)
- Expected torque: 708,000 N·m (from 6M × 0.118)
- Actual applied torque: **ZERO** (from `get_dof_actuation_forces()`)
- This confirms: **PhysX implicit position drive mode is NOT active**

## 2. Which Activation Method Worked

**Explicit effort PD works:**
- `robot.set_joint_effort_target(tau, joint_ids=cyl_ids)` delivers torque
- `tau_app == tau_cmd` (verified)
- Joints respond to commanded torque

## 3. Explicit Effort PD Results

| Test | Result |
|------|--------|
| Gravity OFF, single joint step | Joint moves toward target ✓ |
| Gravity ON, zero-command PD hold | EE drift 0.17m over 5s |
| Gravity ON, PI-D hold | EE drift 0.61m (integral overshoot) |

PD-only at baseline gains (15M/6M/1M/1M) produces 0.17m drift — much better
than passive 2m drift, but still above 0.03m target.

## 4. USD Patch

Not tested. The revolute arm joints have actuators attached but the PhysX
drive mode is inactive. A USD patch to enable `drive:type="force"`
`drive:target_type="position"` on each revolute joint may fix position drives.

## 5. Joint Types

All active joints (DOFs 2,3,11,16,20) are REVOLUTE with limits in radians.
The prismatic cylinder constraint joints are passive couplings and cannot
be independently driven.

## 6. Real Cylinder Joints

8 hydraulic cylinder bodies exist in the USD (Kolben/Rohr = piston/tube).
Their prismatic constraint joints have [-inf,inf] limits and NO actuators.
They are closed-loop mechanical couplings, not independent DOFs.

## 7. Final Hold Drift

| Method | Drift (5s) |
|--------|-----------|
| Passive (no drive) | 2.0 m |
| Implicit position drive | 2.0 m (same — zero effort) |
| Explicit PD (baseline) | 0.17 m |
| Explicit PD (moderate) | 0.25 m |

## 8. Can Mapper/Jacobian Work Resume?

**Partially.** The explicit effort PD proves the joints ARE controllable.
However, EE drift of 0.17m per 5s (0.034 m/s) is still above the required
0.005 m/horizon threshold. Gain tuning or I-term addition is needed before
the hold can meet the pass criterion.

The explicit effort PD approach can serve as temporary backend for
mapper/Jacobian tests IF the hold drift is first reduced below threshold
through gain tuning.
