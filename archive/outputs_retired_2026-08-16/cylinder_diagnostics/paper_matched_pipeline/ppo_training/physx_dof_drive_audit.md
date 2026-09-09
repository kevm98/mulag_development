# PhysX DOF Drive Audit

## 1. Did 200x Stiffness Reach PhysX?

**YES.** PhysX readback confirmed:

| Joint | cfg Stiffness (200x) | PhysX Stiffness | Match? |
|-------|---------------------|-----------------|--------|
| Drehzapfen_joint | 3,000,000,000 | 3,000,000,000 | ✓ |
| Ausleger_I_joint | 1,200,000,000 | 1,200,000,000 | ✓ |
| Ausleger_II_joint | 200,000,000 | 200,000,000 | ✓ |
| Messerkopf_Schwenk_joint | 200,000,000 | 200,000,000 | ✓ |

Stiffness values are correctly applied to PhysX.

## 2. Joint Types

**All active "cylinder" joints are REVOLUTE**, not prismatic:

| Joint | Limits | Type |
|-------|--------|------|
| Drehzapfen_joint | [-0.087, 0.268] rad | REVOLUTE |
| Ausleger_I_joint | [-0.151, 0.173] rad | REVOLUTE |
| Ausleger_II_joint | [-0.160, 0.140] rad | REVOLUTE |
| Messerkopf_Schwenk_joint | [-0.074, 0.096] rad | REVOLUTE |

These are arm rotational joints, not hydraulic cylinder prismatic joints.
The USD model has separate prismatic "constraint" joints (e.g.,
`Drehzapfen_constraint_joint_1`) that represent the actual cylinders,
but these are NOT being driven by the actuator config.

## 3. Applied Efforts: ZERO

**Position drives are producing ZERO torque despite large position errors:**

| Time | Drehzapfen error | Effort | Ausleger_I error | Effort |
|------|---------|--------|----------|--------|
| 0s | 0.0005 rad | 0.0 | -0.005 rad | 0.0 |
| 1s | 0.0052 rad | 0.0 | -0.118 rad | 0.0 |

With stiffness 6M and error 0.118 rad, the expected torque is 708,000 N·m.
Actual applied torque: **ZERO**.

## 4. Root Cause

The stiffness values reach PhysX, but the **drive mode is not active**.
`ImplicitActuatorCfg` sets stiffness/damping in the articulation, but
PhysX's internal PD controller only produces forces when the joint's
drive mode is set to position control. The USD joints likely have
`drive:type="none"` or the articulation root doesn't have position
drives enabled.

The `set_joint_position_target()` calls write position targets to the
articulation view, but without the drive mode enabled, PhysX ignores them.

## 5. Conclusion

The arm is entirely passive — no PD forces are applied to hold position
against gravity. This explains:
- Why stiffness sweep (1× to 200×) had zero effect
- Why zero command produces identical drift to any command
- Why the arm sags 0.3m per second regardless of targets

The fix requires enabling position drive mode in the USD joint configuration
or using a different articulation setup that activates PhysX PD drives.
