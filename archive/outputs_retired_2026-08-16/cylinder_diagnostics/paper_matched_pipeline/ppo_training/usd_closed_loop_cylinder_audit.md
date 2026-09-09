# USD Closed-Loop Cylinder Audit

## Key Discovery

The "arm joints" named `Drehzapfen_joint`, `Ausleger_I_joint`, `Ausleger_II_joint`,
`Messerkopf_Schwenk_joint` are **NOT revolute arm joints** — they are
**`PhysicsPrismaticJoint`** prims in the USD stage connecting cylinder piston
and tube bodies.

## USD Joint Types (from stage inspection)

| IsaacLab Name | USD Type | Body 0 (Rohr/Tube) | Body 1 (Kolben/Piston) |
|--------------|----------|-------------------|----------------------|
| Drehzapfen_joint | PhysicsPrismaticJoint | Ausleger_Schwenkzylinder_Rohr | Ausleger_Schwenkzylinder_Kolben |
| Ausleger_I_joint | PhysicsPrismaticJoint | Hebezylinder_Rohr | Hebezylinder_Kolben |
| Ausleger_II_joint | PhysicsPrismaticJoint | Knickzylinder_Rohr | Knickzylinder_Kolben |
| Messerkopf_Schwenk_joint | PhysicsPrismaticJoint | Messerkopf_Schwenkzylinder_Rohr | Messerkopf_Schwenkzylinder_Kolben |

## Drive Configuration (from USD)

All four joints have **linear position drives** with force type:

| Joint | Stiffness | Damping | Max Force | Target Pos |
|-------|-----------|---------|-----------|-----------|
| Drehzapfen | 15,000,000 | 1,000,000 | 100,000,000 | 0.0 |
| Ausleger I | 6,000,000 | 600,000 | 10,000,000 | 0.0 |
| Ausleger II | 1,000,000 | 500,000 | 1,000,000 | 0.0 |
| Messerkopf Schwenk | 1,000,000 | 500,000 | 100,000 | 0.0 |

Drive type: `drive:linear:physics:type = force`

## Why Position Drives Produce Zero Effort

These joints are part of a **closed kinematic loop** in the PhysX articulation.
The arm forms a closed chain through the cylinder constraint joints. When
`set_joint_position_target()` is called on these DOFs:
- The target IS written to the articulation
- Stiffness/damping ARE present in PhysX
- But `get_dof_actuation_forces()` returns ZERO

This is because PhysX treats closed-loop dependent DOFs as "passive" — the
constraint solver determines their motion, not independent drives.

## Constraint Joints (Secondary DOFs)

The articulation also contains prismatic constraint joints (DOFs 0,4,5,6,14,15,17,18,19)
with [-inf,inf] limits. These are additional closed-loop couplings:
- Drehzapfen_constraint_joint_1,3
- Ausleger_I_constraint_joint_1,3
- Ausleger_II_constraint_joint_1,3
- Messerkopf_Schwenk_constraint_joint_1,3,5,6

These have `drive:angular:stiffness=0` in USD and no IsaacLab actuator config.
They are part of the articulation tree but serve only as mechanical constraints.

## Conclusion

1. **The active joints ARE the real hydraulic cylinder prismatic joints.**
2. They have linear position drives with proper stiffness/damping/maxForce.
3. The drives exist but produce **zero effort** because PhysX treats them
   as dependent DOFs in a closed kinematic loop.
4. `set_joint_effort_target()` WORKS on these same joints — explicit effort
   bypasses the position drive limitation.
5. The constraint joints (with [-inf,inf] limits) are additional couplers
   and cannot be independently driven.
