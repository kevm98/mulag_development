# Real Cylinder Joint Audit

## Finding

The USD model has actual hydraulic cylinder bodies (Kolben/Rohr = piston/tube)
connected via prismatic constraint joints. These constraint joints are PASSIVE —
they translate cylinder extension to arm joint rotation mechanically. They have
NO actuators and are not independently drivable.

## Cylinder Bodies Found

| Body | Description |
|------|-------------|
| Ausleger_Schwenkzylinder_Kolben | Boom swivel cylinder piston |
| Hebezylinder_Rohr | Lift cylinder tube |
| Ausleger_Schwenkzylinder_Rohr | Boom swivel cylinder tube |
| Hebezylinder_Kolben | Lift cylinder piston |
| Knickzylinder_Rohr | Knuckle cylinder tube |
| Knickzylinder_Kolben | Knuckle cylinder piston |
| Messerkopf_Schwenkzylinder_Rohr | Cutter head swivel cylinder tube |
| Messerkopf_Schwenkzylinder_Kolben | Cutter head swivel cylinder piston |

## Constraint Joints (Prismatic, Not Actuated)

| DOF | Joint Name | Type | Limits | Actuator |
|-----|-----------|------|--------|----------|
| 0 | Drehzapfen_constraint_joint_1 | PRISMATIC | [-inf,inf] | none |
| 4 | Drehzapfen_constraint_joint_3 | PRISMATIC | [-inf,inf] | none |
| 5 | Ausleger_I_constraint_joint_1 | PRISMATIC | [-inf,inf] | none |
| 6 | Ausleger_II_constraint_joint_3 | PRISMATIC | [-inf,inf] | none |
| 14 | Ausleger_II_constraint_joint_1 | PRISMATIC | [-inf,inf] | none |
| 15,17,18,19 | Messerkopf_Schwenk_constraint_joint_* | PRISMATIC | [-inf,inf] | none |

These are closed-loop constraint joints that couple cylinder bodies to arm links.
They cannot be position-driven directly.

## Active Revolute Arm Joints

| DOF | Joint Name | Limits (rad) | Actuator |
|-----|-----------|-------------|----------|
| 2 | Drehzapfen_joint | [-0.09, 0.27] | drehzapfen_cylinder |
| 3 | Ausleger_I_joint | [-0.15, 0.17] | ausleger_i_cylinder |
| 11 | Ausleger_II_joint | [-0.16, 0.14] | ausleger_ii_cylinder |
| 16 | Messerkopf_Schwenk_joint | [-0.07, 0.10] | messerkopf_schwenk_cylinder |
| 20 | Messerkopf_joint | [-1.57, 1.57] | messerkopf_passive_coupled |

## Decision

**Use the revolute arm joints as the control interface.** The cylinder constraint
joints are passive mechanical couplings and cannot be independently driven.
The `JointToCylinderAdapter` correctly maps between arm joint angles and
cylinder extensions for observation/logic purposes.

Fix: enable position drive mode on the revolute arm joints, or use
explicit effort PD (`set_joint_effort_target`) as temporary backend.
