# Articulation vs Constraint Interface Report

## Section A: Articulation DOFs (from IsaacLab)

| DOF | Name | USD Type | Bodies | Has Drive | Position Control? |
|-----|------|----------|--------|-----------|-------------------|
| 2 | Drehzapfen_joint | PhysicsPrismaticJoint | Ausleger_Schwenkzylinder_Rohr↔Kolben | YES (linear, force) | target written, effort=0 |
| 3 | Ausleger_I_joint | PhysicsPrismaticJoint | Hebezylinder_Rohr↔Kolben | YES (linear, force) | target written, effort=0 |
| 11 | Ausleger_II_joint | PhysicsPrismaticJoint | Knickzylinder_Rohr↔Kolben | YES (linear, force) | target written, effort=0 |
| 16 | Messerkopf_Schwenk_joint | PhysicsPrismaticJoint | Messerkopf_Schwenkzylinder_Rohr↔Kolben | YES (linear, force) | target written, effort=0 |

**These are the hydraulic cylinder prismatic joints.** They connect cylinder
tube (Rohr) to piston (Kolben) bodies. `set_joint_position_target()` writes
targets. Stiffness/damping reach PhysX. But `get_dof_actuation_forces()` = 0.

## Section B: Constraint Joints (Loop Closers)

| DOF | Name | Drive | Drive Stiffness | Can Command? |
|-----|------|-------|----------------|-------------|
| 0 | Drehzapfen_constraint_joint_1 | angular | 0 | NO — effort PD also gives 0 |
| 4 | Drehzapfen_constraint_joint_3 | angular | 0 | NO |
| 5 | Ausleger_I_constraint_joint_1 | angular | 0 | NO |
| 6 | Ausleger_II_constraint_joint_3 | angular | 0 | NO |
| 14 | Ausleger_II_constraint_joint_1 | angular | 0 | NO |
| 15,17,18,19 | Messerkopf_Schwenk_constraint_joint_* | angular | 0 | NO |

These are additional closed-loop linkage joints. They have no actuator config,
stiffness=0, and cannot be driven via position target or effort target.

## Decision

The 4 active joints (DOFs 2,3,11,16) ARE the correct cylinder interface.
They are prismatic joints between cylinder bodies with linear position drives.

The position drives exist in USD but PhysX ignores them because these DOFs
are part of a closed kinematic loop. The constraint solver determines their
motion, not independent drive forces.

**Explicit effort PD (`set_joint_effort_target`) works** on these same joints
— this bypasses the position drive limitation and directly applies forces
to the articulation DOFs.
