# Active Joint Identity Audit

## Question: Are we controlling arm joints or cylinder joints?

### Current Active Joint IDs

`_cylinder_joint_ids` = [2, 3, 11, 16]:
- 2: `Drehzapfen_joint` — REVOLUTE, limits [-0.087, 0.268] rad
- 3: `Ausleger_I_joint` — REVOLUTE, limits [-0.151, 0.173] rad
- 11: `Ausleger_II_joint` — REVOLUTE, limits [-0.160, 0.140] rad
- 16: `Messerkopf_Schwenk_joint` — REVOLUTE, limits [-0.074, 0.096] rad

`_tool_joint_id` = 20:
- 20: `Messerkopf_joint` — REVOLUTE, limits [-1.571, 1.571] rad

### These are ARM REVOLUTE JOINTS, not cylinder prismatic joints.

The names ("Drehzapfen_joint" = pivot joint, "Ausleger_I_joint" = boom I joint,
etc.) are mechanical arm joint names, not hydraulic cylinder names.

### Real Cylinder Joints Exist in USD but Are Not Driven

The USD articulation has separate constraint joints that represent the
actual hydraulic cylinders:

- `Drehzapfen_constraint_joint_1` (DOF 0) — likely the hydraulic cylinder
- `Ausleger_I_constraint_joint_3` (DOF 1) — Ausleger I cylinder
- `Ausleger_I_constraint_joint_1` (DOF 5) — other end/constraint
- `Ausleger_II_constraint_joint_3` (DOF 6) — Ausleger II cylinder
- `Messerkopf_Schwenk_constraint_joint_*` (DOFs 15,17,18,19)

These are likely PRISMATIC joints (limits [-inf, inf]) representing
hydraulic cylinder extension. They have `(none)` actuator — NOT driven.

### Body Names Confirm

- `Ausleger_Schwenkzylinder_Kolben` = boom swivel cylinder piston
- `Hebezylinder_Rohr` = lift cylinder tube
- `Ausleger_Schwenkzylinder_Rohr` = boom swivel cylinder tube
- `Knickzylinder_Rohr` = knuckle cylinder tube
- `Messerkopf_Schwenkzylinder_Rohr` = cutter head swivel cylinder tube

The hydraulic cylinders exist as separate bodies and constraint joints,
but the controller drives the arm revolute joints directly.

### Structural Issue

The controller is designed to drive hydraulic cylinders (prismatic
extension) but is connected to arm revolute joints. The `JointToCylinderAdapter`
converts between joint angles and cylinder extensions, creating the
mental model that "cylinders" are being driven. But the actual PD
actuators are attached to revolute arm joints.

### Recommended Fix

Either:
A. Activate position drives on the revolute arm joints (they CAN hold
   position if the drive mode is enabled — the joint type is fine,
   the issue is the drive mode)
B. Remap the controller to drive the actual prismatic cylinder
   constraint joints
