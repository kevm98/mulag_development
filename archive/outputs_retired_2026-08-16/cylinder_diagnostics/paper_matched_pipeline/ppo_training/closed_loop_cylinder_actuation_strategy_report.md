# Closed-Loop Cylinder Actuation Strategy Report

## 1. PhysX/IsaacLab Closed-Loop Mechanism

The Mulag arm uses a **closed kinematic loop**: the arm links form a chain, and
the hydraulic cylinders form parallel loops that close the chain. In PhysX:

- The arm revolute joints + cylinder prismatic joints form a tree articulation
- `PhysicsPrismaticJoint` prims connect `Rohr` (tube) ↔ `Kolben` (piston) bodies
- Additional `constraint_joint_*` DOFs close the kinematic loop
- The closed loop creates dependent DOFs whose motion is determined by the
  constraint solver, not independent position drives

This matches the official PhysX documentation: "loops can be closed by adding
rigid-body joints between articulation links."

## 2. Are Cylinders Articulation DOFs or Constraints?

**Both.** The cylinder prismatic joints (DOFs 2,3,11,16) ARE IsaacLab articulation
DOFs. They are named `Drehzapfen_joint`, `Ausleger_I_joint`, etc. but are
`PhysicsPrismaticJoint` prims in the USD. They connect cylinder piston/tube
bodies directly.

The additional `constraint_joint_*` DOFs (0,4,5,6,14,15,17,18,19) are secondary
closed-loop couplers with [-inf,inf] limits and zero stiffness.

## 3. Do Cylinder Constraints Have Drives?

**Yes.** Each of the 4 active joints has a `drive:linear` with:
- `type = force` (position control via PD)
- `stiffness` matching the ImplicitActuatorCfg values
- `damping`, `maxForce`, `targetPosition` all properly set

The drives exist in the USD and their properties reach PhysX.

## 4. Can They Be Commanded Directly?

**Position targets are written but ignored.** `set_joint_position_target()` writes
targets to the articulation, stiffness/damping are present, but
`get_dof_actuation_forces()` returns **zero**. PhysX's constraint solver
determines these DOFs' motion in the closed loop.

**Explicit effort targets work.** `set_joint_effort_target()` successfully
delivers torque. `tau_app == tau_cmd` (verified).

## 5. Is the Current Revolute-Joint Interface Correct?

The interface IS correct — we're commanding the right joints. The issue is
that PhysX ignores position drives on closed-loop DOFs. Explicit effort
control bypasses this limitation.

## 6. Recommended Actuation Strategy

**Use explicit effort PD** for the 4 cylinder prismatic joints (DOFs 2,3,11,16).
This is Solution B (acceptable temporary diagnostic) and should be documented
as "not final hydraulic actuation."

The PIDF pipeline continues to produce `delta_x_des_H` (cylinder displacement),
which maps to prismatic joint target changes. The explicit PD converts
position errors to joint efforts.

## 7. Minimum Fix

1. Replace `set_joint_position_target` with explicit PD via `set_joint_effort_target`
   in the `_apply_action` path (when task_space mode is active)
2. Tune PD gains to achieve EE drift < 0.005 m/horizon
3. Accept that this bypasses the PhysX implicit position drive

## 8. Can Mapper/PPO Resume?

**Yes, after hold drift is fixed.** The explicit effort PD proves joints are
controllable. Once PD gains achieve hold drift < 0.005 m/horizon, the
J_eff measurement, mapper tests, and PPO can proceed with the explicit
PD backend.
