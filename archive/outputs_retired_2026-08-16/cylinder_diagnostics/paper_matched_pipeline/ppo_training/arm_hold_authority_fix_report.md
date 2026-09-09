# Arm Hold Authority Fix Report

## Goal

Make the arm hold its reset pose under gravity with zero command.

## Tests Run

### 1. Target Writing Verification

The debug target mechanism (`set_debug_manual_cylinder_targets`) correctly writes
cylinder position targets. Verified by printing targets vs actuals during hold:
- Targets remain constant throughout hold test
- `_apply_action()` reads `_debug_manual_cylinder_position_target` and calls
  `set_joint_position_target()` at every physics substep (decimation=2)

Result: **Target writing is correct.** Targets are persistent every step.

### 2. Root Pinning

The Unimog base must be pinned to prevent the vehicle from tipping.
`write_root_pose_to_sim()` and `write_root_velocity_to_sim()` are called each step.

Result: **Base stays static.** Root position drift < 0.001m over 5s.

### 3. Stiffness Sweep (Positive Values)

| Scale | Drehzapfen Stiffness | EE Drift (3s) |
|-------|---------------------|---------------|
| 1× | 15 M | 0.222 m |
| 50× | 750 M | 0.222 m |
| 100× | 1.5 B | 0.422 m |
| 200× | 3.0 B | 0.289 m |

### 4. Stiffness Sweep (Negative Values — possible sign test)

| Scale | EE Drift (3s) |
|-------|---------------|
| 1× | 0.155 m |
| 2× | 0.513 m |
| 5× | 0.513 m |
| 10× | 0.513 m |
| 20× | 0.513 m |

Negative stiffness at 2×+ produced identical drift (0.513m), suggesting negative
values are clamped to zero by PhysX. But even at 1× negative, drift was 0.155m
(better than positive 1× = 0.222m).

### 5. Drift Behavior

Cylinders drift AWAY from their hold targets regardless of stiffness magnitude:

- Cyl 0: drifts ~6mm in 3s (target error)
- Cyl 1: drifts ~17cm in 3s
- Cyl 2: drifts ~14cm in 3s
- Cyl 3: drifts ~7cm in 3s

At 200× positive stiffness (3 BILLION N/m), cylinders still drift away.

## Root Cause Analysis

The PD stiffness changes alone cannot fix the hold problem. At 3B N/m stiffness,
position errors of 0.1-17cm should produce forces of 300M-51B N. These should
easily overcome gravity. Yet the arm still drifts.

Possible explanations:

1. **PhysX position drive force is saturated internally** — The USD joint
   definition may have force/torque limits that override the Python config.
2. **Joint axis convention mismatch** — The USD joint axis direction is opposite
   to what the position drive expects, causing the drive to push in the wrong
   direction (then clamped/clipped).
3. **The arm is not actually controllable via position drives** — The cylinder
   joints may be kinematic rather than dynamic, or the articulation root is
   not properly configured for driven control.
4. **Joint units mismatch** — If the Drehzapfen is a revolute joint, the
   position error is in radians (not meters), and stiffness is in N·m/rad (not N/m).
   The gravitational torque might be much larger than expected.

## Remaining Work

The hold authority issue requires deeper PhysX/USD-level investigation:

1. Verify the USD joint type (prismatic vs revolute) and drive axis direction
2. Check if `ImplicitActuatorCfg` stiffness values actually reach PhysX or are
   overridden by the USD authoring
3. Test with `ArticulationCfg.actuators` set to `DCMotorCfg` or `IdealPDActuatorCfg`
   instead of `ImplicitActuatorCfg`
4. Read back actual joint forces/torques from `get_dof_stiffness()` after simulation
   start to confirm values are applied

## Interim Conclusion

The arm CANNOT hold position under gravity with the current actuator configuration.
PD stiffness tuning (1× to 200×) does not improve hold authority — drift remains
0.15-0.42m per 3 seconds regardless. Negative stiffness values are likely clamped
to zero by PhysX.

**Until the arm can hold position with zero command, no PPO, mapper, or Jacobian
work can proceed.** The low-level bridge from cylinder commands to EE position is
non-functional because the actuators cannot maintain position against gravity.
