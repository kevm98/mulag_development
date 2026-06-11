# Prismatic Drive Authority

Status: direct and learned preflight logic implemented; final classification requires Isaac runtime.

## Static Findings

- The active hydraulic ROM path writes prismatic position targets with `robot.set_joint_position_target`.
- FD effort actuation is disabled in the active hydraulic ROM task.
- Torque is not the active command path; torque buffers are explicitly zeroed in final ROM mode.
- Asset config uses an explicit `ImplicitActuatorCfg` joint list instead of `".*"`.
- The Python config leaves stiffness, damping, effort limit, and velocity limit as `None`, so the authoritative values are USD/runtime authored and require Isaac introspection.

## Implemented Runtime Decision

The built-in asset preflight applies direct visible position target pulses to the four command joints and classifies:

- `TARGET_NOT_CHANGING`;
- `TARGET_CHANGING_DRIVE_NOT_TRACKING`;
- `DRIVE_TRACKING_LINKAGE_NOT_MOVING`;
- `FULL_TRANSMISSION_WORKING`.

The learned preflight then checks whether the full model chain changes the accumulated target, whether measured prismatic q follows, and whether mechanical linkage joints move.

## Requested Runtime Classification Mapping

- `TARGET_NOT_CHANGING`: accumulated target motion is below threshold.
- `TARGET_CHANGING_DRIVE_NOT_TRACKING`: target motion is meaningful but measured cylinder motion is below threshold.
- `DRIVE_TRACKING_LINKAGE_NOT_MOVING`: measured cylinder motion is meaningful but mechanical family motion is below threshold.
- `FULL_TRANSMISSION_WORKING`: target, cylinder, and linkage motion all pass thresholds.

## Current Static Classification

`REQUIRES_RUNTIME`.

Static code confirms the writer path and that torque/FD effort is disabled, but only Isaac can confirm drive force authority, target buffer readback, measured effort saturation, and linkage transmission.
