# Stop-and-mow engineering report

Date: 2026-07-12

## Objective

Make the hydraulic arm—not truck translation—track meaningful ground paths. The
operating sequence is park, mow with the rams, lift, advance straight, brake, and repeat.
The curve must remain in front of the truck; a sweep must explicitly go out and back.

## Implemented architecture

`dls_solver/tests/stop_and_mow_gate.py` is a deterministic phase-state gate. During a
work phase it disables scripted vehicle motion, applies the wheel position brake, plans
in live truck-frame coordinates, and scores the cutter and truck separately. During
relocation it stops scoring the cutter path, holds/stows the arm, drives the wheels, then
brakes before the next work phase.

The arm cascade remains deliberately simple:

- cutter error -> cached 3x4 physics-measured Jacobian -> CPU DLS;
- desired cylinders -> measured-cylinder DLS;
- target increments -> locked ID/FD learned actuator -> PhysX targets;
- live cylinder and cutter measurements close the loops.

No MPC, OSC, analytic Jacobian slice, runtime FK teleport, model retraining, or RL was
introduced.

## Verified arm results

The following are from seed 17 with unchanged 5 cm tracking and non-vacuity thresholds.

| Parked work phase | Geometry | Goal path | Mean | p95 | Ground | Truck drift | Result |
|---|---|---:|---:|---:|---:|---:|---:|
| Sweep station 0 | lateral 0.35 m out and back; fixed truck-X | 0.798 m | 0.0282 m | 0.0344 m | 100% | 0.013 m | PASS |
| Bowed curve station 0 | lateral 0.35 m, forward bow 0.10 m, exact return | 1.135 m | 0.0273 m | 0.0354 m | 100% | 0.014 m | PASS |

For the curve, truck-frame X stayed in `[5.486, 5.586]` m, so the complete path was in
front of the truck. Both paths caused real ram motion; neither used truck motion to
generate its geometry.

The cached ground cutter Jacobian is rank 3/3 with singular values
`8.235, 5.695, 2.785` and condition number `2.96`. The locked learned actuator rank gate
and cylinder controller baseline were not changed.

## Combined state-machine status

The complete repeated state machine is **not repeatably accepted**. An earlier two-row
sweep run passed:

- station 0 p95 0.0386 m, ground 100%;
- 0.505 m forward / 0.029 m lateral relocation;
- station 1 p95 0.0313 m, ground 100%;
- aggregate arm work 2.086 m.

However, repeated runs after adding the forward-curve branch exposed non-repeatable
wheel/brake dynamics. A representative current sweep relocation reached 0.627 m rather
than 0.50 m; the following parked work phase then drifted and lost ground contact. Since
that result is not repeatable, the combined gate is reported FAIL even though the arm
work phases themselves are valid.

## Experiments rejected

- Continuous truck-driven mowing: rejected architecturally; it let truck motion create
  the apparent cutter path.
- 0.40 m forward semicircle: arm tracking passed, but its load pose caused severe lateral
  truck motion. Replaced with the smaller bowed lateral curve.
- Fixed single-pose mapped arc after relocation: cannot reliably return to the same
  closed-linkage branch; station homing errors exceeded 1.6 m.
- Steering pre-alignment under the wheel brake: created a large release impulse.
- Equal/common steering feedback: wrong-sign and corrected-sign trials either amplified
  lateral motion or bound the axle. Removed.
- Unpowered brake-release settling: hid 0.162 m lateral motion before scoring. Rejected as
  dishonest.
- Root-velocity zero as an ideal brake: produced articulation constraint impulses and
  worse lateral motion. Removed.
- Repeating several arm sweeps at one station: hydraulic hysteresis and cutter lift
  accumulated; one closed sweep per station is the robust pattern.
- Reduced path lengths, relaxed thresholds, MPC, model retraining, and RL tuning were not
  used.

## Remaining blocker and next action

The arm controller is no longer the blocker. The next focused task is to make the truck
phase transition deterministic at a settled 0.50 m endpoint without root-state writes.
Instrument wheel velocity targets, actual wheel velocities, brake joint effort, chassis
velocity, and tire/contact forces from the last driven step through 2 s of braking. Fix
that transition before enabling a repeated production cycle or PPO.

## Commands

See the repository [README](../README.md) for headless and viewer commands. Viewer paths
use standard IsaacLab `AppLauncher` behavior and `VisualizationMarkers`: green reference,
red goal, blue cutter.
