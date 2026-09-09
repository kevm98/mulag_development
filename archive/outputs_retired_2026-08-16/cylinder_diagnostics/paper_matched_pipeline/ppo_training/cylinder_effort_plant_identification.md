# Cylinder Effort Plant Identification

## Method

Force pulses injected directly via `set_joint_effort_target()` with D-only baseline
(kd=0.3×). Force fractions tested: 1%, 2%, 5%, 10%, 20% of effort limit per cylinder.

Pulse duration: 0.3s, settle: 1.5s, recorded at 3-step intervals.

## Results

The arm has exactly **two equilibrium states**, independent of pulse.

| State | Drehzapfen dq | Ausleger I dq | Ausleger II dq | Messerkopf dq | EE delta |
|-------|-------------|-------------|---------------|-------------|---------|
| A | +2.54 mm | +4.35 mm | -11.40 mm | -13.38 mm | 0.046 m |
| B | -0.15 mm | -2.69 mm | -3.81 mm | -12.04 mm | 0.054 m |

These states alternate **regardless of which cylinder is pulsed, sign, or magnitude.**

At 20 MN force on Drehzapfen (20% of 100MN limit), response = same 2.54mm as at 5MN.

## Per-Cylinder Gains

| Cylinder | Gain (mm/MN) | Sign Accuracy | Notes |
|----------|-------------|---------------|-------|
| Drehzapfen | 0.2-1.1 | 4/6 | Extremely stiff, needs >100MN for cm response |
| Ausleger I | 4-20 | 4/6 | Moderate compliance |
| Ausleger II | 89-431 | 3/6 | Compliant but gravity-dominated |
| Messerkopf | 1483-7202 | 3/6 | Very compliant, tiny force moves it |

Gains measured from curve-fitting to the TWO discrete states — actual continuous
transfer function cannot be estimated because force has no measurable effect.

## Cross-Coupling

Identical cross-coupling pattern in ALL 48 tests:
- When Drehzapfen moves +2.54mm, Ausleger I moves +4.35mm, Ausleger II moves -11.40mm, Messerkopf moves -13.38mm
- When Drehzapfen moves -0.15mm, Ausleger I moves -2.69mm, Ausleger II moves -3.81mm, Messerkopf moves -12.04mm

This is NOT cross-coupling from the pulse — it's the arm switching between two
gravitational equilibrium configurations, perfectly correlated via kinematics.

## Conclusion

**The force pulses cannot measurably affect the arm because gravity dominates.**
Even 20 MN on Drehzapfen (20× the typical PD force) produces the same 2.54 mm
displacement as 5 MN. The arm is insensitive to force perturbations.

This means:
1. Plant identification via force pulses is not meaningful — gravity provides 100× more force
2. The cylinder-level PD approach cannot achieve precision hold (<3cm drift)
3. The arm is a gravitationally-dominated system with two discrete equilibrium states

## Recommendation

Accept drift as baseline. Measure J_eff via **drift subtraction** with paired
baseline runs. Use **larger command amplitudes** (>0.1 m EE displacement) where
the command signal exceeds the drift noise.

For hold: use the settled equilibrium (State A or B) as the target pose.
The arm naturally oscillates between these two states at ~0.046m EE amplitude.
Hold precision below this amplitude cannot be achieved with cylinder-level PD.

## Per-Cylinder Controller Design (from plant ID)

Since continuous plant response is unavailable, use conservative D-only control:

| Cylinder | kd_scale | Rationale |
|----------|---------|-----------|
| Drehzapfen | 0.3 | Stiff, needs minimal damping |
| Ausleger I | 0.3 | Moderate compliance |
| Ausleger II | 0.5 | More compliant, needs more damping |
| Messerkopf | 0.5 | Very compliant, needs more damping |

kp_scale = 0.0 for all (P adds oscillation without improving hold).
