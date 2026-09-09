# Result — common robustness-boundary sweep (actuator authority kappa)

Run 2026-09-02. 48 evaluations, mesh 1337, 176 envs, 900 steps, seed 17.
Design: `experiments/E5_plant_robustness_envelope/README.md`. Thresholds pre-declared there before any curve existed.

`Delta_x_applied = kappa * Delta_x_actuator`, applied at the same boundary on every
stack (after the actuator model, before the PhysX target).

**Every stack scored at 4.0 km/h**, the trained speed. Over the 2.0-7.0 range R and IF
sit at 910 and 1014 mm lateral — broken before kappa is applied — while at 4.0 they are
at 41 and 24 mm. See `outputs/kappa_speed2to7_superseded/`.

## Median lateral error (mm)

| kappa | ROM | ID->FD | AF frozen | AF adaptive |
|---|---|---|---|---|
| 1.00 | 40.69 | 23.66 | 45.18 | 27.03 |
| 0.80 | 43.38 | 25.05 | 44.57 | 27.76 |
| 0.60 | 46.50 | 27.43 | 46.08 | 30.13 |
| 0.50 | 50.88 | 27.57 | 46.03 | 34.29 |
| 0.40 | 51.41 | 28.93 | 48.14 | 39.62 |
| 0.30 | 55.55 | 30.52 | 51.38 | 42.60 |
| 0.20 | 58.28 | 37.39 | 56.59 | 52.25 |
| 0.10 | 72.12 | 44.83 | 72.01 | 71.94 |
| 0.05 | 89.40 | 61.38 | 85.79 | 81.62 |
| 0.03 | 95.31 | 73.69 | 93.57 | 84.73 |
| 0.02 | 97.08 | 78.52 | 97.58 | 89.32 |
| 0.01 | 100.49 | 89.31 | 94.13 | 95.90 |

## Hard breakdown — NOT REACHED

Pre-declared gate: ground contact < 0.700 or safety termination > 0.080.

**No stack breaks down at any kappa tested, down to 0.01.** Contact never falls below
0.70 and safety never exceeds 0.080. A `kappa_crit` on this gate is therefore undefined
for all four stacks: the machine tolerates a 100x authority cut without tripping it.

This is the vacuous-gate failure E5's own Traps section warns about, now measured. The
gate is not wrong, it is simply not binding on this axis at this operating point.

## Degradation onset (lateral >= 1.5x own nominal)

| stack | nominal mm | bar mm | onset kappa |
|---|---|---|---|
| ROM | 40.69 | 61.04 | 0.10 |
| ID->FD | 23.66 | 35.49 | 0.20 |
| AF frozen | 45.18 | 67.78 | 0.10 |
| AF adaptive | 27.03 | 40.55 | 0.30 |

Read literally this favours the FROZEN stack (0.10 vs 0.30). The metric normalises each
stack to its own nominal, so the adaptive arm's better nominal (27.0 vs 45.2 mm) sets it a
stricter absolute bar — 40.6 mm against 67.8 mm. It measures relative graceful degradation,
not envelope, and it penalises the better controller for being better. Reported as measured.

## Iso-performance — the usable result

AF adaptive matches AF frozen's FULL-authority tracking (45.18 mm) down to
**kappa = 0.30**, a **3.3x actuator-authority saving**. Adaptive leads frozen at
every rung from kappa 1.0 to 0.2, converges at 0.1 and is marginally behind at 0.01.

**Provenance:** E5 pre-declared hard breakdown and degradation onset, NOT iso-performance.
This crossing was proposed from the partial curve before the full table existed, so it is
not fitted to the answer — but it is an added analysis and is labelled as such, not as
pre-registered.

## Cross-stack finding

ID->FD is the strongest stack overall: best at nominal (23.66 mm) and clearly best under
degradation (44.83 mm at kappa=0.1 against ~72 mm for the other three). AF adaptive does
not beat ID->FD at any rung. The adaptation claim here is a within-stack frozen-vs-adaptive
comparison and must be captioned as such; the figure does not support 'the adaptive stack
is the most robust controller'.

## What this supports, and what it does not

SUPPORTED: online adaptation buys a 3.3x actuator-authority margin over its own frozen
baseline, and dominates it across the useful range of kappa.

NOT SUPPORTED: 'online adaptation extends the closed-loop robustness envelope' in the
breakdown-boundary sense. No boundary was reached, so no envelope was extended. Stating
it that way would not survive anyone re-running this sweep.
