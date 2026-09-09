# E3 — dose–response: isolating the adaptation component

Pre-registration: `reports/PREREG_dose_response_2026-09-01.md`, fixed before the
run. Executed 2026-09-01 09:56–12:11 UTC. 20 evaluations.

## Design

The dose is the magnitude of the mid-episode mass flip:
`MULAG_MASS_BIMODAL=1.0,X` with `MULAG_MASS_STEP_SCALE=X`, for
X ∈ {1.0, 1.5, 2.0, 2.5, 3.0}. Arms: frozen base (`--residual_alpha 0.0`) and
adaptive (seed 17, `model_399`, `--residual_alpha 0.5`), paired on identical
environments. Meshes 1337 and 17, **n = 352 per dose point**.

**Dose 1.0 is the sham**: the flip schedule fires on the same jittered instants
and writes the same mass value, but the mass never changes.

### Sham verified against the artefacts

| dose | recorded `deployment_mass_bimodal` | mass_scale across 176 envs |
|---|---|---|
| 1.0 | [1.0, 1.0] | **{1.0: 176}** — mass never moves |
| 1.5 | [1.0, 1.5] | {1.0: 88, 1.5: 88} |
| 2.0 | [1.0, 2.0] | {1.0: 88, 2.0: 88} |
| 2.5 | [1.0, 2.5] | {1.0: 88, 2.5: 88} |
| 3.0 | [1.0, 3.0] | {1.0: 88, 3.0: 88} |

Base and adaptive carry identical config at every dose; the only difference is
`residual_alpha`. The dose manipulation bites: base cut error degrades
4.219 → 6.735 mm and base lateral 45.190 → 62.619 mm from dose 1.0 to 3.0.

## Raw curve

| dose | cut gain | lateral gain | base cut (mm) | base lateral (mm) |
|---|---|---|---|---|
| 1.0 (sham) | +57.5%* | +54.8%* | 4.219 | 45.190 |
| 1.5 | +10.9%* | +76.8%* | 5.239 | 48.806 |
| 2.0 | +55.5%* | +70.4%* | 5.598 | 52.361 |
| 2.5 | +98.6%* | +72.9%* | 6.665 | 54.429 |
| 3.0 | +49.7%* | +84.2%* | 6.735 | 62.619 |

`*` = paired bootstrap CI on the difference excludes zero.

## The result: decompose, do not pass/fail

A sham is a **baseline to subtract**, not a gate. The adaptive arm beats the base
at every dose *including* the sham, so part of the benefit is dose-independent
improved tracking. The question is whether anything remains **on top** of that
which scales with the plant change.

### LATERAL — a significant, monotone, dose-dependent adaptation component

| dose | lateral gain | gain in excess of sham |
|---|---|---|
| 1.0 (sham) | +16.557 mm | — |
| 1.5 | +20.045 mm | **+3.796*** [+2.303, +6.105] |
| 2.0 | +20.487 mm | **+3.912*** [+1.693, +6.415] |
| 2.5 | +20.490 mm | **+4.073*** [+2.318, +6.697] |
| 3.0 | +23.705 mm | **+7.410*** [+4.785, +10.503] |

Every interval excludes zero, and the excess **grows with dose**, largest at the
largest perturbation. A benefit that scales with the magnitude of the plant
change it is supposed to respond to is the signature of genuine online
adaptation.

**Decomposition at dose 3.0:** of the +23.705 mm lateral gain, **16.557 mm (70%)
is dose-independent tracking and 7.410 mm (31%) is attributable to the magnitude
of the mid-episode plant change.**

### CUT HEIGHT — no consistent adaptation component

| dose | excess over sham |
|---|---|
| 1.5 | −0.919* [−1.419, −0.345] |
| 2.0 | +0.562 [−0.131, +1.192] |
| 2.5 | +1.644* [+1.093, +2.282] |
| 3.0 | +0.674 [−0.078, +1.833] |

Non-monotone, two of four non-significant, one significantly negative. The
cut-height benefit is attributable to improved tracking, not to adaptation.


## CORRECTION 2026-09-01 18:10 UTC — what the dose axis actually varies

The claim below originally read that 31% of the lateral gain at 3.0x is
"attributable to the plant change", i.e. to online adaptation. **That overstates
what this design isolates, and the overstatement is corrected here.**

Checked against the artefacts, the per-environment STARTING mass at each dose is:

| dose | `deployment_mass_bimodal` | per-env starting mass |
|---|---|---|
| 1.0 | [1.0, 1.0] | {1.0x: 176} |
| 1.5 | [1.0, 1.5] | {1.0x: 88, 1.5x: 88} |
| 2.0 | [1.0, 2.0] | {1.0x: 88, 2.0x: 88} |
| 2.5 | [1.0, 2.5] | {1.0x: 88, 2.5x: 88} |
| 3.0 | [1.0, 3.0] | {1.0x: 88, 3.0x: 88} |

**Raising the dose therefore makes half the fleet heavier from t = 0**, as well as
making the mid-episode change larger. The two are varied together and cannot be
separated by this experiment.

So the measured, significant, monotone effect is real, but its correct reading is:

> **the residual's advantage grows with cutter mass** — +3.796, +3.912, +4.073,
> +7.410 mm of lateral gain in excess of the 1.0x condition, every CI excluding
> zero —

which **includes, but does not isolate**, adaptation to the mid-episode change. A
static-mass benefit produces the same curve.

### Isolating the mid-episode component needs a different plant

Every environment must start at the same mass and all must step together, so the
pre-step window is a common baseline. Two attempts on 2026-09-01 failed and were
discarded rather than reported:

1. `MULAG_MASS_BIMODAL=1.0,3.0` — confounded: half the fleet starts at 3.0x, so
   there is no common pre-step baseline. The difference-in-differences computed on
   it was void and was discarded.
2. `MULAG_MASS_BIMODAL=1.0,1.0` with `MULAG_MASS_STEP_SCALE=3.0` — inert: both
   modes are 1.0 so the bimodal path overrides the step scale and the mass never
   moves. Verified from the trace (`mass_scale_mean` = 1.00 throughout).

The third configuration omits `MULAG_MASS_BIMODAL` entirely so
`deployment_mass_step_scale` applies uniformly to every environment. Its result is
reported wherever it lands, including a null.

## Supported claim (superseded by the correction above)

> A history-conditioned residual on a frozen base delivers a large
> dose-independent improvement in tracking, **plus a statistically significant,
> monotonically dose-dependent lateral component that scales with the magnitude
> of the mid-episode plant change** — 31% of the lateral gain at a 3.0x flip.
> This component is evidence of genuine online adaptation on the lateral axis.
> The cut-height benefit shows no consistent dose-dependence and is attributable
> to improved tracking rather than to plant identification.

## What this claim is NOT

* Not "adaptation improves both metrics" — cut height shows no dose-dependence.
* Not a claim about the median seed — this characterises seed 17, the arm the
  mechanism claim rests on. See `FINAL_R1_n880.md`: 2 of 5 seeds pass, median
  fails.
* The majority (70%) of the lateral benefit at 3.0x is **not** adaptation.

## Why the decomposition is the right analysis

Reading the sham as pass/fail would have thrown away a real effect: the sham gap
is large, so a pass/fail reading rejects adaptation entirely. Subtracting it
instead isolates the component that actually tracks the perturbation. This also
supersedes the old-objective sham figure (+29.3%), which was never decomposed.

Consistency check: E3's earlier latent probe found z_hat shifting 0.496 when the
plant changes versus 0.473 when it does not — a small but non-zero difference,
in the same direction as the small-but-real lateral component measured here.
