# G1 headroom exists and is reachable -- through payload mass

**The first clean G1 pass in this project.** Measured with the repaired
instrument; supersedes the withdrawn "+24% to +59%" figure, which came from the
biased sequential evaluator.

## Method

One **fresh process per level**, all with identical seed (17), mesh (1337) and
env count (176), so env *k* is the same terrain and the same manifest factor
draw in every run -- common random numbers. Paired **env-for-env across runs**,
reported as bootstrap CIs on the **paired median log-ratio** (20k resamples).

Separate processes are what makes this trustworthy: no sequential state can leak
between levels, which is the defect that invalidated `--paired_mass_scale`
(A/A control: 1.1-2.4 mm between arms identical by construction).

`MULAG_PAYLOAD_RANGE=X,X` pins every env to scale X and replaces the
`mass_scale` column only, so the privileged context still reads it from
`DIM_ORDER[4]`. All other manifest columns are drawn from the same seed in every
run, so they are identical per env across levels and cancel in the pairing.

Baseline (payload 1.0x): lateral 29.429 mm, cut height 9.228 mm, n=176.

## Dose-response

| payload | lateral median | vs baseline | 95% CI | cut median | vs baseline | 95% CI |
|---|---|---|---|---|---|---|
| 1.5x | 33.02 mm | +10.1% | [+8.4, +12.8] | 17.41 mm | **+96.8%** | [+78.0, +120.9] |
| 2.0x | 43.06 mm | +44.2% | [+35.4, +51.8] | 47.10 mm | **+451.0%** | [+377.9, +558.8] |
| 3.0x | 141.69 mm | +371.0% | [+323.0, +421.3] | 496.90 mm | +5327.7% | [+4369.7, +5957.4] |

Monotone, every CI far from zero.

## Reading

- **Cut height is the degradation axis, not lateral.** At 1.5x cut error nearly
  doubles while lateral moves 10%. Reading lateral alone understates the effect
  by ~10x and is how this headroom was missed.
- **1.5x-2.0x is the adaptation window.** Large headroom with the policy still
  functioning.
- **3.0x is past breakdown**, not a harder operating point: 497 mm cut error is a
  destroyed task, with nothing for an estimator to recover.
- **This corrects E5's conclusion** that a breakdown point "cannot be found by
  scaling factors". That conclusion is correct *for the hydraulic and authority
  axes* -- flow/compliance/seal/pressure are inert on FD, and authority is
  clamped at 62% removable. It does not hold for payload mass, which is
  mechanical, acts through PhysX, and reaches breakdown comfortably.

## Why this matters for online adaptation

G1 states that adaptation can only recover headroom a degradation creates. Until
now no perturbation in this project had created any, so every ablation was
measured at the noise floor -- and the nominal-plant gate confirmed it
(privileged vs mean/shuffled/zero, all CIs including zero,
`outputs/rif_delay/p44_gate.json`).

There is now headroom, and `mass_scale` is already a column of the privileged
context, so the existing 72-wide route observes it with no width change.

**Caveat carried forward:** training directly into a wide payload range collapses
the policy into a reward local optimum -- lifting the cutter off the ground
(ground contact 0.879 -> 0.242, safety violations 0.014 -> 0.643 on a 1-3x run).
The range must be ramped, with ground contact watched as the canary.
