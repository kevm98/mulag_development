# Final result: online payload adaptation, three seeds

Run 2026-09-06 22:56 to 2026-09-07 01:21 CEST on the L4. Held-out mesh 3141,
176 environments, 900 steps, one arm per process, paired per environment index.
Pre-registered in `PREREG_FINAL_SEEDS_2026-09-06.md`. Architecture locked to
context columns (`--train_context_only`) before any of this was scored.

## What was run

24 scored evaluations. Frozen is scored once per mass and shared across seeds:
with the context zeroed the network computes `base(x) + W_ctx . 0 = base(x)`,
so any of the three checkpoints IS the frozen policy, bit for bit. The base
task columns are identical across all three seeds at 0.125463, which is the
evidence that context-only training left the base untouched.

All three seeds put their entire context weight on `mass_scale` and exactly
zero on alpha, tau and delay. Nothing told them which channel to use.

| seed | base cols | alpha | tau | delay | mass_scale |
|---|---|---|---|---|---|
| 17 | 0.125463 | 0 | 0 | 0 | 0.405202 |
| 23 | 0.125463 | 0 | 0 | 0 | 0.385290 |
| 101 | 0.125463 | 0 | 0 | 0 | 0.660184 |

These are three independent **context-training** seeds on one shared frozen
base, not three independent trainings of the whole controller. That is the
right scope for a claim about the adaptation mechanism and it must be written
that way.

## Statistic

Paired per-environment **median difference** with a percentile bootstrap,
B = 20000, plus the fraction of environments improved.

The median log-ratio used in earlier tables is not reported as primary here. It
disagrees in sign with the marginal medians in several cells (seed 23, 5.0x cut:
medians 267.44 -> 111.14 mm but ratio -27.3 %). Neither number is wrong: with an
even sample and a two-lobed paired distribution, `median(x-y)` and `median(x/y)`
can fall on opposite sides of their null values. For an effect this
heterogeneous the difference and the win fraction are the honest summaries.

## The frozen policy has a large, monotone deficit

| payload | cut mm | lateral mm | contact | safety viol. |
|---|---|---|---|---|
| 3.0x | 10.37 | 84.37 | 0.883 | 0.096 |
| 4.0x | 167.52 | 318.63 | 0.636 | 0.281 |
| 5.0x | 267.44 | 400.89 | 0.601 | 0.303 |

The collapse between 3.0x and 4.0x is the robustness deficit H1 requires. At
3.0x there is no deficit to repair.

## Adaptive vs Frozen, paired

Positive = adaptive better. `*` = 95 % CI excludes zero.

### Cut height (mm)

| mass | seed | frozen | adaptive | median diff | 95 % CI | envs better |
|---|---|---|---|---|---|---|
| 3.0 | 17 | 10.37 | 22.10 | -4.47 | [-8.68, -0.82]* | 40.9 % |
| 3.0 | 23 | 10.37 | 24.62 | -6.23 | [-11.81, -3.24]* | 36.4 % |
| 3.0 | 101 | 10.37 | 8.87 | +2.84 | [+0.62, +6.25]* | 60.2 % |
| 3.0 | POOL | | | -2.03 | [-3.51, +0.04] | 45.8 % |
| 4.0 | 17 | 167.52 | 36.53 | +122.54 | [+71.52, +215.27]* | 71.6 % |
| 4.0 | 23 | 167.52 | 47.97 | -2.47 | [-14.17, +23.99] | 48.9 % |
| 4.0 | 101 | 167.52 | 19.01 | +124.89 | [+53.84, +217.25]* | 76.7 % |
| 4.0 | POOL | | | **+71.52** | **[+45.34, +110.88]*** | 65.7 % |
| 5.0 | 17 | 267.44 | 48.78 | +187.61 | [+124.64, +231.41]* | 71.6 % |
| 5.0 | 23 | 267.44 | 111.14 | -46.68 | [-86.02, -8.23]* | 39.2 % |
| 5.0 | 101 | 267.44 | 40.46 | +110.11 | [+65.76, +183.59]* | 71.6 % |
| 5.0 | POOL | | | **+82.18** | **[+39.00, +115.03]*** | 60.8 % |

### Lateral (mm)

| mass | seed | frozen | adaptive | median diff | 95 % CI | envs better |
|---|---|---|---|---|---|---|
| 3.0 | 17 | 84.37 | 106.01 | -25.22 | [-35.50, -17.86]* | 34.7 % |
| 3.0 | 23 | 84.37 | 169.04 | -49.96 | [-85.15, -28.70]* | 33.5 % |
| 3.0 | 101 | 84.37 | 90.59 | -9.42 | [-18.74, -1.35]* | 42.0 % |
| 3.0 | POOL | | | **-22.75** | **[-30.04, -17.27]*** | 36.7 % |
| 4.0 | 17 | 318.63 | 113.18 | +140.27 | [+80.34, +218.15]* | 77.8 % |
| 4.0 | 23 | 318.63 | 368.47 | -34.89 | [-91.96, +51.07] | 47.7 % |
| 4.0 | 101 | 318.63 | 150.40 | +73.10 | [+35.60, +139.99]* | 67.0 % |
| 4.0 | POOL | | | **+71.58** | **[+46.12, +127.55]*** | 64.2 % |
| 5.0 | 17 | 400.89 | 154.89 | +185.14 | [+121.96, +229.67]* | 70.5 % |
| 5.0 | 23 | 400.89 | 350.63 | +4.35 | [-24.78, +100.48] | 50.6 % |
| 5.0 | 101 | 400.89 | 196.77 | +77.19 | [+12.93, +152.38]* | 60.8 % |
| 5.0 | POOL | | | **+107.05** | **[+51.11, +141.68]*** | 60.6 % |

### Ground contact (fraction, higher better)

| mass | POOL median diff | 95 % CI |
|---|---|---|
| 3.0 | -0.00 | [-0.01, +0.01] |
| 4.0 | **+0.12** | **[+0.09, +0.17]*** |
| 5.0 | **+0.09** | **[+0.05, +0.13]*** |

Per seed at 4.0x: 17 `+0.21*`, 23 `+0.02` ns, 101 `+0.17*`.

## Verdict on H1

`H1: E_adaptive < E_frozen` where the frozen policy has a measurable
robustness deficit.

**Supported at 4.0x and 5.0x.** Pooled and paired, cut height, lateral error and
ground contact all improve with the 95 % CI excluding zero, at both masses. The
effect is large: the pooled cut median improves by 71.5 mm at 4.0x and 82.2 mm
at 5.0x against a frozen baseline of 167.5 and 267.4 mm.

**Not supported at 3.0x, and lateral is significantly worse there** in all three
seeds (pooled -22.75 mm, CI [-30.04, -17.27]). This is the predicted behaviour:
the frozen policy holds contact at 0.883 at 3.0x, so there is no deficit to
repair and the conditioning only costs. The mechanism should be gated to the
loaded regime rather than run everywhere.

### Seed 23 is a real exception and stays in the table

Two of three seeds show large significant gains on every channel at 4.0x and
5.0x. Seed 23 does not: neutral on cut at 4.0x, **significantly worse on cut at
5.0x** (-46.68 mm, CI [-86.02, -8.23]), neutral on lateral and contact. The
pre-registration allows "two strong and one neutral" to support H1 on a
convincing pooled result; seed 23 at 5.0x is worse than neutral. The correct
report is that the pooled effect is large and significant and that it is carried
by two of three seeds, with one seed not benefiting. It is not 3/3 and must not
be written as such.

Both oracles sit near 750 mm lateral, so all seeds have the same trade-off curve
available. What differs is where the online estimate lands on it: seed 17 at
113 mm, seed 101 at 150 mm, seed 23 at 368 mm. The difference is in the
estimate, not the policy.

## The oracle is not a ceiling

Seed 101's oracle at 5.0x reads **652.29 mm cut error with contact 0.220**, far
worse than the frozen policy it is supposed to bound. Its adaptive arm on the
same checkpoint reads 40.46 mm at contact 0.803.

The context enters as `W_ctx . c` with `c = (m-1)/0.5`, so `c = 8` at 5.0x. Seed
101 carries the largest weight, 0.660 against 0.405 and 0.385, and it is seed 101
that blows up at the largest `c`. The weight norm does not order the trade-off at
3.0x (seed 23 has the smallest norm and the middle lateral cost) but it does
predict over-drive at the extreme.

This is why the deployable arm can beat the privileged one. The estimate arrives
gradually through the EMA and is held at nominal by the deadband, so it never
reaches the over-driving region the true value walks straight into. Damping is
doing real work, which corroborates the beta-ladder finding that gradual arrival
is the mechanism.

## Wrong-information control

Mixed payload 1.0-5.0x resampled per episode, seed 17 checkpoint. Shuffling the
true payload across environments preserves its distribution and magnitude and
destroys only the pairing. At fixed payload a "wrong" context would be the zero
vector, which IS the frozen policy, so the shuffle on a mixed plant is the
version of this control with content.

| comparison | cut mm | median diff | 95 % CI | better in |
|---|---|---|---|---|
| frozen -> true | 17.75 -> 1.63 | +12.66 | [+9.62, +21.54]* | 94.9 % |
| frozen -> shuffled | 17.75 -> 6.43 | +11.10 | [+7.24, +19.81]* | 84.7 % |
| **shuffled -> true** | 6.43 -> 1.63 | **+3.55** | **[+3.01, +4.14]*** | **83.0 %** |

Contact: `shuffled -> true +0.03, CI [+0.02, +0.03]*`, better in 76.7 %.
Lateral: `shuffled -> true -65.33, CI [-141.83, +83.20]`, not significant.

**Correct payload information beats wrong payload information**, significantly,
on cut and on contact. The claim survives the control.

But the control also bounds it honestly: most of the raw gain over frozen is
available from a wrong payload of the right magnitude (11.10 of 12.66 mm on the
cut median). Roughly a quarter of the cut benefit and a quarter of the contact
benefit is specific to identifying the payload correctly; the rest is generic
conditioning on a plausible load. The mechanism is part learned robustness
offset and part online identification, and it should be described as both.

## Limitations, stated not buried

- Ridge `lambda` was not selected on held-out data for these three fits.
  `outputs/estimator_final/` holds no fixed-payload dumps, so the selection loop
  found no test sets and fell to its first rung, `lambda = 1`. It is identical
  across all three seeds so it cannot confound the seed comparison, and
  estimator quality is bounded by the oracle arm in the same table.
- Seed 17's estimator here is a degree-2 fit and is **not** the degree-1 fit that
  produced the earlier 4/4 held-out result. This table supersedes those numbers
  rather than extending them. They must not be pooled.
- `hard_impact_step_fraction` rises at 4.0x, 0.406 frozen to 0.601 adaptive. It
  counts steps above a force threshold over all active steps and is confounded by
  the contact gain. Conditioned on contact the rate goes 0.638 -> 0.702, a 10 %
  relative rise rather than 48 %, and at 5.0x the oracle is better than frozen
  (0.703 vs 0.745). Report the conditional rate.
- Three seeds is three points. No standard deviation is quoted.
- One mesh, 3141. Held out, never used to design or diagnose this campaign.
