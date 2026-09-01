# E3 — FINAL result: method R1 at n=880

Pre-registration `reports/PREREG_rebalanced_2026-08-31.md`, amendments 1 and 2.
Completed 2026-09-01 08:41 UTC. This is the declared endpoint.

## Verdict: the pre-registered bar is NOT met

**2 of 5 seeds pass both metrics. The median seed fails.**

880 paired environments, five held-out meshes (1337, 17, 4242, 909, 7001),
scored against `base_R1` on the declared flip plant.

| seed | cut | lateral | paired cut (mm) | paired lateral (mm) | verdict |
|---|---|---|---|---|---|
| 17  | +62.6%* | +78.6%* | +2.493 [+1.914, +3.055] | +23.057 [+21.786, +24.375] | **PASS** |
| 23  | +40.9%* | +21.3%* | +1.503 [+1.161, +1.969] | +9.570 [+7.864, +10.914] | **PASS** |
| 101 | −1.1% | +28.6%* | −0.044 [−0.387, +0.436] | +11.669 [+10.530, +13.310] | trade-off |
| 42  | −23.3%* | −18.2%* | −1.617 [−2.072, −0.976] | −13.275 [−15.762, −10.064] | both worse |
| 7   | −84.8%* | −72.1%* | −43.057 [−57.557, −34.757] | −141.451 [−145.249, −137.207] | both worse |

`*` = bootstrap CI on the paired difference excludes zero.

## The power increase refuted seed 101 rather than rescuing it

Amendment 2 raised n from 528 to 880 for every arm, because seed 101 had missed
the bar by **0.019 mm** on its cut lower bound. The pre-declared rule was that
whatever the increase returned would be the answer.

| | cut effect | interval |
|---|---|---|
| n=528 | +12.5% | +0.601 mm [−0.019, +1.119] |
| n=880 | **−1.1%** | −0.044 mm [−0.387, +0.436] |

With 67% more data the effect went to zero and slightly negative. The n=528
result was noise — precisely as the measured ±10% cut-height false-positive floor
predicted, since +12.5% sat inside that band. **This is the power test working
as designed: it discriminated, and it discriminated against the hypothesis.**

Per amendment 2 there is no third increase. R1 is closed as a negative.

## What is robust, and is worth reporting

The two passing seeds barely moved when n rose 67% — seed 17 +63.7% → +62.6%
cut, seed 23 +39.6% → +40.9%. Stable estimates under a large increase in n is
strong evidence these are real effects, not noise.

### Condition 2 is satisfied for the seeds that pass

The capacity-matched control (`--residual_no_history`: same frozen base, same
architecture, same 53,836 trainable parameters, same seed, budget and reward —
only the history window zeroed), n=880:

| comparison | cut | lateral |
|---|---|---|
| control → seed 17 | +51.3%* (+1.902 [+1.661, +2.282]) | +81.6%* (+23.988 [+23.066, +25.336]) |
| control → seed 23 | +51.4%* (+1.804 [+1.588, +2.030]) | +26.1%* (+11.781 [+10.439, +12.869]) |

**Where the method works, the gain is the live history, not the extra capacity.**
Both metrics, both seeds, all four CIs excluding zero.

## Verdict against the four conditions

| condition | status |
|---|---|
| 1. median of five improves both, CIs exclude zero | **NOT MET** — 2 of 5 |
| 2. capacity-matched control does not account for the gain | **MET, for seeds 17 and 23** |
| 3. sham does not account for the gain | not run — moot at the median; would be needed for a conditional claim |
| 4. holds on the protocol as declared, not a subset | **MET** — all five seeds reported |

## The defensible claim

Not "online adaptation improves both metrics." The evidence supports:

> A history-conditioned residual on a frozen base produces large, replicated
> improvements in **both** cut-height and lateral tracking on 2 of 5 seeds
> (+62.6%/+78.6% and +40.9%/+21.3% at n=880), and where it works the gain is
> attributable to the live history rather than to added capacity. It fails on
> 2 of 5 seeds and is null on 1. The seed variance is intrinsic: it is not an
> optimisation-stability artefact, since lowering the learning rate (R2) merely
> reassigned which seeds fail — rescuing seed 7 from −86.9% to +14.0% while
> destroying seed 17 from +63.7% to −15.0%.

## Methods that were tried and failed

| version | change | outcome |
|---|---|---|
| R1 | rebalanced lateral kernel (std 0.09, weight 4.0), lr 3e-4 | 2 of 5, median fails |
| R2 | lr 1e-4 | 1 of 5 on screen, abandoned; lr trades seeds rather than stabilising |

## Cost

~11 h wall: 10 R1+R2 training runs, 47 evaluations, one A/A instrument test.
