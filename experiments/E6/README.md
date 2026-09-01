# E6 — Latent bandwidth sweep

**Defends:** *Online Adaptation* · **Covers:** PPO-AAM-FD, one checkpoint
**Status:** complete 2026-08-19 · **verdict: mechanism NOT supported** · not
cited in the thesis. Kept as the record of a question asked and answered.

## The question

E4 scored four latents on the deployed stack and the deployed estimate came
**last** — behind a constant carrying no plant information and behind a constant
that is actively wrong about the plant. Chapter 7 reads this as the estimate's
*time-variation* disturbing the feedback loop.

That reading is an inference, not a measurement. E4's arms differ in **two**
respects at once:

| arm | plant content | movement |
|---|---|---|
| estimated | correct | high |
| mean | none | none |
| shuffled | wrong | none |

Content and bandwidth are perfectly confounded, so either could explain the
ranking. An examiner asked exactly this: *what experiment separates harmful
temporal latent variability from simply incorrect latent information?*

## The design

Hold content fixed, sweep bandwidth. Every arm is **the same checkpoint reading
the same estimate**; a one-pole low-pass on the **policy path only** limits how
fast that estimate may move. The ID adapter keeps the full-rate estimate
throughout — which is also the split Chapter 8 proposes as the fix: band-limit
the latent where a feedback loop consumes it, leave it responsive where a
one-step model consumes it.

| arm | policy latent | bandwidth | content |
|---|---|---|---|
| `bw_estimated` | estimated | unlimited | correct |
| `bw_lpf_5p0` | lpf 5.00 Hz | high | correct |
| `bw_lpf_1p0` | lpf 1.00 Hz | | correct |
| `bw_lpf_0p2` | lpf 0.20 Hz | | correct |
| `bw_lpf_0p04` | lpf 0.04 Hz | τ = 4 s | correct |
| `bw_frozen` | frozen @ tick 40 | zero | correct |
| `bw_mean` | mean | zero | **none** |

`bw_frozen` and `bw_mean` are the discriminator. Same bandwidth, different
content — so a gap between them is content, and a gap along the lpf series is
bandwidth.

## How to read the outcome

| pattern | conclusion |
|---|---|
| monotone trend along the lpf series | bandwidth is doing the work — §7.9's mechanism is supported |
| `frozen` ≈ `mean`, both beating `estimated` | bandwidth, not content |
| `frozen` clearly beats `mean` | content matters after all; §7.9 needs rewriting |
| flat everywhere | **neither** — the E4 ranking was noise |

A flat result is a real outcome and must be reported as one. It would say the
"responsiveness" mechanism is unsupported and that Chapter 8's design
recommendation has to be **withdrawn**, not softened. The experiment is worth
running because it can fail in a way that costs the thesis a claim.


## Result (2026-08-19, 33 paired cells, one checkpoint)

**The proposed mechanism is not supported.** Reported against the pre-registered
reading table above, the outcome is the "flat everywhere" row.

Mean lateral error, bandwidth high to low:

| arm | mean (m) | vs estimated | cells better | median vs estimated |
|---|---|---|---|---|
| `bw_estimated` | 0.09002 | — | — | 0.0% |
| `bw_lpf_5p0` | 0.07310 | -18.8% | 14/33 | +1.7% |
| `bw_lpf_1p0` | 0.07351 | -18.3% | 18/33 | +5.7% |
| `bw_lpf_0p2` | 0.06892 | **-23.4%, resolved** | 22/33 | -0.6% |
| `bw_lpf_0p04` | 0.07090 | -21.2% | 16/33 | +6.4% |
| `bw_frozen` | 0.08013 | -11.0% | 13/33 | -0.8% |
| `bw_mean` | 0.07240 | -19.6% | 15/33 | -3.7% |

Four reasons the bandwidth mechanism fails on this evidence:

1. **The means are tail-driven, not typical.** Every arm improves the mean by
   11-25%, but the medians are flat to slightly worse (+1.7% to +6.4% for the
   lpf arms on lateral error). Most arms improve fewer than half the cells while
   improving the mean by a fifth. Band-limiting removes a few very bad cells; it
   does not improve typical behaviour.
2. **There is no monotone bandwidth trend.** Spearman rank correlation against
   bandwidth order is **-0.314** on lateral error and **-0.371** on cut height,
   the wrong sign for "task improves as bandwidth falls".
3. **The zero-bandwidth arm that preserves content is the worst alternative.**
   `bw_frozen` has the smallest mean improvement of the six. A pure bandwidth
   story predicts it should be the best.
4. **Only 1 of 18 contrasts resolves**, at 0.2 Hz, with non-resolving neighbours
   on both sides. At 95% over 18 contrasts roughly one false positive is
   expected, so this is not treated as a real optimum.

`frozen` vs `mean` (content at zero bandwidth) is unresolved on all three
metrics, so content is not established as the mechanism either.

**What this costs the thesis.** As stated above before the run, a flat result
means §7.9's "responsiveness rather than accuracy" mechanism is unsupported and
Chapter 8's design recommendation to filter the latent on the policy path must be
**withdrawn**, not softened. What survives is the weaker and still useful
statement: the deployed full-rate estimate is the worst configuration tested, the
gap lives in the tail rather than the median, and no perturbation of the latent
recovers a resolved improvement in typical behaviour.

**What would settle it.** The tail is the whole effect, so the next experiment
should characterise the bad cells directly rather than sweep bandwidth again:
identify which (mesh, family) cells carry the mean gap under `bw_estimated` and
test whether they share a plant draw, a trajectory family or a latent excursion.

## Traps

- `FACTOR_DOMAIN=combined_test` is mandatory. Under `nominal` the plant does not
  vary, the latent is near-constant, and every arm collapses to the same thing.
- The **controller** arm stays `estimated` in every run. Degrading it too would
  reintroduce the confound this experiment exists to remove. The summariser
  refuses a grid where any manifest says otherwise.
- `FREEZE_TICK` is in **50 ms model ticks**, not 60 Hz control steps. 40 = 2.0 s,
  after the AAM's 20-row (1.0 s) warm-up, so the latch holds a real estimate.
- **Validate the instrument.** If the low-pass never reached the config, every
  lpf arm scores bit-identically to `bw_estimated` and the sweep reports a flat
  line indistinguishable from a true null. `summarise_e6.py` raises on that
  rather than reporting it. Verified live before this grid was queued: at
  0.04 Hz on mesh 17, 13 of 16 family metrics moved against `estimated`.
- No privileged arm here. It needs its own adapter and would change two things
  at once. E4 already reports it.

## Run

```bash
bash experiments/E6/code/run_e6_latent_bandwidth.sh
python experiments/E6/code/summarise_e6.py \
  --root outputs/e6_latent_bandwidth \
  --out experiments/E6/results/e6_summary.json
```

~7 min per arm × 7 arms ≈ 50 min serial on one GPU.
