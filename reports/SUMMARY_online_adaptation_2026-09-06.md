# Online payload adaptation: consolidated result

Everything below was measured 2026-09-05/06. Each row links to the report that
records the run. Pending items are marked and not counted as evidence.

## The mechanism

A frozen PPO policy reads a 70-wide observation: 66 task columns plus a 4-wide
context `(alpha, tau, delay, mass_scale)`. The base's weights on all four
context columns are **exactly 0.000000**, so it is blind to payload by
construction, while degrading ~95x on cut height over payload 1.0-5.0x.

Training touches ONLY those context columns (`--train_context_only`): **1024
parameters against 59,588**, with the task columns verified bit-identical
throughout at mean |w| = 0.125463. At deployment the privileged payload is
replaced by one estimated at runtime from the task observation alone -- cylinder
effort, the actuator prediction residual, cylinder state -- smoothed by an EMA
that starts each episode at nominal and converges while driving.

**The control is provable, not trained.** With the context zeroed the trained
checkpoint computes `base(x) + W_ctx . 0 = base(x)`, so `ctxzero` IS the frozen
policy bit for bit.

## The claim, as supported

> Online payload estimation **recovers ground contact when a frozen controller
> is losing it**, producing large joint improvements at heavy payload
> (held-out, 4/4) and on a large time-varying excursion -- from an estimate that
> never reads ground truth -- **at the cost of lateral tracking when contact is
> already adequate**.

**The governing rule, which predicts every configuration measured:**

| configuration | frozen-policy contact | outcome |
|---|---:|---|
| constant 5.0x | 0.539 | joint pass |
| step 1.0 -> 5.0x | 0.728 | joint pass |
| constant 3.0x | 0.803 | Pareto pass |
| step 1.0 -> 3.0x | 0.841 | fail |
| step 2.0 -> 5.0x | 0.871 | fail |
| constant 1.0x | 0.954 | fail |

Monotone in the frozen policy's ground contact. The arm makes exactly one trade
-- contact for lateral -- and whether it pays depends on how much contact there
is to recover. The nominal-payload limitation is the special case with the most
contact, not a separate defect.

This follows from the structural finding: four trainable context columns can
only slide the policy along a cut/contact-versus-lateral curve, so the trade is
always the same one.

**Withdrawn:** an earlier form of this document claimed "adaptation works on
time-varying plants" on the strength of the 1.0 -> 5.0x step. Two further step
configurations were then run and both fail. One of three passes, and the claim
in that form over-read a single favourable configuration.

### Constant heavy payload, HELD-OUT meshes

Meshes 3141 and 8191 were never used to design, diagnose or read the campaign.

| mesh | payload | cut | lateral | contact | outcome |
|---|---|---:|---:|---:|---|
| 3141 | 3.0x | +510.9 %* | +16.3 %* | +0.077* | joint |
| 8191 | 3.0x | +501.6 %* | +3.6 % | +0.101* | Pareto |
| 3141 | 5.0x | +1996.3 %* | +253.3 %* | +0.348* | joint |
| 8191 | 5.0x | +2152.4 %* | +265.0 %* | +0.442* | joint |

4/4 pass. Held-out effects are LARGER than the selection mesh's, so this is not
selection bias. (`RESULT_online_heldout_2026-09-06.md`)

### Time-varying payload

Payload steps mid-episode at 6 s +- 2 s of jitter, in a 15.0 s episode. A plant
gate is run on the FROZEN policy first: it must land between the two static
plants, and does (57 % and 38 % of the way, with all five metrics bracketed).

| step | cut | lateral | contact | outcome |
|---|---:|---:|---:|---|
| 1.0 -> 3.0x | +175.4 %* | -13.8 %* | +0.058* | fail |
| 1.0 -> 5.0x | **+636.1 %*** | **+52.0 %*** | **+0.121*** | **joint** |
| 2.0 -> 5.0x | +28.1 %* | -23.1 %* | +0.022* | fail |

**One of three passes.** The 2.0 -> 5.0x step carries no nominal phase and fails
anyway, which refutes the earlier account that the 3.0x failure came from its
nominal opening. What separates them is the frozen policy's contact: 0.728 where
the arm wins, 0.841 and 0.871 where it trades.
(`RESULT_massstep_2026-09-06.md`)

## Ablations

| ablation | result |
|---|---|
| oracle (true payload) vs online estimate | online WINS at 3.0x and 5.0x |
| beta ladder 1.0 / 0.02 / 0.005 | monotone in smoothing at 3.0x; jitter and value-only both refuted |
| beta payload-dependence | 0.005 better at 3.0x, far worse at 5.0x; no single best |
| A/A on the headline arm | cut -10.6 %, lateral -0.5 %, contact 0.014 -- effects are 10-140x the floor |
| A/A on the frozen base, five runs | cut spans 15.67-23.56 mm (~40 %), contact 0.776-0.803 (~3 %) |
| reward re-pricing, 8x lateral | no effect at 3.0x, backfires at 5.0x |
| A-RMA phase 3 | fixes nominal, destroys cut and contact at every payload |
| freeze estimate pre-step | **PENDING** -- decides "tracks" vs "better on a varying plant" |
| 2.0 -> 5.0x step | **PENDING** -- adaptation without nominal contamination |

## Mechanism of the effect

The estimate must **arrive gradually**. Constant full-strength conditioning
(the oracle) collapses lateral; unsmoothed per-step estimates do nothing at all,
because they fluctuate faster than the policy responds; a smoothed ramp from
nominal works. Accuracy is not the mechanism -- at 3.0x the estimate is 3.11x
against a true 3.00x, essentially exact, and still behaves completely
differently from the constant truth. (`RESULT_beta_ladder_2026-09-06.md`)

## The limitation, with its cause

At nominal payload the arm loses lateral, 45.99 -> 81.50 mm. Four fixes failed:
a degree-2 estimator, a hard deadband, a slower EMA, and A-RMA phase 3.

The unifying explanation is structural. With only four context columns
trainable, the reachable behaviours form a roughly **one-dimensional family in
which cut and contact trade against lateral**. 1024 parameters entering one
layer cannot restructure the controller, only slide it along that curve.
Different training signals pick different points and none bends it: run 1 sits
at the cut/contact end, A-RMA at the lateral end, and an eightfold lateral price
merely moved the point.

Escaping it would need more adaptation capacity than four input columns, or
rebalancing the objective during the original distillation rather than
afterwards. Neither is tested and neither is claimed.
(`RESULT_arma3_2026-09-06.md`)

## Method notes that changed results

1. **The in-process sweep layout is invalid.** An A/A arm repeating the baseline
   scored -61.8 % cut against itself. Two same-day negatives were withdrawn.
   Use one arm per process. (`RESULT_AA_sweep_invalid_2026-09-06.md`)
2. **The cut channel's floor is ~40 %, not the 10 % the success rule assumes.**
   Ground contact measures the same physics with ~3 % spread and should be the
   primary vertical channel. (`RESULT_cut_instrument_floor_2026-09-06.md`)
3. **Verify the plant before reading the arm.** The first mass-step attempt
   scheduled a step at 20 s in a 15 s episode; the frozen-policy gate caught it.

## Outstanding

* **n = 1 seed.** The project's bar is a median over five. Not run.
* **The time-varying result is selection-mesh only.** The held-out confirmation
  covers constant payload.
* Two ablations pending, listed above.
