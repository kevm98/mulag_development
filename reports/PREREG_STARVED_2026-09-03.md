# Pre-registration: starved-encoder adaptation campaign

Written 2026-09-03, **before any arm of this campaign was trained**. The success
rule below is fixed and is not re-decided after the seeds are seen.

## Question

Does live interaction history improve closed-loop tracking **beyond a
capacity-matched residual that reads zeros**?

The 2026-09-02 campaign answered no: history separated from zeroed history on
1 of 5 seeds under both checkpoint rules, while capacity alone beat the frozen
base on 4 of 5. This campaign re-tests the question after removing three
identified confounds. It does not re-test the residual claim, which stands.

## What was wrong with the previous campaign

1. **The encoder read the whole observation.** 70 features per step, 1400
   numbers over a 20-step window, almost all terrain and task state, into a
   53k-parameter head. Extra weights could act as a second policy, so zeros
   remained competitive.
2. **The primary comparison mixed checkpoints.** Control at `model_150` against
   adaptive at `model_300` confounds live history with 150 extra iterations.
3. **The primary metric was full-episode error against the frozen base.** That
   is a test of the residual, and a more expressive residual wins it regardless
   of what its encoder reads.

A fourth observation shaped the design: on seed 17 the history advantage was
already present *before* the mid-episode flip (difference-in-differences
+0.85 mm, interval [-5.10, +11.01], not resolved). The flip is therefore a poor
primary factor for this plant. A stable operating point is the effect that was
actually measured.

## Design

Three arms per seed. `control` and `adaptive` share the frozen base,
architecture, trainable parameter count, training budget, reward, optimiser,
plant schedule and **checkpoint**. The single manipulated variable is whether
the encoder reads real numbers or zeros.

| Arm | Residual | History input |
|---|---|---|
| `base` | off | — |
| `control` | on | zeroed |
| `adaptive` | on | live, plant-relevant columns |

Two changes from the previous campaign, and only two:

1. **Starved encoder** (`--history_cols plant`): tracking error, cylinder
   extension and velocity, valve command, measured-minus-commanded cylinder
   state, actuator prediction residual, normalised cylinder effort. 26 columns
   plus the 4-wide previous action, 30 features per step, 600 numbers over the
   window instead of 1400. Terrain profile, headroom, target lead, alignment,
   truck speed, path preview and cutter pose are excluded, so the head cannot
   clone the task policy. The column set is written into the checkpoint as a
   buffer, so an evaluation cannot feed a different set from the trained one.
2. **Constant per-episode cutter mass** drawn over the configured range, held
   for the episode. No mid-episode flip.

Reward, residual width, alpha, learning rate, history length, latent dimension,
iteration count and environment count are unchanged from the 2026-09-01 runs.

## Fixed before the first run

- **Primary comparison:** `adaptive` vs `control`, paired per environment.
- **Secondary:** `adaptive` vs `base`. Reporting only this one would repeat the
  previous campaign's error, since it cannot separate history from capacity.
- **Context:** `control` vs `base`.
- **Checkpoint:** both residual arms scored at the same final checkpoint. No
  per-arm training-reward peak.
- **Success rule:** one channel improves with its paired bootstrap interval
  excluding zero, and the other improves or stays within its non-inferiority
  bound (cut height 10%, lateral 1.5%). Identical to the Table 7.6 rule.
- **Statistic:** paired per-environment median difference, percentile bootstrap
  B=20000; percentage as the median log-ratio.
- **Verdict:** the campaign is positive if the **median seed passes the PRIMARY
  comparison**.

## Declared outcomes

- **Must have:** primary succeeds on >= 3/5 seeds.
- **Strong:** 4/5, and the error-versus-mass slope is flatter for `adaptive`
  than for `control`.
- **Negative:** if the primary median fails, live history is not the source of
  the gain under this encoder either, and the thesis reports the residual result
  without a history headline. The rule is not relaxed afterwards.

## Gate before paying for five seeds

The frozen base must actually degrade with cutter mass. If a fixed policy is
insensitive to the mass draw, the encoder has nothing to identify and the
campaign is not worth running. Measured first, on the frozen base alone, over a
constant-mass ladder. If the tested range does not move the base, the range is
widened until it does, and that widened range is the one the campaign uses.

## Stopping rule

If the primary median fails with the starved encoder, exactly one further
campaign is run: regress the encoder onto the known plant factor, freeze it, and
train only the residual head, keeping the zero-history twin. No further
variants, no new losses, no wider head.
