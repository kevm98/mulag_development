# Pre-registration addendum: extending the starved-encoder campaign to 10 seeds

Written 2026-09-05, **after** the first five seeds were scored and **before** any
extension seed was trained. This is an extension decided on the strength of an
observed result, and it is recorded as such rather than presented as part of the
original plan.

## What the first five seeds gave

Primary (`control` -> `adaptive`, live history at equal capacity): 2 of 5 seeds
pass. The median seed fails against the declared must-have of 3 of 5. The
original campaign is therefore **negative on its own terms**, and that verdict
stands regardless of what this extension returns.

Two further observations shaped this decision:

- Of the two passing seeds, only seed 17 improves a *working* control. Seed 7's
  control collapses below the frozen base on both channels, so history there
  recovers a degenerate run rather than adding to a functioning one.
- The variation does not track operating conditions. Mesh 2718 is the hardest
  for the frozen base and is not where the advantage concentrates, and the
  history arm is more stable across meshes on two seeds and less stable on two
  others. What separates the arms is the training run, not the terrain.

With n = 5 and 2 passes, "history helps on a minority of runs" and "history does
nothing and this is sampling noise" are not distinguishable. That is the only
question this extension exists to answer.

## What changes

Nothing except the number of seeds. Same frozen base, same starved encoder
(`--history_cols plant`), same constant per-episode cutter mass over 1.0 to 5.0,
same 400 iterations, same alpha, same final-checkpoint rule, same three reserved
meshes, same 528 paired environments per seed.

Five additional seeds: **1, 2, 3, 4, 5.** These are the first five positive
integers, none of which is already used. The list is fixed here so it cannot be
tuned, and no seed is dropped after the fact for any reason.

An earlier draft of this addendum declared ten seeds (1-6, 8-11) for a 15-seed
campaign. It was cut to five for run time before any extension seed was trained,
and the five kept are the first five of that declared list rather than a
selection made later.

## Fixed before the first extension run

- **Success rule per seed:** unchanged. One channel improves with its paired
  bootstrap interval excluding zero, and the other improves or stays within its
  non-inferiority bound (cut height 10 %, lateral 1.5 %).
- **Statistic:** unchanged. Paired per-environment median difference, percentile
  bootstrap B = 20000, percentage as the median log-ratio.
- **Campaign verdict over all 10 seeds:** positive only if **6 or more of 10**
  pass the primary, which is the same median-seed rule as before.
- **Reporting:** the original five and the five extension seeds are reported
  separately as well as pooled. The 2-of-5 result is not replaced by the pooled
  number.

## Stopping rule

The extension may be stopped early for time or cost. If it is, the analysis uses
**every seed whose two arms are both scored at the moment of stopping**, and the
stop must not be triggered by the results. Stopping because the numbers look
favourable, or continuing because they do not, invalidates the extension and it
will be reported as inconclusive.

No seed is excluded after being trained. A seed whose control collapses below the
frozen base is kept and flagged, as seed 7 is.

## Declared outcomes

- **Positive:** >= 6 of 10 pass. Live history is a real if unreliable effect on
  this plant.
- **Negative:** < 6 of 10. The first campaign's verdict is confirmed at twice the
  sample size, and the thesis reports that live history is not a reliable source
  of gain here.
- Either way the encoder explanation stays eliminated, since the starved encoder
  is used throughout.


## Outcome and stop

Stopped 2026-09-05 at 16:37 CEST after four of the five extension seeds, with
seed 5's control arm part-trained and discarded. Analysis therefore uses every
seed whose two arms are both scored: **seeds 17, 23, 101, 42, 7, 1, 2, 3**, plus
seed 4 if its reports validate.

The stop was **not** triggered by the results. Two facts fix that. First, the
verdict was already arithmetically determined before the stop: 3 passes from 8
scored, 2 seeds outstanding, maximum reachable 5 against a threshold of 6.
Continuing could not have changed the outcome in either direction. Second, the
GPU was reassigned to a different experiment
(`PREREG_ONLINE_GRAD_2026-09-05.md`) at the author's direction.

**Result: the primary fails.** Live history at equal capacity succeeds on 3 of 8
seeds under the pre-registered rule, against a declared threshold of 6 of 10.
Reported with the original 2-of-5 alongside it, not replaced by it.
