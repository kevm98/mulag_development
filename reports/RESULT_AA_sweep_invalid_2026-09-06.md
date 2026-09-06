# The A/A fails: the paired-in-process sweep is not a valid instrument

Run 2026-09-06 on mesh 2718, payload 3.0x, 176 environments, 900 steps, frozen
base. The last arm repeats the baseline setting, at a rate differing in the
fourth decimal so it lands under its own key.

## Result

Baseline `0.55:0.25`: cut 23.31 mm, lateral 94.19 mm, contact 0.747,
safety 0.110.

| beta:rate | cut | lateral | contact | accel | safety |
|---|---:|---:|---:|---:|---:|
| 0.70:0.25 | -29.5 %* | -9.4 % | 0.729 | 81.1 | 0.141 |
| 0.85:0.25 | -29.9 %* | -41.7 %* | 0.689 | 86.6 | 0.153 |
| 0.60:0.06 | -30.6 %* | -21.7 %* | 0.749 | 68.2 | 0.110 |
| **0.55:0.2499 (A/A)** | **-61.8 %*** | **-43.1 %*** | **0.574** | **150.0** | **0.296** |

The A/A arm is the SAME SETTING as the baseline. It scores -61.8 % on cut and
-43.1 % on lateral against itself, both intervals excluding zero, with ground
contact 0.747 -> 0.574, p95 tool acceleration 86.7 -> 150.0 and the violation
rate 0.110 -> 0.296.

It is also the WORST arm in the table, worse than every genuine parameter
change. The penalty for not being first exceeds every effect the sweep was
built to measure, and it grows with position: the A/A sits last.

## What is withdrawn

**Both sweep negatives, unconditionally.**

* `PREREG_ORACLE_PHI_2026-09-06.md` -- "no two-parameter task-space offset
  Pareto-improves at 1.0x or 3.0x". Same layout, same baseline-first ordering.
  **Withdrawn.** The question is reopened, not answered.
* The smoothing oracle at 1.0x, 3.0x and 5.0x. **Withdrawn.**

`RESULT_inprocess_artifact_2026-09-06.md` raised this as a suspicion and offered
a competing explanation -- that any perturbation away from a trained operating
point hurts. The A/A rules that competitor out: the A/A arm is not a
perturbation at all, and it is penalised hardest.

## What is NOT affected

* **The context-only campaign.** It scores every arm in its own process and
  pairs by environment index, which is invariant 3's other permitted layout. It
  never places two arms in one process.
* **The root-cause report.** Action saturation, contact, impacts and tool
  acceleration there are single-arm properties of the frozen base, taken from
  first-rollout or dedicated-process runs, not within-sweep comparisons.
* **The mechanism results.** Context weights are read off checkpoints and
  involve no rollout comparison at all.

## The corrected design

One arm per process, paired by environment index under common random numbers,
exactly as the context-only campaign already does. It costs about six minutes
per arm instead of two and a half, and it is the only layout on this stack whose
A/A has been checked.

The task-space oracle is re-run under that design before anything is concluded
about it. That matters beyond bookkeeping: a two-parameter reference offset is
the simplest adaptation mechanism available here, and it was closed on a
measurement that has now failed its own control.

## The invariant this adds

Invariant 3 named two safe layouts. A third was invented, looked equivalent,
and is not. **Any new comparison layout gets an A/A before its results are
read**, which is invariant 1 applied to the layout rather than to the
checkpoint. The A/A cost one 25-minute run and it overturned two conclusions.
