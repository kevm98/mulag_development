# The paired-in-process sweep has a first-rollout artifact

Found 2026-09-06 while reading the smoothing oracle. It calls into question two
negative results produced earlier the same night, both of them mine.

## The pattern

Both oracles sweep several settings inside ONE process, baseline first, to pair
on the environment. Across both, at every payload, EVERY non-baseline arm is
50-80 % worse on cut height, and the size of the penalty does not track the
parameter.

Smoothing oracle, payload 1.0x, cut change against the in-process baseline:

| beta:rate | cut |
|---|---:|
| 0.35:0.25 | -81.4 % |
| 0.70:0.25 | -57.3 % |
| 0.85:0.25 | -76.2 % |
| 0.60:0.25 | -65.0 % |
| 0.60:0.12 | -57.3 % |
| 0.60:0.06 | -78.7 % |

`beta` 0.35 and 0.85 are opposite extremes of the smoothing coefficient and give
-81 % and -76 %. `rate` 0.25 and 0.06 differ four-fold and give -65 % and -79 %.
There is no monotone relation with either knob.

The task-space oracle at the same payload gave the SAME range on a completely
different parameter: -57.2 % to -79.8 % across nine offsets.

Two unrelated knobs, opposite directions, one shared penalty. That is not a
parameter effect.

## The corroboration that costs nothing

The frozen base at payload 3.0x, mesh 2718, 176 environments, 900 steps:

* as the FIRST rollout inside the task-space sweep: **15.67 mm** cut
* in its OWN dedicated process, Stage A `base_p3.0`: **16.70 mm** cut

Those agree to 6 %, inside the cross-process instability invariant 19 documents.
So a first in-process rollout reproduces a dedicated process, and only the
LATER in-process rollouts are degraded. The baseline is always first.

## What it means

The negatives in `PREREG_ORACLE_PHI_2026-09-06` and the smoothing sweep were
each read as "no setting of this knob beats the baseline". If later rollouts in
a process are penalised regardless of setting, that reading is unsafe: every arm
was compared against a privileged reference.

**Neither negative is withdrawn on suspicion, and neither is defended.** An A/A
control is running: the payload 3.0x smoothing grid repeats the baseline setting
as its LAST arm, at a rate differing in the fourth decimal so it lands under its
own key. Two nominally identical settings, first and last in the same process.
The gap between them is the artifact, measured rather than argued, and it bounds
how much of both negatives is real.

## What is NOT affected

The context-only campaign scores every arm in its OWN process and pairs by
environment index, which is invariant 3's other permitted layout. It never puts
two arms in one process, so this artifact cannot reach it. That is the campaign
carrying the current claim, and it was already built that way before this was
found.

The root-cause report is also unaffected: its numbers are single-arm properties
of the frozen base (action saturation, contact, impacts, tool acceleration) read
from first-rollout or dedicated-process runs, not comparisons between arms
inside a sweep.

## The lesson, for the invariants

Invariant 3 says compare arms paired, never sequentially in one process, and
offers "per-env arms in blocks, or separate processes with common random
numbers" as the two safe layouts. The paired sweeps used a THIRD layout not on
that list: sequential rollouts in one process, paired by environment. It looked
equivalent and is not. Whatever the mechanism turns out to be, the layout needs
its own A/A before it is trusted again, which is exactly what invariant 1 says
about any new instrument.
