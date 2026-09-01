# Pre-registration addendum — both-metric pass condition

Fixed 2026-08-30 ~13:00, BEFORE seeds 42, 23 and 17 were scored at 1200
iterations. Seeds 101 and 7 were already scored at 1200 when this was written
and are listed below as known values, not as evidence for the criterion.

## The claim being tested

Online adaptation improves **both** cut-height error and lateral error against
the frozen baseline. Not one at the expense of the other.

## Pass condition, fixed in advance

An arm PASSES if, paired against the frozen base over 528 environments and three
held-out meshes:

* cut-height error improves, bootstrap CI excluding zero, AND
* lateral error improves, bootstrap CI excluding zero.

The claim holds if the **median seed passes** at equal budget (1200 iterations,
five seeds). Reporting the best seed is not a result.

An arm that improves one metric and degrades the other is a TRADE-OFF and is
recorded as a failure of this claim, however large the winning side is.

## Why this supersedes "read cut height, not lateral"

That rule is about PERTURBATION SENSITIVITY: payload mass moves cut error ~10x
more than lateral, so cut height is the channel that reveals what a plant change
costs. It was never a success criterion. A controller can win on cut height by
spending lateral, and seed 101 did exactly that (+23.1% cut, -61.2% lateral).
The two uses of the metric are different questions; conflating them is how a
trade-off gets published as a benefit.

## State at the time of writing (not evidence for the criterion)

| seed | budget | cut | lateral | verdict |
|---|---|---|---|---|
| 17  | 400  | +160.8% | +35.5% | both better |
| 23  | 400  | +40.7%  | +27.4% | both better |
| 42  | 400  | +24.1%  | -37.3% | trade-off |
| 101 | 400  | +7.1%   | -62.0% | trade-off |
| 7   | 400  | -59.3%  | -17.2% | both worse |
| 101 | 1200 | +23.1%  | -61.2% | trade-off |
| 7   | 1200 | -15.3%  | -4.4%  | both worse |

Across seeds the two metrics are positively rank-correlated (Spearman ~0.70):
they move together, so "both better" is an achievable target rather than a
contradiction. The one lever with evidence behind it is training budget — seed 7
improved on BOTH metrics from 400 to 1200 with nothing else varied.

## Read

If seeds 42 and 101 move into "both better" at 1200, budget was the blocker and
the claim stands on the median seed. If they stay in the trade-off zone, budget
is not sufficient and that is the finding to report — not a reason to fall back
to reporting cut height alone.
