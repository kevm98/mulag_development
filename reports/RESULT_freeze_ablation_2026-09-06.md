# The benefit is conditioning, not tracking

Payload steps 1.0 -> 3.0x at 6 s +- 2 s, mesh 2718, 176 environments, 900 steps.
`MULAG_ESTIMATOR_FREEZE_S=4.0` latches the estimate at 4 s, BEFORE the step
window opens, so that arm carries a pre-step estimate for the rest of the
episode and cannot follow the change. Everything else is identical.

The reading was declared in `PPO_AAM_FD/experiment_scripts/freeze.sh` before the
run:

> live beats frozen -> the arm genuinely tracks the payload change
> live equals frozen -> the benefit is conditioning, not tracking, and the
> claim must be worded accordingly

## Result

| arm | cut | lateral | contact | safety |
|---|---:|---:|---:|---:|
| `ctxzero` (frozen policy) | 11.02 mm | 63.04 mm | 0.841 | 0.069 |
| `ctxest` FROZEN at 4 s | 4.67 mm | 76.69 mm | 0.925 | 0.008 |
| `ctxest` LIVE | 3.71 mm | 81.30 mm | 0.927 | 0.009 |

Paired frozen -> live: cut **+19.1 %, ns**; lateral **-3.2 %, ns**. Neither
channel differs significantly.

## What it establishes

**Updating the estimate within an episode adds nothing measurable.** An estimate
latched before the payload even changes performs as well as one that follows it.

**The estimate is still doing work.** The frozen-estimate arm beats the frozen
POLICY substantially: cut 11.02 -> 4.67 mm, contact 0.841 -> 0.925, violations
0.069 -> 0.008. And the estimate's VALUE matters across payloads -- it helps at
5.0x and hurts at 1.0x. So this is not "the context is ignored"; it is "the
context is identified early and does not need to be revised".

**A second, independent ablation agrees.** The beta ladder found `beta = 0.005`,
whose 3.3 s time constant should badly lag a step at 4-8 s, matching
`beta = 0.02` at both step doses. Two different manipulations of the estimator's
ability to follow a change both find it does not matter.

## Consequences for the claim

Supported:

* **online payload identification** -- the value is estimated at runtime from
  the task observation, never given
* **payload-conditioned control recovers ground contact when a frozen controller
  is losing it**

NOT supported, and not to be written:

* "adapts to a changing payload"
* "tracks the payload online"
* any phrasing whose force comes from the estimate following the plant

The honest description is **online identification feeding a conditioned policy**.
That is what the RMA family actually does, and it is a real result, but the
strong sense of "adaptation" is not earned by this evidence and the ablation
that would have earned it came out negative.

## What would earn the stronger claim

A plant whose payload changes in a way that a single early identification cannot
cover -- several changes per episode, or a continuously varying payload -- would
force the estimate to be revised for the arm to keep working. Neither is tested
here. Under a single mid-episode step in a 15 s episode, one early
identification is evidently sufficient, so the experiment cannot separate the
two hypotheses in the arm's favour no matter how it comes out.
