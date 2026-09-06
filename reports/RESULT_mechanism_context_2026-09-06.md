# Mechanism: the payload channel learns, and only the payload channel

Measured 2026-09-06 on the Stage A adaptive arm, before any of its task
performance was scored. Invariant 9: probe the mechanism before the outcome.

## The measurement

`PPO_AAM_FD/experiment_scripts/ctxweights.py`, mean |w| over the actor's input
layer `mlp.0.weight` (256, 70), split into the 66 task columns and the four
context columns `(alpha, tau, delay, mass_scale)`.

| checkpoint | base cols | context cols | alpha | tau | delay | mass_scale |
|---|---:|---:|---:|---:|---:|---:|
| frozen base `phase1_init_70` | 0.125463 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| adaptive `model_0` | 0.125508 | 0.000485 | 0.000000 | 0.000000 | 0.000000 | 0.001941 |
| adaptive `model_1596` | 0.144944 | 0.009999 | 0.000000 | 0.000000 | 0.000000 | 0.039994 |

## What it says

**The channel learns.** `mass_scale` goes from an exact 0.000000 to 0.039994,
about 28 % of the mean task-column weight. The policy is reading the payload.

**Only the channel that carries information learns.** `alpha`, `tau` and
`delay` are constant across this campaign, so they carry no gradient, and they
remain at exactly 0.000000 after 1596 iterations. This was declared as the check
in `ctxweights.py` before the number was read: had all four moved together, the
run would have been fitting noise. They did not.

## Why this settles the old failure

`PPO_AAM_FD/memory.md` records a 1000x gradient multiplier on these same columns
moving action sensitivity only from 0.001 % to 0.004 % of action scale, and reads
it as a plasticity problem that closed the anchored fine-tune route.

That reading was wrong, and this measurement is why. Those attempts ran on the
manifest's 0.8-1.5 payload range, which `env_cfg.py` itself records as costing a
fixed policy nothing measurable. A channel that predicts nothing produces no
gradient no matter how large a multiplier is applied to it. There was nothing to
learn, not an inability to learn.

On 1.0-5.0x, where the frozen base degrades 95x on cut height, the same columns
move on their own with no multiplier, no anchor change and no architectural
trick.

## What it does NOT say

It does not say the adaptive arm tracks better. Weight on a channel is
necessary for the payload to matter and is not sufficient for the task metric to
improve. The capacity-matched blind twin is what decides that, and it is scored
under `PREREG_PAYLOAD_CONDITIONING_2026-09-06.md` with the rule fixed in
advance.

Recorded here separately from the outcome so that the mechanism stands as its
own finding whichever way the gate falls.

## Correction carried alongside

`--stage_iterations` is per STAGE, not a total. The curriculum runs four stages
of 400, so the run ends at `model_1596`, and `model_399` is the end of STAGE 1
at `action_scale=0.0030, filter=0.85` rather than the deployment configuration
(`0.0040, 0.55`). The Stage A driver asked for `model_399`, which exists, so it
would not have failed loudly; it would have scored an undertrained stage-1
policy and labelled it the arm. Scoring was intercepted before it ran and both
arms are scored at the same final index, with the run aborted if the two arms
did not train to equal length.

## The blind twin, measured after training

| checkpoint | base cols | alpha | tau | delay | mass_scale |
|---|---:|---:|---:|---:|---:|
| blind `model_1596` | 0.134125 | 0.000000 | 0.000000 | 0.000000 | **0.000000** |
| adaptive `model_1596` | 0.144944 | 0.000000 | 0.000000 | 0.000000 | **0.039994** |

The blind arm's payload weight is still EXACTLY zero after 1596 iterations. Its
context is zeroed every step, so those columns receive no gradient and cannot
move. It is therefore structurally incapable of using the payload while being
identical in width, architecture and parameter count.

Both arms moved their task columns off the base's 0.125463, to 0.134125 and
0.144944, so both genuinely trained on the task rather than idling.

This is what makes the comparison tight. The two arms

* started as literally the same policy, because the base's context weights are
  exactly zero so the context value could not change its output;
* trained for the same 1596 iterations, on the same plant, at the same seed,
  under the same objective, with the run aborted if the two lengths differed;
* differ only in whether the payload channel carried information, and that
  difference is visible in the weights rather than assumed.

Any task difference between them is therefore attributable to the payload
channel. Whether there IS one is the gate's business, not this file's.
