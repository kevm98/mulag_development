# The unanchored fine-tune walks away, and the fix is 1024 parameters

Measured 2026-09-06 during Stage A of `PREREG_PAYLOAD_CONDITIONING_2026-09-06`.
Reported because it changes the design, not as a result about adaptation.

## What happened

Both arms were fine-tuned unanchored from the distilled init on
`MULAG_PAYLOAD_RANGE=1.0,5.0` for 1596 iterations. Scored at fixed payload 3.0x
on mesh 2718:

| arm | cut | lateral | contact | impacts | safety |
|---|---:|---:|---:|---:|---:|
| frozen base | 16.70 mm | 99.66 mm | 0.788 | 0.464 | 0.112 |
| blind | 1782.01 mm | 59.52 mm | **0.012** | 0.005 | 0.959 |

The blind arm is airborne 99 % of the time. It won lateral error, 59.5 mm
against the base's 99.7 mm, by abandoning the ground entirely: no contact, no
disturbance, no cut. That is not a better controller, it is a controller that
left the task.

`memory.md` predicts exactly this: "Without the anchor, PPO is free to walk away
from the distilled behaviour, and does -- it will trade line-tracking for ground
contact because that raises its objective." Here the trade runs the other way,
buying lateral with contact, because the objective was rebalanced to weight
lateral at 4.0 with a 0.09 m kernel. That rebalance was calibrated for a BOUNDED
residual at alpha = 0.5. Applied to an unbounded full-policy fine-tune it is
exploitable, and PPO exploited it.

## The one thing worth keeping from it

Training-log ground contact at the end of training:

    adaptive  0.457      contact_total  +0.44
    blind     0.023      contact_total  -2.96

Same objective, same budget, same seed, same architecture. The arm that could
see the payload held about 20x the ground contact of the one that could not. So
payload information does move the quantity that matters. But both fell below the
frozen base's 0.788, so neither is deployable and the comparison between them
measures which arm degraded less. That is not the result being sought and it is
not reported as one.

## The fix

`--train_context_only`. Freeze every actor parameter and mask the input layer's
gradient so only the context columns move: 4 columns x 256 hidden = **1024
trainable parameters**, against 59,588 in the actor.

It cannot collapse, and the reason is structural rather than a matter of tuning:

* a nominal plant normalises to the zero context vector, and the base's context
  weights are exactly zero, so at iteration 0 the policy IS the frozen base;
* the task columns are frozen, so nothing the policy does on the nominal plant
  can change at all;
* all the run can learn is how to modulate the frozen policy by the payload.

It also collapses the control structure. With the context zeroed, the trained
policy computes `base(x) + W_ctx . 0 = base(x)`, so **the blind twin is the
frozen base, bit for bit**, and does not need to be trained to be available. The
falsification arm and the baseline coincide, and any gain of the live-context arm
over the zeroed one is the payload channel and nothing else.

That is a stronger control than Stage A's, obtained by removing capacity rather
than adding an arm.

## Consequence for the campaign

Stage A scoring was cut after the payload 3.0 block. Scoring the same damaged
policies at 5.0, 1.0 and mixed would have spent about an hour of GPU restating
that the fine-tune broke them. The smoothing oracle is unaffected: it runs on the
frozen base.
