# Pre-registration: does online adaptation help?

Written and committed **before the arms were run**. The design, the metric, the
analysis and the pass condition are fixed here so the result cannot be shaped
after seeing it.

## Claim under test

A history-conditioned residual on a frozen policy recovers error created by a
mid-episode plant change, and it does so **because it senses the change online**
-- not merely because it adds parameters.

## Design

Frozen distilled base policy (`distilled_v1/phase1_init_70.pt`, verified exactly
context-invariant). On top of it:

    u = clip(u0 + alpha * g(o, phi(H)), -1, 1),  alpha = 0.25

`phi` (temporal-conv history encoder) and `g` (residual head) are the ONLY
trained parameters: 53,836 trainable against 59,588 frozen. The residual head is
zero-initialised, so **both arms start as bit-for-bit the same frozen policy**
(verified: max |wrapped - base| = 0.0).

## Plant

* payload drawn per environment in 1.0-1.4x
* **cutter mass steps x2.0 mid-episode**, at 20 s +/- 8 s **jittered per
  environment**

The jitter is not cosmetic. A step that always fires at the same instant can be
compensated on a timer, so a NON-adaptive arm could match an adaptive one and
hide the effect. Jitter forces the change to be detected rather than anticipated.

Mass is the right axis: it reaches PhysX natively (`set_masses`), unlike the four
hydraulic factors which are inert on the FD stack. Measured degradation on this
policy: cut error **+180%** at 1.5x, **+334%** at 2.0x, with the policy still
functioning (ground contact 0.875 -> 0.798).

## Arms

Identical script, identical seed, identical capacity, identical training budget.
One flag differs:

| arm | history |
|---|---|
| **adaptive** | live |
| **control** | `--no_history` -- zeroed, so the residual CANNOT adapt |

This pair separates "a residual head helps" from "**online adaptation** helps".
The second is the claim; the first is not interesting.

## Metric

**Primary: cut-height error.** Payload moves cut error +180%/+334% while lateral
moves +10%/+26%, so lateral would understate the effect ~10x. Lateral is reported
as secondary, not as the decision.

## Analysis, fixed in advance

* paired, block-per-cycle per-env arms; cycle DERIVED as
  lcm(#path families, #height scales)
* `MULAG_PAIR_PERIOD` so paired environments draw the same plant
* pooled across 6 meshes, one fresh process each (a single process at 6 meshes
  thrashed at 28/31 GiB and hung)
* bootstrap CIs (20k resamples) on the **paired median log-ratio**
* a **null (zero-action) arm** must be reported alongside -- Stage 0 recorded a
  do-nothing controller BEATING a frozen PPO because the episodes demanded no
  motion. Current check: null is 114% worse on cut, 109% on lateral, so the task
  does demand control.
* an **A/A control** (two identical arms) must return CIs including zero

## Pass condition

**The adaptive arm's cut-height error is lower than the control arm's, with the
bootstrap 95% CI on the paired median log-ratio excluding zero.**

Anything else is a null and will be reported as one. Specifically:

* CI spanning zero -> null, regardless of point estimate
* a win on lateral but not cut height -> null (lateral is secondary)
* a win that disappears when the A/A control is applied -> instrument artifact

## What would invalidate the result

* the residual's action does not depend on the history (probe it; the RMA context
  route failed exactly here at 0.004% of action scale)
* the null arm beats either trained arm
* the A/A control produces a difference of the same size as the effect
