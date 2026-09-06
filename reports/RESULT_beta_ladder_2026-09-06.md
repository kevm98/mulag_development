# The mechanism is the trajectory, and slower is better

Payload 3.0x, mesh 2718, 176 environments, 900 steps. Every arm is the SAME
checkpoint (`ctxonly17/model_1596`). Only how the payload number arrives changes.

| arm | cut | lateral | contact | safety | vs ctxzero |
|---|---:|---:|---:|---:|---|
| `ctxzero` frozen base | 16.91 mm | 93.06 mm | 0.803 | 0.098 | - |
| ORACLE, constant 4.00 | 3.66 mm | 509.40 mm | 0.936 | 0.007 | +408.6 %* / -63.7 %* |
| `beta = 1.0`, no ramp, max jitter | 28.81 mm | 104.68 mm | 0.810 | 0.062 | -39.3 % / -6.5 % |
| `beta = 0.02` | 12.16 mm | 82.04 mm | 0.889 | 0.022 | +74.6 %* / +11.5 % |
| `beta = 0.005` | **7.20 mm** | **85.37 mm** | **0.891** | **0.014** | **+143.6 %* / +16.1 %*** |

## What it settles

**Smoothing is the mechanism, and the response is monotone in it.** Going
1.0 -> 0.02 -> 0.005 moves cut from -39.3 % to +74.6 % to +143.6 % and lateral
from -6.5 % to +11.5 % to +16.1 %. At `beta = 0.005` both channels improve with
intervals excluding zero: a joint improvement, better than the arm that produced
the held-out result.

**Two alternatives are eliminated.**

* *Jitter does the work.* Refuted: `beta = 1.0` has the most jitter and produces
  nothing at all, both channels non-significant. The raw estimate fluctuates
  faster than the policy responds, so its effect averages to nominal.
* *Only the value matters.* Refuted: the oracle presents a constant, perfectly
  smooth 4.00 and collapses lateral, while `beta = 0.005` converges to about
  4.22 and holds it. Nearly the same value, opposite outcomes.

What is left is the **ramp**. Every working arm starts each episode at nominal
and approaches the estimate over roughly `1/beta` steps. No constant context can
reproduce that, which is also why re-pricing the reward could not fix the oracle
(`RESULT_run3_pricing_refuted_2026-09-06.md`): a reward weight selects where on
the reachable family a run settles, and the ramp is a path through the family
rather than a point in it.

## The A/A on the headline arm

Payload 5.0x, the same configuration scored in two separate processes:

| | first | repeat | paired |
|---|---:|---:|---|
| cut | 15.95 mm | 16.89 mm | -10.6 %, significant |
| lateral | 109.05 mm | 102.06 mm | -0.5 %, ns |
| contact | 0.802 | 0.788 | 0.014 |

The headline is +1508 % cut, +216.3 % lateral and +0.241 contact. Those are
roughly 140x, 400x and 17x the A/A magnitudes. The result is far above this
instrument's floor on every channel.

## Consequences

1. **The mechanism is understood.** The claim no longer rests on an unexplained
   inversion. Online estimation helps because the estimate ARRIVES GRADUALLY,
   not because it is more accurate than the truth -- it is not, and at 3.0x it
   is nearly exact either way.
2. **`beta` is a genuine hyperparameter and must be reported as one.** It was
   set to 0.02 as a default before any of this ran, so the headline is not a
   tuned artefact, but the effect clearly depends on it and 0.005 is better.
3. **The held-out result stays as measured, at `beta = 0.02`.** Restating it at
   0.005 would need its own held-out run; the ladder was pre-declared as an
   ablation on the selection mesh and is not a licence to re-roll the headline
   on the reserved meshes.
4. **The ladder hit its edge.** 0.005 is the slowest rung tested and is the
   best AT 3.0x, so the optimum may lie beyond it there. Reported as a boundary
   rather than extended silently.

## CORRECTION, same day: beta is payload-dependent

The ladder above ran only at payload 3.0x, and the conclusion "slower is better"
was stated without qualification. Scored afterwards at 1.0x and 5.0x:

| payload | beta = 0.02 | beta = 0.005 |
|---|---|---|
| 1.0x | +11.3 %* / -43.9 %* | +48.6 %* / -38.0 %* |
| 3.0x | +74.6 %* / +11.5 % | **+143.6 %* / +16.1 %*** |
| 5.0x | **+1508.0 %* / +216.3 %*** | +331.1 %* / +129.6 %* |

At 5.0x the slower setting is dramatically WORSE, +331 % against +1508 % on cut
and +130 % against +216 % on lateral. The plausible reason is that at heavy
payload the estimate must REACH its value quickly to be useful, and a 200-step
time constant spends too much of the episode under-conditioned.

So there is no single best beta. Slower helps at moderate load and hurts at
heavy load, `beta = 0.02` remains the best overall choice, and the held-out
result stands exactly where it was measured. Generalising a single-payload
ablation to a global recommendation was an error, and the ladder's conclusion is
restricted to the payload it was run at.
