# Pricing lateral eight times higher changes nothing

Run 3 (`ctxonly17d`) repeats run 1 with `dense_lateral_error_weight` 1.0 -> 8.0
and nothing else changed. Payload 3.0x, mesh 2718.

| run | arm | cut | lateral | contact | safety |
|---|---|---:|---:|---:|---:|
| 1 | ctxzero | 16.91 mm | 93.06 mm | 0.803 | 0.098 |
| 1 | ctxlive | 3.66 mm | 509.40 mm | 0.936 | 0.007 |
| 3 | ctxzero | 27.17 mm | 112.42 mm | 0.750 | 0.139 |
| 3 | ctxlive | 2.87 mm | **525.95 mm** | 0.936 | 0.009 |

Paired: run 1 `ctxzero -> ctxlive` is +408.6 %* cut / -63.7 %* lateral; run 3 is
+1183.0 %* cut / -61.7 %* lateral. The collapse is unchanged.

## Two explanations now refuted

1. **Saturated Gaussian.** Refuted by `CORRECTION_lateral_diagnosis_2026-09-06`:
   a second, linear lateral term carries gradient everywhere, and changing sigma
   by 2.8x changed nothing.
2. **Under-pricing.** Refuted here: multiplying the term that DOES carry the
   gradient by eight moved lateral from 509.40 to 525.95 mm, i.e. not at all.

An eight-fold price change producing no behavioural change is itself
informative. It says the policy is not choosing between cut and lateral on the
margin at all. Whatever the context columns do at full conditioning strength,
it is not a trade the objective can re-price.

## What the evidence now points at

The oracle and the online arm are the SAME checkpoint. They differ only in the
context VALUE presented at score time:

    ctxlive   constant 4.00 (the true payload, normalised)  ->  lateral 509 mm
    ctxest    EMA-ramped, converging to about 4.22          ->  lateral  82 mm

Nearly identical values, a six-fold difference in outcome. So the determining
factor is not the number but how it arrives: either the ramp from nominal at
episode start, or the per-step dither around the converged value.

The queued beta ladder separates these. `beta = 1.0` removes the ramp and
maximises jitter; `beta = 0.005` slows the ramp and minimises jitter. If the
ramp matters, 1.0 should behave like the oracle; if jitter matters, 1.0 should
behave like the online arm. That experiment is now the most important one
outstanding, because the headline result currently rests on a mechanism that is
not understood.

## A third A/A, unplanned

Run 1 and run 3 `ctxzero` are the same frozen policy by construction: both
compute `base(x) + W_ctx . 0`. They read **16.91 and 27.17 mm** on cut, 61 %
apart, and 93.06 against 112.42 mm on lateral.

That is a third independent confirmation that the cut-height channel's
run-to-run floor is far worse than the 10 % the project's success rule assumes,
and it further supports scoring ground contact, which read 0.803 against 0.750
across the same pair.

## Payload 5.0x: the intervention backfired

| run | arm | cut | lateral | contact | safety |
|---|---|---:|---:|---:|---:|
| 1 | ctxzero | 328.50 mm | 455.85 mm | 0.539 | 0.363 |
| 1 | ctxlive | 45.97 mm | 385.95 mm | 0.634 | 0.299 |
| 3 | ctxzero | 246.66 mm | 377.64 mm | 0.610 | 0.286 |
| 3 | ctxlive | 16.21 mm | **745.30 mm** | 0.794 | 0.123 |

Paired: run 1 is +166.8 %* cut / **+0.2 %** lateral, a pass. Run 3 is +939.6 %*
cut / **-51.7 %*** lateral, a clear fail.

Multiplying the lateral penalty by eight made lateral **worse**, 385.95 ->
745.30 mm. Not neutral, backwards. Together with the null at 3.0x, the reward
weight does not control this behaviour in either direction.

## A structural hypothesis

With only the four context columns trainable, the policy's reachable behaviours
are a very low-dimensional family: 1024 parameters entering one layer cannot
restructure the controller, only slide it along whatever curve those columns
parameterise. If better cut and contact co-vary with worse lateral along that
curve, then a reward weight can choose WHERE on the curve the run lands but
cannot bend the curve, which is exactly the pattern observed -- an eight-fold
price change moving the operating point without ever buying lateral back.

That would also explain how the online arm escapes it. The EMA does not select a
different fixed point on the curve; it produces a TIME-VARYING trajectory
through it, starting at nominal and converging. A trajectory is not reachable by
any constant context value, whatever the reward weights.

This is a hypothesis with one supporting pattern, not a finding. The beta ladder
tests its central claim directly.

## Run 3 is abandoned as a direction

The run-1 configuration stays the basis of the reported result. Run 3's value is
entirely negative: it removes reward pricing from the list of explanations, and
it contributes a fourth A/A data point, since run 1 and run 3 `ctxzero` are the
same frozen policy by construction and read 328.50 against 246.66 mm on cut at
5.0x, 33 % apart.
