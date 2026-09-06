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
