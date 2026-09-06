# Correction: lateral was never gradient-free, it was under-priced

Written 2026-09-06 after cancelling run 2 at iteration 300 of 1600.

## What I claimed

`RESULT_ctxonly_run1_2026-09-06.md` attributed the lateral collapse
(93 -> 509 mm at payload 3.0x) to a saturated Gaussian: training error ran
0.18-0.55 m against `lateral_tracking_std_m = 0.09`, so
`exp(-(e/sigma)^2)` sits at 1e-7 to 1e-11 and its gradient with it. The stated
conclusion was that "lateral supplied no learning signal", and run 2 was
launched with `sigma = 0.25` to fix it.

## Why that was wrong

There are TWO lateral terms, not one:

    Gaussian   lateral_tracking_weight * exp(-(e / lateral_tracking_std_m)^2)
    Dense      -dense_lateral_error_weight * clamp(|e|, max=3.0)

The dense term is LINEAR in the error and clamped only at 3 m, so its gradient
is constant and non-zero everywhere the policy actually operates. Lateral always
had a learning signal. The Gaussian saturating is real and irrelevant.

The evidence was in the run before I read it properly: run 2 at sigma = 0.25
produced a lateral trajectory and an `Episode_Reward/lateral` band (4.7-9.0)
indistinguishable from run 1's (4.3-10.4). Changing sigma by a factor of 2.8
changed nothing, which it could not have done if sigma were the binding term.

## What is actually happening

A pricing problem, and the policy is behaving correctly under the prices it was
given. Going from `ctxzero` to `ctxlive` at payload 3.0x:

    cut Gaussian gain   8.0 * [exp(-(0.0037/0.03)^2) - exp(-(0.0169/0.03)^2)]
                      = 8.0 * [0.9848 - 0.7274]  =  +2.06 per step
    plus contact rewards, call the total roughly +3
    lateral dense cost  1.0 * 0.416 m            =  -0.42 per step

It traded 0.42 for about 3.0. That is not a defect in the mechanism, an
objective bug in the sense of a missing gradient, or a failure of the context
columns. It is the correct answer to a badly weighted question.

## The fix, and why it is one change

`dense_lateral_error_weight` 1.0 -> **8.0**, with `lateral_tracking_std_m` left
at the project's tuned 0.09. At 8.0 the same 0.416 m excursion costs 3.33 per
step, which exceeds the roughly 3.0 it buys, so the trade stops being
attractive. Nothing else changes from run 1.

Run 2 changed sigma, which the evidence now says is inert here. Run 3 changes
the weight that actually carries the gradient. Only one thing moves, so the
comparison with run 1 stays clean.

## Cost and lesson

Run 2 was cancelled at iteration 300, so this cost about 20 minutes rather than
the 2.5 hours it would have cost had it run to completion and produced a null
that looked like "conditioning cannot hold the line".

The lesson is narrow and worth keeping: **read the whole reward before blaming
one term of it.** The Gaussian was the term named in `PPO_AAM_FD/memory.md`, so
it was the term I checked, and I stopped there. The dense companion term is four
lines below it in `terrain_env.py` and inverts the diagnosis.

`RESULT_ctxonly_run1_2026-09-06.md` keeps its measurements, which are unaffected;
its stated cause for the lateral collapse is superseded by this file.
