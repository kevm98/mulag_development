# Online payload adaptation: 4/4 on held-out meshes

Scored 2026-09-06 on meshes **3141 and 8191**, which were never used to design,
diagnose or read this campaign. 176 environments, 900 steps, paired per
environment, percentile bootstrap B = 20000.

## The arms

Both are the SAME checkpoint, `ctxonly17/model_1596`. Only the source of the
payload number differs.

* `ctxzero` -- context zeroed. It computes `base(x) + W_ctx . 0 = base(x)`, so it
  IS the frozen policy, bit for bit. The control is provable, not trained.
* `ctxest` -- payload estimated at runtime from the 66 task observation columns
  (cylinder effort, actuator prediction residual, cylinder state). Never reads
  the simulator's value. The estimate starts each episode at nominal and
  converges while driving.

Only **1024 parameters** were trained, against 59,588 in the actor, and the task
columns were verified bit-identical throughout at mean |w| = 0.125463.

## Result

| mesh | payload | arm | cut | lateral | contact | safety |
|---|---:|---|---:|---:|---:|---:|
| 3141 | 3.0x | ctxzero | 20.79 mm | 122.98 mm | 0.783 | 0.130 |
| 3141 | 3.0x | **ctxest** | **3.97 mm** | **89.01 mm** | **0.925** | **0.010** |
| 8191 | 3.0x | ctxzero | 24.01 mm | 84.56 mm | 0.776 | 0.095 |
| 8191 | 3.0x | **ctxest** | **4.63 mm** | **81.57 mm** | **0.915** | **0.010** |
| 3141 | 5.0x | ctxzero | 399.17 mm | 519.70 mm | 0.499 | 0.401 |
| 3141 | 5.0x | **ctxest** | **13.83 mm** | **110.91 mm** | **0.824** | **0.032** |
| 8191 | 5.0x | ctxzero | 509.76 mm | 562.06 mm | 0.417 | 0.464 |
| 8191 | 5.0x | **ctxest** | **15.58 mm** | **123.08 mm** | **0.830** | **0.053** |

Paired, `ctxzero -> ctxest`:

| mesh | payload | cut | lateral | contact | outcome |
|---|---:|---:|---:|---:|---|
| 3141 | 3.0x | +510.9 %* | +16.3 %* | +0.077* | joint improvement |
| 8191 | 3.0x | +501.6 %* | +3.6 % | +0.101* | Pareto (lateral maintained) |
| 3141 | 5.0x | +1996.3 %* | +253.3 %* | +0.348* | joint improvement |
| 8191 | 5.0x | +2152.4 %* | +265.0 %* | +0.442* | joint improvement |

**4 of 4 cells pass the pre-registered rule. Three are joint improvements**,
meaning both scored channels improve with intervals excluding zero.

## Why this is not selection-mesh luck

The held-out effects are **larger** than the selection mesh's, not smaller:
+1996 % and +2152 % on cut at 5.0x against 2718's +1508 %. Selection bias
inflates the mesh a result was chosen on; here the reserved meshes are stronger.

The contact gains, +0.077 to +0.442, sit two to nine times above the A/A floor
measured on this instrument (+0.036 to +0.048). Contact is the channel to trust:
it carries roughly 3 % run-to-run spread against cut height's 40 %, and it is
the same physics, since `cut_height_error` is `relu(h - reference)` and exactly
zero in contact.

This is also the comparison that closed the previous investigation. The
online-gradient campaign had a favourable rung on 2718 that reversed to -79.6 %
on 3141. This one does the opposite.

## What is NOT established

1. **One seed.** n = 1. The project's own bar is a median over five seeds, and
   that has not been run.
2. **Nominal payload fails.** At 1.0x on mesh 2718 the same arm loses lateral,
   45.99 -> 81.50 mm, because the estimator over-reads the nominal plant as
   1.41x. A degree-2 estimator reading 1.14x is fitted and queued but unscored.
   The claim is therefore restricted to loaded operation until that is settled.
2. **The online arm beats the privileged oracle** (+311.5 % cut, +220.8 %
   lateral at 5.0x on 2718) and that is unexplained. The estimator is accurate
   at 3.0x, 3.11x against 3.00x, so accuracy does not explain it. A beta ladder
   testing the EMA soft-start hypothesis is queued.

An unexplained result that favours the claim is a reason for more work, not
fewer caveats.
