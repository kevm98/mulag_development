# A-RMA phase 3 fixes nominal by destroying the loaded gain

`PREREG_ADAPTATION_2026-09-06.md` declared, before this ran:

> **Success** is the nominal harm removed -- `ctxzero -> ctxest` at 1.0x no
> longer significantly negative on lateral -- **while 3.0x and 5.0x still pass**.
> A fix that rescues nominal by destroying the loaded gain is not a fix, and
> both are scored to make that visible.

That is exactly what happened, and the criterion caught it.

## The retrain

Same 1024 context parameters, retrained with `MULAG_CONTEXT_ABLATE=estimated`
and the fitted estimator in the loop, so the policy sees the distribution it is
deployed on. The retrain worked and produced a materially different policy:

| checkpoint | task columns | mass_scale weight |
|---|---:|---:|
| `ctxonly17` (trained on TRUE context) | 0.125463 | 0.405202 |
| `arma3` (trained on ESTIMATE) | 0.125463 | **0.931542** |

2.3x more weight on the payload channel, task columns still bit-identical,
`alpha`/`tau`/`delay` still exactly zero. Consistent with the policy learning to
compensate for the estimator's known shrinkage.

## Result, paired `ctxzero -> ctxest`

| payload | config | cut | lateral | contact | outcome |
|---|---|---:|---:|---:|---|
| 1.0x | run1 | +11.3 %* | **-43.9 %*** | +0.008* | fail |
| 1.0x | arma3 | -43.3 %* | **+2.5 %** | -0.047* | fail |
| 3.0x | run1 | +74.6 %* | +11.5 % | +0.039* | Pareto |
| 3.0x | arma3 | **-70.5 %*** | +21.7 %* | **-0.241*** | fail |
| 5.0x | run1 | +1508.0 %* | +216.3 %* | +0.241* | joint |
| 5.0x | arma3 | **-57.8 %*** | +235.6 %* | **-0.315*** | fail |

**The nominal lateral harm IS removed**: -43.9 %* becomes +2.5 %, non-significant
and indistinguishable from the frozen base. The literature's diagnosis was right
about the cause -- the policy had never seen an estimate during training.

**But the loaded gain is destroyed.** At 3.0x cut collapses to -70.5 %* and
ground contact falls by 0.241, which is the mechanism the entire result rests
on: cut-height error on this plant is almost entirely contact loss
(corr = -0.975). A-RMA did not repair the arm, it moved the policy to a
different and worse operating point that happens to favour lateral everywhere.

## Verdict, and the end of the patching

**A-RMA phase 3 is not adopted.** `ctxonly17` (run 1) remains the configuration.

Four fixes for the nominal harm have now been tried and each has failed:

| attempt | rationale | outcome |
|---|---|---|
| degree-2 estimator | reduce the 1.41x over-read to 1.14x | 81.50 -> 79.65 mm, still significant |
| hard deadband | suppress the estimate near nominal | 81.50 -> 89.92 mm, worse, and costs 5.0x |
| `beta = 0.005` | slower ramp, the ladder's own prescription | 81.50 -> 74.02 mm, still significant, and -78 % at 5.0x |
| A-RMA phase 3 | train on the deployed distribution | nominal fixed, loaded gain destroyed |

Per the pre-registration, this ends the patching. The claim is scoped in writing:

> Payload-conditioned online adaptation delivers large joint improvements at
> heavy and time-varying payload, and **degrades lateral tracking at nominal
> payload**. It should be gated off below a known payload threshold.

That gating is operationally reasonable, since the machine's load state is known
in practice, but it is a limitation of the method and is reported as one rather
than presented as a design choice.

## The structural reading, which unifies three results

A-RMA improves lateral at EVERY payload (+2.5 %, +21.7 %*, +235.6 %*) and
destroys cut and contact at EVERY payload (-43.3 %*, -70.5 %*, -57.8 %*;
contact -0.047, -0.241, -0.315). It did not repair a defect. It moved the policy
to the opposite end of a trade-off.

That unifies three otherwise separate observations:

* **run 1**, trained on the true context, lands at the cut/contact end.
* **arma3**, trained on the noisy estimate, lands at the lateral end.
* **run 3**, which multiplied the lateral price eightfold, moved the operating
  point without ever buying lateral back
  (`RESULT_run3_pricing_refuted_2026-09-06.md`).

The common explanation is that **with only the four context columns trainable,
the reachable behaviours form a roughly one-dimensional family in which cut and
contact trade against lateral**. 1024 parameters entering one layer cannot
restructure the controller; they can only slide it along whatever curve those
columns parameterise. Different training signals select different points on that
curve and none of them bends it.

This is why four nominal fixes all traded rather than fixed, and it converts the
nominal limitation from an unexplained defect into a structural property of the
method with a stated cause.

**What it implies for escaping the trade-off.** More adaptation capacity than
four input columns -- for example modulating hidden activations rather than the
input layer, which can express a richer family -- or rebalancing the objective
during the original distillation rather than afterwards, when the base policy's
own trade-off is already fixed. Neither is tested here and neither is claimed.

## What is unaffected

The loaded-to-loaded step test and the freeze ablation both bear on the LOADED
claim, which is the one that survives, and neither uses the arma3 checkpoint.
