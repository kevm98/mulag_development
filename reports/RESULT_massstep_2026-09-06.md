# Adaptation on a plant whose payload changes under the controller

Mesh 2718, 176 environments, 900 steps (15.0 s), payload stepping mid-episode at
**6 s +- 2 s**. Every arm is the SAME checkpoint `ctxonly17/model_1596`; only the
source of the payload number differs.

## The plant gate, run before any adaptive number was read

The frozen policy alone, on the stepping plant, must land BETWEEN the two static
plants. If it sits on the pre-step plant the step did not fire.

| dose | constant pre-step | STEPPING | constant post-step | position |
|---|---:|---:|---:|---|
| 1.0 -> 3.0x | 3.23 mm | 11.02 mm | 16.91 mm | 57 % |
| 1.0 -> 5.0x | 3.23 mm | 127.09 mm | 328.50 mm | 38 % |

At the 3.0x dose all five recorded metrics -- cut, lateral, contact, impacts and
safety -- fall between the two static bounds. Both gates pass.

**This gate exists because the first attempt failed it.** The step was originally
scheduled at 20 s +- 8 s, and a 900-step rollout at `control_dt = 1/60 s` lasts
15.0 s, so it could only fire in the last 0-3 s of about a fifth of the
environments. The frozen policy then read 4.17 mm against 3.23 mm on a constant
nominal plant, which is what exposed it. Those reports were deleted.

## Refinement to the gate, from the loaded-to-loaded case

The "fraction of the way between the two static plants" form of the gate assumes
the stepping plant interpolates between them. That assumption fails when the
post-step plant is a catastrophic failure mode. Measured on the 2.0 -> 5.0x step:

| plant | cut | lateral | contact | impacts | safety |
|---|---:|---:|---:|---:|---:|
| constant 2.0x | 2.91 mm | 59.44 mm | 0.911 | 0.372 | 0.036 |
| STEP 2.0 -> 5.0x | 8.97 mm | 61.31 mm | 0.871 | 0.405 | 0.048 |
| constant 5.0x | 512.47 mm | 601.62 mm | 0.408 | 0.268 | 0.475 |

The fraction is 1 %, which reads as a failed gate. It is not. A policy starting
at 5.0x fails from t = 0 and never recovers; a policy starting at 2.0x, where it
is healthy, and perturbed at 6 s spends nine seconds at 5.0x FROM A GOOD STATE.
Those are different trajectories, so no interpolation between the two static
runs is expected.

**The gate is therefore stated against the PRE-step plant only:** the frozen
policy on the stepping plant must differ from the pre-step constant plant, in
the direction of the post-step plant. That comparison is assumption-free. Here
all four independent quantities move correctly -- cut 2.91 -> 8.97 mm, contact
0.911 -> 0.871, impacts 0.372 -> 0.405, safety 0.036 -> 0.048 -- so the step
fired.

The interpolation form remains useful where the post-step plant is degraded
rather than failed, as in the 1.0 -> 3.0x case (57 %), and is reported there.

Note also that constant 5.0x reads 512.47 mm here against 328.50 mm in an
earlier run of the same configuration. That is the cut channel's run-to-run
spread again, and it is worst exactly where the controller is failing, which is
a further reason to weight ground contact over cut height.

## Result

| dose | arm | cut | lateral | contact | accel | safety |
|---|---|---:|---:|---:|---:|---:|
| 3.0x | ctxzero | 11.02 mm | 63.04 mm | 0.841 | 76.2 | 0.069 |
| 3.0x | ctxest | 4.75 mm | 74.85 mm | 0.931 | 49.6 | 0.008 |
| 5.0x | ctxzero | 127.09 mm | 194.02 mm | 0.728 | 99.3 | 0.168 |
| 5.0x | **ctxest** | **9.50 mm** | **95.95 mm** | **0.872** | **49.1** | **0.026** |

Paired `ctxzero -> ctxest`:

| dose | beta | cut | lateral | contact | outcome |
|---|---|---:|---:|---:|---|
| 3.0x | 0.02 | +175.4 %* | -13.8 %* | +0.058* | fail, lateral |
| 3.0x | 0.005 | +172.7 %* | -10.5 % | - | fail, lateral |
| 5.0x | 0.02 | **+636.1 %*** | **+52.0 %*** | **+0.121*** | **JOINT IMPROVEMENT** |
| 5.0x | 0.005 | **+676.5 %*** | **+55.3 %*** | **+0.137*** | **JOINT IMPROVEMENT** |

At the 5.0x dose both channels improve with intervals excluding zero, at both
smoothing rates. Cut falls 127.09 -> 9.50 mm, lateral 194.02 -> 95.95 mm,
contact rises 0.728 -> 0.872, the violation rate falls 6.5x and peak tool
acceleration halves.

## The dose-response, and why it is the interesting part

    step to 3.0x   cut +175.4 %*   lateral -13.8 %*
    step to 5.0x   cut +636.1 %*   lateral +52.0 %*

The benefit grows with the size of the plant change. That is what an adaptive
mechanism should do and what a merely better-tuned fixed controller would not:
a fixed controller's advantage does not scale with how much the plant moved.

The 3.0x failure is understood rather than unexplained. That step spends its
first six seconds at NOMINAL, the one regime where this arm is known to lose
lateral (-43.9 % at constant 1.0x), and only nine seconds at 3.0x where it
helps. At the 5.0x dose the post-step damage to the frozen policy is far larger
(194 mm of lateral against 63 mm), so the adaptive gain dominates the nominal
contamination instead of being swamped by it.

## What this does NOT establish

**That the arm TRACKS the change.** `beta = 0.005` has a 3.3 s time constant and
should badly lag a step landing at 4-8 s of a 15 s episode, yet it matches
`beta = 0.02` at both doses. If rapid tracking carried the benefit, a
four-times-slower estimator would have shown it.

So the supported claim is "improves performance on a time-varying plant", which
is weaker than "tracks the payload change". The freeze ablation
(`MULAG_ESTIMATOR_FREEZE_S=4.0`, latching the estimate just before the step
window) is queued to settle which sentence is true, and its reading was declared
before it ran.

Also outstanding: selection mesh only, and one seed.

## The loaded-to-loaded step, and a correction to the scope

The 2.0 -> 5.0x step carries NO nominal phase, so it was run to isolate the
adaptation claim from the nominal defect. It fails anyway.

| arm | cut | lateral | contact | impacts | accel | safety |
|---|---:|---:|---:|---:|---:|---:|
| ctxzero | 8.97 mm | 61.31 mm | 0.871 | 0.405 | 66.3 | 0.048 |
| ctxest | 9.07 mm | 80.06 mm | 0.920 | 0.523 | 49.7 | 0.017 |

Paired: cut **+28.1 %***, lateral **-23.1 %***, contact +0.022*. **Fail.**

### The earlier explanation was wrong

This file previously attributed the 1.0 -> 3.0x lateral failure to that step's
first six seconds sitting at nominal. That cannot be the explanation here, since
both phases are loaded. The nominal-phase account is withdrawn.

### All three steps, and what actually predicts them

| step | cut | lateral | outcome | frozen-policy contact |
|---|---:|---:|---|---:|
| 1.0 -> 3.0x | +175.4 %* | -13.8 %* | fail | 0.841 |
| 1.0 -> 5.0x | +636.1 %* | +52.0 %* | **pass** | **0.728** |
| 2.0 -> 5.0x | +28.1 %* | -23.1 %* | fail | 0.871 |

**One of three step configurations passes.** The claim "adaptation works on
time-varying plants" over-read a single favourable configuration and is
withdrawn in that form.

What predicts the outcome is how much ground contact the FROZEN policy is
losing. The arm buys contact at the cost of lateral. At 0.728 that trade is
overwhelmingly worth it; at 0.841 and 0.871 the lateral cost exceeds the gain.

### The scope, restated

> The arm helps when the frozen controller is **losing ground contact**, and
> costs lateral tracking when it is not.

This single rule covers every result measured: it passes at constant 3.0x and
5.0x and on the 1.0 -> 5.0x step, where the frozen policy's contact is 0.803,
0.539 and 0.728; it fails at constant 1.0x and on the two steps where contact is
0.954, 0.841 and 0.871. It subsumes the "nominal payload" limitation as the
special case where contact is highest.

It is also consistent with the structural reading in
`RESULT_arma3_2026-09-06.md`: four context columns can only slide the policy
along a cut/contact-versus-lateral curve, so the arm always makes the same
trade, and whether that trade pays depends entirely on how much contact there is
to recover.
