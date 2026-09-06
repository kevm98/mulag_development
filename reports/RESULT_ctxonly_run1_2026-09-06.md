# 1024 parameters of payload conditioning: cut, contact and safety win; lateral breaks

Context-only campaign run 1 (`ctxonly17`), scored 2026-09-06 on mesh 2718,
payload 3.0x, 176 environments, 900 steps.

## The arms

All three are the SAME 70-wide policy. Only the payload number changes.

| arm | cut | lateral | contact | impacts | p95 accel | safety |
|---|---:|---:|---:|---:|---:|---:|
| frozen base | 23.56 mm | 79.48 mm | 0.783 | 0.413 | 78.1 | 0.110 |
| `ctxzero` | 16.91 mm | 93.06 mm | 0.803 | 0.460 | 90.1 | 0.098 |
| `ctxlive` | **3.66 mm** | **509.40 mm** | **0.936** | 0.663 | **61.7** | **0.007** |

`ctxzero` is the trained checkpoint with its context zeroed. It computes
`base(x) + W_ctx . 0 = base(x)`, so it IS the frozen policy and is the matched
control. It is used as the reference rather than the separate `base` run,
because that route carries the 40 % cut-height instrument spread measured in
`RESULT_cut_instrument_floor_2026-09-06.md`.

Primary, paired per environment, B = 20000:

    ctxzero -> ctxlive     cut +408.6 %*     lateral -63.7 %*

## What was trained

`--train_context_only`: every actor parameter frozen, the input layer's
gradient masked to the context columns. **1024 trainable parameters against
59,588**, one 58th of the actor. Verified across the run: task columns stayed
bit-identical at mean |w| = 0.125463, while `mass_scale` went
0.000000 -> 0.169 (iter 650) -> 0.405202 (iter 1596), and the three constant
context axes stayed at exactly 0.000000 throughout.

At the end the payload column is 3.2x the mean task-column weight. It is the
most heavily weighted input dimension in the policy.

## The mechanism, and why the cut number is not a metric artefact

`cut_height_error` is `relu(h - reference)` and exactly zero in contact, so it
is nonzero only while the cutter is airborne. The cut improvement is therefore
mostly the contact improvement: 0.803 -> 0.936, close to the 0.96 the frozen
base manages at NOMINAL payload.

That could have been bought by burying the cutter, so the guard matters:

* worst-family ground contact rose, it did not fall
* the safety-violation rate FELL 0.098 -> 0.007, a 14x reduction
* p95 tool acceleration FELL 90.1 -> 61.7 m/s2

Hard impacts rose 0.460 -> 0.663 while accelerations fell, which is the
signature of riding the ground rather than striking it. Combined with the
falling violation rate this is healthier contact, not harder contact.

This is the root cause of `RESULT_rootcause_contact_2026-09-06.md` being
addressed: the heavy-payload failure was impact and rebound with command
authority to spare, and a policy told what it is carrying keeps its cutter on
the ground.

## Why lateral collapsed: an objective bug, not the mechanism

Training-time lateral error ran **0.18-0.55 m** against a Gaussian kernel of
**sigma = 0.09 m**. `exp(-(e/sigma)^2)` is then between 1e-7 and 1e-11, and the
gradient `2*w*e/sigma^2 * exp(-(e/sigma)^2)` with it. Measured:
`Episode_Reward/lateral` ran 4.3-10.4 against a possible ~3600 at weight 4.0
over 900 steps, i.e. **0.2 % of maximum**.

Lateral supplied no learning signal for essentially the whole run while cut and
contact did. 1024 parameters optimised the axes that had gradient, which is
correct behaviour under a mis-specified objective.

`PPO_AAM_FD/memory.md` records the same error inverted. sigma 0.300 was once
"ten times wider than the error it was meant to correct" at 61 mm, and the fix
to 0.09 was calibrated at that scale. On the 1-5x payload range the errors are
200-550 mm, so 0.09 is now roughly three times too narrow. The lesson is that
this sigma is regime-dependent and must be set from the error scale the run
actually visits, not inherited.

## All three payloads

Primary is `ctxzero -> ctxlive`, paired per environment, B = 20000.

| payload | cut | lateral | verdict |
|---|---:|---:|---|
| 1.0x | +6.9 %, ns | -1.7 %, ns | no change |
| 3.0x | +408.6 %* | -63.7 %* | fail, lateral |
| 5.0x | +166.8 %* | +0.2 %, ns | **pass** |

`ctxzero -> ctxlive` at 1.0x moves NEITHER channel significantly. Payload
conditioning leaves the nominal plant alone, which is the required behaviour and
the thing a fixed task-space offset could not express. At 1.0x the A/A row reads
-40.6 %* on cut between two runs of the same policy, independently reproducing
the instrument floor.

Payload 5.0x, the arms in full:

| arm | cut | lateral | contact | p95 accel | safety |
|---|---:|---:|---:|---:|---:|
| `ctxzero` | 328.50 mm | 455.85 mm | 0.539 | 173.1 | 0.363 |
| `ctxlive` | 45.97 mm | 385.95 mm | 0.634 | 75.1 | 0.299 |

Every axis improves, and the A/A row at this payload is non-significant on both
channels, so the effect is above the instrument.

### The caveat on the 5.0x pass, which matters

Lateral there is 455.85 -> 385.95 mm. "Maintained" is true under the rule, but
**both arms are failing to track the line**: neither is a working line-tracker
at 5x, so the pass is earned in a regime where lateral is already lost for
everyone. The cut, contact, acceleration and safety gains are real and large;
the claim "the controller works at 5x payload" is NOT supported by them.

This makes run 2 more important than the 5.0x pass. The question worth answering
is whether conditioning can hold the line AND keep the cutter down once lateral
has gradient, and that is what sigma = 0.25 tests.

## Verdict

**Fails the pre-registered rule.** One channel improves with its interval
excluding zero; the other is far outside its 1.5 % non-inferiority bound. It is
recorded as a failure and not re-scored.

What it establishes, which no earlier campaign did:

1. Payload conditioning of a frozen policy produces a **large, real** effect on
   cut height, ground contact and safety, all far beyond the measured
   instrument floor.
2. The effect comes from 1024 parameters that cannot restructure the policy,
   with the control provable rather than trained.
3. The failure axis is explained by a measured objective defect with a
   one-flag fix, rather than being attributed to the mechanism.

Run 2 (`ctxonly17w`) repeats it with `--lateral_tracking_std_m 0.25`, which puts
the gradient peak at 0.18 m, inside the range the policy occupies. Nothing else
changes.
