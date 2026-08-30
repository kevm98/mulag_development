# Fine-tuning collapses the policy — and payload was not the cause

**Correction.** This project has been recording that training into a wide payload
range collapses the policy into a reward local optimum -- "lifting the cutter off
the ground" -- with ground contact 0.879 -> 0.242 and safety violations
0.014 -> 0.643 cited as evidence. That attribution is **wrong**. The collapse
happens with the plant completely nominal.

## The control

Same zero-context warm start (`prive_warmstart/model_299_e6.pt`, context columns
verified **exactly 0.0**), same stage, same seed, same 40 iterations, 256 envs.
One difference between arms: whether the payload varies at all.

| | Arm A — payload 1.0-1.5 | Arm B — **no payload variation** |
|---|---|---|
| ground_contact_fraction | 0.465 -> 0.316 | 0.695 -> **0.324** |
| cut_height_error_m | 0.0598 -> 0.1944 | 0.0177 -> **0.0334** |
| lateral_error_m | 0.0518 -> 1.4447 | 0.0256 -> **1.1229** |
| Episode_Reward/total | 13.29 -> 0.32 | 17.74 -> **0.75** |

Both collapse. The control has **no plant variation whatsoever**, so payload
cannot be the mechanism. Lateral error grows ~44x in 40 iterations on a nominal
plant.

Payload does move the starting point (Arm A begins at 0.465 ground contact vs
Arm B's 0.695, consistent with the measured dose-response), but the collapse
trajectory is the same with and without it.

## What this invalidates

- The "payload -> cutter lifting" local-optimum story, as a *causal* claim.
- The payload curriculum built to work around it (`payload_curriculum.sh`): it
  ramps the wrong variable.
- Any fine-tuned checkpoint produced with this resume path today, including
  `2026-08-29_11-46-17/model_997.pt`, which started at ground contact 0.211 and
  cut error 0.121 m -- already degraded before payload could act.

## What it does not touch

The payload dose-response in
[G1_PAYLOAD_HEADROOM_2026-08-29.md](G1_PAYLOAD_HEADROOM_2026-08-29.md) is
unaffected: that was measured by **evaluating a fixed checkpoint** at several
payloads, with no training involved, so a training-time defect cannot reach it.
The headroom (+96.8% cut error at 1.5x, +451% at 2.0x) stands.

## Cause: the learning rate

Three further arms, all nominal plant, same warm start, 40 iterations, differing
only in learning rate (arm C -- keeping the checkpoint's optimizer -- is not
available: a migrated checkpoint requires `--reset_optimizer_on_resume`).

| arm | learning rate | ground contact | cut_height_error_m | lateral_error_m | reward total |
|---|---|---|---|---|---|
| B | default (~1e-3) | 0.695 -> **0.324** | 0.0177 -> 0.0334 | 0.0256 -> **1.1229** | 17.74 -> **0.75** |
| D | **1e-4** | 0.695 -> **0.680** | 0.0177 -> **0.0022** | 0.0256 -> 0.1467 | 17.74 -> **8.60** |
| E | 1e-5 | 0.695 -> 0.637 | 0.0177 -> **0.0010** | 0.0256 -> 0.1853 | 17.74 -> 8.57 |

At 1e-4 and 1e-5 the policy is healthy: ground contact is preserved and
cut-height error **improves ~8x** rather than degrading. At the default rate it
is destroyed. A fresh Adam at ~1e-3 on a converged policy is a destructive
restart, and every fine-tune run in this project today used it.

**Fix: fine-tune at `--learning_rate 1e-4`.** 1e-5 is equivalent on these
metrics and can be used if a run still looks unstable over longer horizons.

Lateral still rises to ~0.15 m in all healthy arms; the warm start's own source
run shows the same (0.0237 -> 0.1889 over 400 iterations), so this is where the
training-time metric sits during fine-tuning and is not specific to the resume.

## Consequence

RMA Phase 1 is unblocked -- it is itself a fine-tune from the warm start, so it
could never have worked at the default rate. Relaunched with payload 1.0-1.5,
`--learning_rate 1e-4`, from the zero-context warm start.
