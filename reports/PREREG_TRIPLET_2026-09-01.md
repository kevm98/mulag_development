# Pre-registration — multi-seed adaptation efficacy (the triplet experiment)

Fixed **2026-09-01 21:53 UTC** (23:53 CEST), after the four capacity-matched
control arms were LAUNCHED for training (21:37 UTC) and **before any of them was
scored against any held-out mesh**.

This experiment supersedes and merges two separately planned tests — the seed
replication and the zero-history control — into one design. They are not run
separately, because run separately neither can answer the other's question.

## The question

> **Does access to live history consistently improve control beyond what extra
> model capacity alone provides?**

Not "does the residual help" (that confounds history with parameters) and not
"does one seed help" (that confounds signal with seed luck). One design, both
confounds closed.

## Design: five triplets

For each of the five seeds **17, 23, 101, 42, 7**, three arms:

| arm | residual | history block | parameters |
|---|---|---|---|
| **base** | none (`--residual_alpha 0.0` / no residual) | — | frozen base only |
| **control** | `rebal<seed>_nohistory` | **zeroed** | identical to adaptive |
| **adaptive** | `rebal<seed>_history` | live | identical to control |

`--no_history` zeroes the history block while keeping the input width
(`train_residual_adapter.py:133`), so the control and the adaptive arm have the
**same architecture and the same parameter count**. The only difference is
whether the history the network reads is the real one or zeros.

Held identical within every triplet: frozen base
(`distilled_v1/phase1_init_70.pt`), architecture, reward
(`--lateral_tracking_std_m 0.09 --lateral_tracking_weight 4.0`), budget
(`--num_envs 256 --iterations 400`), residual authority (`--alpha 0.5`),
optimiser (`--lr 3e-4`), `--history 20 --latent_dim 8`, training seed, plant
change (`MULAG_MASS_BIMODAL=1.0,3.0`, `MULAG_MASS_STEP_SCALE=3.0`,
`MULAG_MASS_STEP_TIME_S=20.0`, `MULAG_MASS_STEP_JITTER_S=8.0`), and the
evaluation environments (`--num_envs 176 --steps 900 --seed 17`, paired).

The base arm is seed-independent by construction — it is the same frozen policy
in every triplet — so `base_R1` is scored once and reused. This is stated so no
reader mistakes it for five separate base runs.

## Checkpoint protocol

**Primary: the R3 rule** — each arm is scored at the saved checkpoint nearest
the peak of its own logged training-reward curve, selected by the same parser
for every arm, control and adaptive alike. Training reward is measured on the
training plants; **the held-out meshes are never consulted in the selection.**

R3 is primary because it was declared the method under test on 2026-09-01 09:55
UTC, *before* its screen was scored — not because of how its numbers came out.
`PREREG_R3_2026-09-01.md` step 2 already commits to completing meshes 909 and
7001 "for all five seeds **and** the control", which is exactly this table.

**Secondary (robustness): `model_399`**, the final checkpoint, no selection step
at all. Reported on the three original meshes. If the attribution conclusion
flips between the two checkpoint rules, that is reported as the headline, not
buried.

**Rule-consistency note, declared here.** Re-deriving the R3 peaks from the
training logs reproduces every recorded peak reward exactly (14229.5, 13732.6,
15720.1, 13357.6, 14728.6) but attributes them 2–12 iterations later. For four
seeds the chosen checkpoint is unchanged. For **seed 101** it moves the
selection from `model_225` to `model_250`. The uniform re-derivation
(`model_250`) is used, so that one parser selects every arm in the table; the
`model_225` numbers already on disk are reported alongside as a sensitivity.

## Success criterion — CHANGED from R1/R2/R3, and why

R1–R3 required **both** metrics to improve. That bar is retained as the
strictest column but it is no longer the pass condition, because it treats "no
measurable change" as a failure even when the change is smaller than the
instrument can resolve.

> **An arm SUCCEEDS if one primary tracking metric improves significantly while
> the other either improves or stays within its established non-inferiority
> bound.**

The non-inferiority bounds are **not chosen here** — they are the noise floors
already measured by A/A test and recorded in `PREREG_R2_2026-09-01.md`:

| metric | non-inferiority bound | source |
|---|---|---|
| cut height | **10%** | A/A + two null comparisons; one false positive at +8.5% in three |
| lateral | **1.5%** | same A/A series |

Outcome labels used in the table:

* **joint improvement** — both metrics improve, CI excluding zero (the old bar)
* **Pareto improvement** — one improves significantly, the other is inside its
  bound (no measurable cost)
* **trade-off** — one improves, the other degrades beyond its bound → FAILS
* **both worse** → FAILS

Significance is the repo statistic throughout: paired per-environment
differences, median, bootstrap CI (B=20000) excluding zero.

## What is and is not blinded — stated plainly

**Not blinded.** The base-vs-adaptive axis is already known for all five seeds
at both checkpoint rules; those numbers were read before this document was
written and are reproduced in it. This experiment does not re-test that axis and
does not claim to.

**Blinded.** The **control arms for seeds 23, 42, 101 and 7 do not yet exist** —
they were launched into training 16 minutes before this file was fixed and none
has been scored. The primary question of this experiment is the
control-vs-adaptive axis, and for four of the five seeds it is being asked of
data that does not exist yet. Only seed 17 has a prior control arm.

This is why the merged design is worth running and the two separate tests were
not: the attribution axis is genuinely untested at four of five seeds.

## Multiplicity

The base-vs-adaptive axis has now been looked at four times against the same
held-out meshes (R1, R2's abandoned screen, R3, and this table). It is reported
with that caveat and is **not** the claim this experiment makes.

The control-vs-adaptive axis is a **first look** at seeds 23, 42, 101 and 7. It
carries no multiplicity penalty and it is the primary result.

## Predictions, fixed in advance

1. The control does **not** beat the base by a meaningful margin on either
   metric. Extra capacity with zeroed history buys nothing. (Seed 17 already
   shows this: cut −0.5%, lateral −3.3% at n=880.)
2. The adaptive arm beats its own control on at least three of five seeds under
   the success criterion above.
3. If prediction 1 fails — if the control tracks the adaptive arm — then the
   advantage is capacity, not adaptation, and the thesis claim fails. That is
   the falsification condition and it is declared here.

## Reporting

All five triplets are reported with their outcome label, whatever it is. No seed
is dropped, re-rolled or added, and the median-of-five rule decides the claim.
If the median seed fails, the reported result is the negative.
