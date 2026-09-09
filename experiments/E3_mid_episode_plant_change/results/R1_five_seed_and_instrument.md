# E3 — method R1 five-seed table, and the instrument that had to be rebuilt first

Run 2026-08-31 22:00 → 2026-09-01 01:30 UTC on `isaaclab-l4`.
Pre-registration: `reports/PREREG_rebalanced_2026-08-31.md` (+ amendment 1).

## Headline

**Method R1 fails its pre-registered bar. 2 of 5 seeds pass; the median seed
does not.** The previously reported result for this pipeline is withdrawn as
non-revalidatable and has been rebuilt from scratch.

## 1. Why everything had to be rescored

Two independent defects in the artefact trail, both found before any seed number
was read:

1. **No pre-2026-08-31 report records its plant.** The keys
   `deployment_mass_bimodal`, `deployment_mass_step_scale`,
   `deployment_mass_step_time_s`, `payload_scale_range` and
   `deployment_mass_step_seed` are **absent** — not null — from every report
   written before that date, because `evaluate_gates.py` did not write them yet.
   The plant those runs used cannot be recovered from the artefacts.
2. **The measurement path changed mid-campaign.** Exactly four files changed at
   **2026-08-31 16:34:33 UTC**: `PPO_ROM/scripts/evaluate_gates.py`,
   `verge_task/terrain_env.py`, and their two `__pycache__` twins. That is after
   every prior report and before every subsequent one. **The repository is not
   under version control, so the diff is unrecoverable.**

Consequence: no report from before that moment is comparable with any report
after it. The nine affected reports (seeds 17, 101, the control) are preserved
in `outputs/rif_delay/archive_unrecorded_plant/` with a README, and every arm
was rescored on the declared plant with current code.

**The former headline — seed 17, +120.1% cut / +78.2% lateral — is withdrawn.**
Rescored properly, seed 17 is **+63.7% / +84.1%**. The gap between those two
numbers is the cost of the unrecorded change.

## 2. Instrument validation

Prereg amendment 1 re-based invariant 1: instead of comparing the null to a
stored constant from vanished code, compare the frozen base with **no residual**
(`base_R1`) against the same base with a **zeroed residual** (`alpha=0`), same
plant, same code, same night. An A/A repeat (`base_R1rep`, identical command,
separate process) measures the noise floor.

On the verdict statistic — paired median difference, bootstrap CI, n=528:

| comparison | cut | lateral |
|---|---|---|
| A/A `base_R1` vs `base_R1rep` | −1.7% | +0.5% |
| NULL `base_R1` vs `alpha=0` | −0.8% | +1.0% |
| NULL2 `base_R1rep` vs `alpha=0` | **+8.5%*** | +0.5% |

**Invariant 1 passes**: a zeroed residual is statistically indistinguishable
from the frozen base. But NULL2 — also identical by construction — came out
significant. **One false positive in three.**

Two bounds follow, and they constrain every number in this file:

* **Cut-height effects below ~10% are not resolvable** by a single pair of runs,
  however tight the CI. Lateral is much tighter (~1.5%).
* **Marginal medians are not effect sizes.** Between two identical runs the
  marginal median moved 6.386 → 7.361 mm (+15.3%) while the paired difference
  was −0.121 mm, CI straddling zero. Per-environment values are wildly unstable
  across processes — 483/528 envs differ by >10%, 138 by >100%, p99 of the
  absolute difference 497 mm — because 900 steps of contact dynamics amplify
  float nondeterminism. Only the paired difference survives.

## 3. The R1 five-seed table

400 iterations, `--alpha 0.5 --lr 3e-4 --num_envs 256`,
`--lateral_tracking_std_m 0.09 --lateral_tracking_weight 4.0`.
Plant: `MULAG_MASS_BIMODAL=1.0,3.0`, `MULAG_MASS_STEP_SCALE=3.0`,
`MULAG_MASS_STEP_TIME_S=20.0`, `MULAG_MASS_STEP_JITTER_S=8.0`,
`MULAG_PAYLOAD_RANGE=1.0,1.0`. Scored vs `base_R1`, 528 paired envs, meshes
1337/17/4242.

| seed | cut | lateral | paired cut (mm) | paired lateral (mm) | verdict |
|---|---|---|---|---|---|
| 17  | +63.7%* | +84.1%* | +2.541 [+1.759, +3.161] | +23.895 [+22.025, +25.245] | **PASS** |
| 23  | +39.6%* | +23.4%* | +1.546 [+1.085, +2.173] | +10.478 [+8.258, +12.435] | **PASS** |
| 101 | +12.5%  | +28.2%* | +0.601 [−0.019, +1.119] | +11.660 [+9.842, +13.442] | trade-off — cut n.s. *and* inside the noise floor |
| 42  | −23.1%* | −20.6%* | −1.525 [−1.953, −0.824] | −15.445 [−19.010, −11.539] | both worse |
| 7   | −86.9%* | −71.9%* | −54.228 [−74.533, −38.181] | −139.539 [−144.620, −134.336] | both worse |

`*` = bootstrap CI on the paired difference excludes zero.

**2 of 5 pass. Median seed FAILS. Pre-registered bar NOT met.**

Seeds 42 and 7 fail by margins far exceeding the ±10% noise floor, so the
negative result is not an instrument artefact.

### Verdict against the four conditions

| condition | status |
|---|---|
| 1. median of five improves both, CIs exclude zero | **NOT MET** (2/5, median fails) |
| 2. capacity-matched control does not account for the gain | not yet tested on the final method |
| 3. sham does not account for the gain | **not yet run** under any rebalanced objective |
| 4. holds on the protocol as declared, not a subset | held to — all five seeds reported |

## 4. Diagnosis

The failure is **variance across seeds**, not a uniform shortfall: +63.7% to
−86.9% on cut with nothing varied but the seed. Training curves are violently
non-monotone, and seed 7 **peaked at 14729 mean reward and collapsed to 6957**
by the final iteration — which is the checkpoint the protocol scores.

Mean reward by quartile:

| seed | 25% | 50% | 75% | end | peak |
|---|---|---|---|---|---|
| 17  | 10960 | 8635 | 14230 | 14210 | 14230 |
| 23  | 9492 | −918 | 14304 | 12087 | 15720 |
| 101 | 4650 | 11753 | 3598 | 11006 | 13733 |
| 42  | 8534 | 6617 | 8849 | 12603 | 13358 |
| 7   | 10790 | 13727 | 13378 | 6957 | 14729 |

Budget is not the lever — 1200 iterations was already measured to score worse
held out than 400. Per the brief's guide, instability across seeds is
optimisation. **Method R2 changes exactly one thing: `--lr` 3e-4 → 1e-4.**
See `reports/PREREG_R2_2026-09-01.md`, written before any R2 run started, which
also declares in advance the risk that a 3x lower lr at unchanged budget may
simply under-train.

## 5. Protocol changes made, and when

| when (UTC) | change | why |
|---|---|---|
| 08-31 22:04 | new pre-registration for R1 | rebalanced objective is a method change; prior prereg was for the old objective at 1200 iters |
| 08-31 23:10 | amendment 1: rebuild all arms post-16:34; re-base instrument check on `base_R1` | unrecorded plants + mid-campaign measurement-code change |
| 08-31 23:20 | sham seed-selection rule fixed (median seed by cut improvement) | so the sham arm cannot be chosen to suit the outcome |
| 09-01 01:35 | R2 pre-registered (lr 1e-4), all five seeds to be retrained | R1 median failed; diagnosis is optimisation instability |

No seed was dropped, re-rolled, or added at any point. The R1 table above was
produced entirely after the last protocol change that affects it.

## 6. Reproduce

```
PPO_AAM_FD/experiment_scripts/score_queue.sh outputs/jobs_seeds.txt   # arms
PPO_AAM_FD/experiment_scripts/base_r1.sh                              # base_R1
PPO_AAM_FD/experiment_scripts/base_r1_repeat.sh                       # A/A
python3 experiments/E3_mid_episode_plant_change/code/seed_table.py --base base_R1 \
  --arms 17=rebal17_history 101=rebal101_history 23=rebal23_history \
         42=rebal42_history 7=rebal7_history
```
