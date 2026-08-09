# PPO2 ROM-B remediation contract

Companion to `rom_b_root_cause.md`. Nothing here has been executed. No training
has been started.

Baseline being remediated:
`logs/rsl_rl/ppo2_rom_b_robust/2026-07-24_12-48-14/model_53_ppo2_rom_b_final.pt`
(`6731bac46e083d7ac99074e3a7b34752b64117a5003755e3d8296747353a69e9`) — preserved,
never overwritten, and never the target of a save path.

## 0. Target

The audit gate is `max` over rows of a 128-sample termination rate against `0.05`.
Section 3 of the root-cause report shows this requires a **true per-row rate of
about 0.010** to pass reliably. Current pooled rates:

| shape | now | target | dominant cause |
|---|---:|---:|---|
| line | 0.0376 | ≤ 0.010 | clearance |
| square | 0.0240 | ≤ 0.010 | head alignment |
| arc | 0.0191 | ≤ 0.010 | head alignment |
| sine | 0.0113 | ≤ 0.010 | truck raw reaction |
| circle | 0.0075 | ≤ 0.010 | truck raw reaction |

Line and square/arc are the whole job. Circle's `p95_acceleration` breaches are a
secondary target: centre the population at ≤ 33 m/s² so the tail clears 40.

## 1. Order of work

**Step 1 — diagnostics (do this first; it gates every number in step 2).**

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  PPO2_ROM_B_PhysX/scripts/diagnose.py \
  --headless --device cuda:0 \
  --num_envs 128 --eval_steps 120 --seed 17 \
  --suite monte_carlo --rounds 1,3,4,5 \
  --trajectory_shapes line,arc,square,circle
```

These are the Monte Carlo rows that actually failed or came closest
(line rounds 1/4 at 0.0547, round 5 at 0.0391; square round 3/5 at 0.0312;
arc round 4 at 0.0312). Seeds are reproduced exactly from the audit.

Then the two acceleration breaches, both from `circle` seed 400030 (round 3):

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  PPO2_ROM_B_PhysX/scripts/diagnose.py \
  --headless --device cuda:0 \
  --num_envs 128 --eval_steps 120 --seed 17 \
  --suite h_sweep --h_values 0.6,0.9,1.0 --rounds 3 \
  --trajectory_shapes circle,line
```

Runtime is roughly `rows x 1 min`; the two commands are ~16 and ~6 rows.

**Step 2 — set the shaping from what step 1 shows** (decision rules in section 2).

**Step 3 — verify cheaply before committing to a long run.** After editing
`env_cfg.py`, confirm the checkpoint is unchanged and the shaping is doing what
you intended, with no PPO updates:

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  PPO2_ROM_B_PhysX/scripts/train.py \
  --headless --device cuda:0 \
  --num_envs 128 --seed 17 \
  --start_curriculum_stage 4 \
  --resume_checkpoint /home/rrlab/rrlab_mulag/logs/rsl_rl/ppo2_rom_b_robust/2026-07-24_12-48-14/model_53_ppo2_rom_b_final.pt \
  --remediation --curriculum_eval_only \
  --curriculum_eval_seed_groups 3
```

This must reproduce `model_53`'s metrics on group 1 (seeds 100017–100021) exactly.
If it does not, a shaping edit leaked into the evaluator and must be found before
proceeding.

**Step 4 — remediation training** (section 3). **Step 5 — final validation**
(section 4).

## 2. Proposed changes, with the evidence each depends on

All are ROM-B scoped edits to `PPO2_ROM_B_PhysX/env_cfg.py`. None touches the
factor ranges, the ROM output path, the observation, the residual authority, the
`0.05` / `0.20 m` / `40 m/s²` limits, ROM-C, or AAM.

### 2.1 Clearance shaping onset — conditional

The brief suggests strengthening shaping inside `0.18–0.20 m`. The reward audit
argues against weight-only: the penalty is exactly flat below `0.180`, so there is
no gradient over the other 90 % of the range, and raising the weight inside a
`0.02 m` band mostly moves the equilibrium from `0.19` to `0.185`.

Decide from `path_profile.mean_clearance_error_m` and `p95_clearance_error_m` in
the step-1 line diagnostics:

* **If line's mean clearance is high (≳ 0.12 m)** the population genuinely rides
  near the limit. Widen the onset so shaping covers the operating range:
  `rom_b_clearance_soft_m: 0.180 -> 0.120`, `residual_ground_band_m: 0.180 -> 0.120`,
  `rom_b_clearance_margin_weight: -4.0 -> -6.0`. Hard limit stays `0.200`.
* **If line's mean clearance is low (≲ 0.06 m) with a long tail**, the failures are
  excursions, not a biased operating point. Keep the onset at `0.180` and instead
  steepen the tail: `rom_b_clearance_margin_weight: -4.0 -> -8.0`, plus a second
  onset at `0.150` with weight `-2.0` so the excursion is penalized before it is
  already at 90 % of the limit.

Either way the penalty stays bounded (`clamp(..., max=1.0)` squared) and the hard
event weight stays at `-20`.

### 2.2 Head-alignment shaping — conditional

The ROM-B head margin currently lives in a `0.005`-wide window
(`0.985 -> 0.980`) and the parent term contributes only `-0.125` at the limit.
`arc` and `square` sit at `0.980–0.981` on every seed and every `h`.

* **If cylinder-4 physical residual stays at ~`1e-6 m`** in the arc/square failure
  records — i.e. the wrist is inert and head attitude is effectively open-loop —
  widening the head shaping alone will not help, because the policy has no
  effective actuator for it. Report this and raise the question of cylinder-4
  authority separately; do **not** silently increase `cylinder4_residual_scale_m`.
* **If cylinder-4 is being used**, widen the onset:
  `rom_b_head_soft_alignment: 0.985 -> 0.992`, `residual_head_threshold: 0.985 -> 0.992`,
  `rom_b_head_margin_weight: -4.0 -> -6.0`. Hard limit stays `0.980`.

### 2.3 Acceleration shaping — conditional

Population `p95` is 30–38 against a `40` gate, with shaping starting at `35`.

* **If the circle path profile shows the `>40` samples concentrated in one or two
  path bins** (startup, phase wrap, reversal), that is a command-discontinuity
  problem, not a shaping problem. Fix the discontinuity; leave the weights alone.
* **If the excess is spread across the path**, lower the onset:
  `rom_b_acceleration_soft_mps2: 35.0 -> 32.0`, `rom_b_acceleration_margin_weight: -1.0 -> -2.0`.
  Hard limit stays `40.0`, event weight stays `-5.0`.

### 2.4 Event penalties — log before changing

Per the brief, do not adjust `rom_b_hard_safety_event_weight` or
`rom_b_acceleration_event_weight` until their firing frequency and reward
contribution are logged. The existing
`rom_b_safety/hard_violation_rate`, `rom_b_safety/acceleration_violation_rate` and
`rom_b_safety/mean_penalty` scalars in `env.py:_get_rewards` already provide this
in TensorBoard. The reward audit indicates the magnitudes are already dominant
(`-20` against a `+0.56` tracking reward at the operating point), so the expected
outcome is no change.

### 2.5 Train/evaluate cost-structure parity — recommended, now available as a flag

The one real parity gap (root cause §2.1): PPO collects with
`defer_safety_terminations=True`, so a clearance/head/limit crossing costs a
bounded per-step penalty and the episode continues, while the evaluator ends the
episode and counts a termination. The policy has never been trained against the
cost structure it is graded on.

`--train_safety_terminations` now sets `defer_safety_terminations=False` **and**
`synchronize_resets=False` for collection only. Unsynchronized resets are the
part that matters: the original stall (`policy/env.py:1222-1228`) was caused by
one replica's violation resetting all of them, not by per-replica termination.

Enable it in the remediation run. Watch iteration wall-clock in the first few
rounds; at ~2 % terminations per 120 scored steps the extra
preposition-plus-settle cost should be small, but if throughput collapses, fall
back to the default and record why.

### 2.6 Training allocation

Keep independent six-factor sampling at `randomization_scale=1.0`. Extra
allocation to failing shapes is already implemented and will target `line`,
`square` and `arc` automatically through `_failed_shapes` /
`--failed_shape_iteration_multiplier`. Do not train on `h`, do not add `h` to
observations, and do not replace the training distribution with the correlated
`h` curve.

## 3. Remediation training command

Do **not** run this until step 1 and step 3 above are complete.

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  PPO2_ROM_B_PhysX/scripts/train.py \
  --headless --device cuda:0 \
  --num_envs 128 \
  --seed 17 \
  --start_curriculum_stage 4 \
  --resume_checkpoint /home/rrlab/rrlab_mulag/logs/rsl_rl/ppo2_rom_b_robust/2026-07-24_12-48-14/model_53_ppo2_rom_b_final.pt \
  --remediation \
  --train_safety_terminations \
  --curriculum_eval_seed_groups 3 \
  --curriculum_eval_seed_offset 100000 \
  --curriculum_eval_steps 120 \
  --minimum_full_scale_rounds 8 \
  --balanced_shape_rounds 40 \
  --iterations_per_shape 2 \
  --failed_shape_iteration_multiplier 3 \
  --max_stage_rounds 40 \
  --learning_rate 5.0e-6 \
  --residual_scale_m 0.002 \
  --cylinder4_residual_scale_m 0.00025 \
  --path_speed_mps 0.18 \
  --save_interval 10
```

What each remediation-specific flag buys:

* `--remediation` loads `model_53`'s weights at stage 4 with **fresh optimizer
  moments, a fresh iteration counter, and curriculum counters at zero**. Without
  it, the plain `--resume_checkpoint` path restores that run's history, which ended
  at `consecutive_passes=5, complete=True` — the first passing round would declare
  completion again. It also refuses `--resume_run_dir`, so history cannot leak in.
* `--curriculum_eval_seed_groups 3` widens promotion from 5 rows to 15 fixed
  held-out rows per round. `model_53` was declared complete on 5 rows at
  `max_termination_rate=0.039`; on 25 audit rows the same policy reaches `0.0547`.
  Five rows cannot distinguish `p=0.01` from `p=0.037`.
* `--curriculum_eval_seed_offset 100000` keeps curriculum seeds
  (100017–100031) disjoint from the final-test families. The script now refuses
  offsets `300000` and `400000` outright.
* `--minimum_full_scale_rounds 8` (up from 5) requires more evidence before early
  stopping.
* `--max_stage_rounds 40` prevents a premature stall stop now that promotion is
  harder to satisfy.

Checkpoint discipline is unchanged and already correct: candidates are accepted
only if `_candidate_improves_or_preserves_acceptance` holds, otherwise the
previous accepted state is restored, so a tracking or safety regression rolls
back. The run writes to a fresh timestamped directory; `model_53` is opened
read-only.

## 4. Final validation

Predeclared, previously unused base seeds. With base `s`, Monte Carlo uses
`s+300000+...` and the h sweep `s+400000+...`. Base `17` consumed 300017–300041
and 400017–400031; the two below are disjoint from those and from each other, and
from the curriculum family (100017–100031).

* **Final seed A: `61`** → Monte Carlo 300061–300085, h sweep 400061–400075.
* **Final seed B: `149`** → Monte Carlo 300149–300173, h sweep 400149–400163.

Fixed now, before training. Neither may be used to select a checkpoint, and no
third seed may be introduced if these fail.

**4a. Diagnostic suite** (the 2026-07-24 seed family; now a regression check, not
a final test):

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  PPO2_ROM_B_PhysX/scripts/evaluate.py \
  --headless --device cuda:0 \
  --num_envs 128 \
  --monte_carlo_rounds 5 \
  --h_sweep_rounds 3 \
  --eval_steps 120 \
  --seed 17 \
  --checkpoint <REMEDIATED_CHECKPOINT>
```

**4b. Final evaluation, seed A:**

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  PPO2_ROM_B_PhysX/scripts/evaluate.py \
  --headless --device cuda:0 \
  --num_envs 128 \
  --monte_carlo_rounds 5 \
  --h_sweep_rounds 3 \
  --eval_steps 120 \
  --seed 61 \
  --checkpoint <REMEDIATED_CHECKPOINT>
```

**4c. Confirmation, seed B:** identical with `--seed 149`.

Acceptance, all of which must hold:

* Monte Carlo `PASS`.
* All nine `h` conditions `PASS`.
* Every individual gate `PASS` — mean error `≤ 0.040`, p95 error `≤ 0.080`,
  p95 acceleration `≤ 40.0`, head alignment `≥ 0.980`, clearance `≤ 0.200`,
  truck drift `≤ 0.001`, termination `≤ 0.05`.
* No regression at `h = 1.0` against `model_53`'s
  (mean 0.0373, p95 0.0687, acc95 35.50, term 0.0391).
* The checkpoint was selected before 4b/4c ran.

### 4.1 A note on the acceptance rule's own precision

Not a proposed change, and not a reason to relax anything — but it should be on
the record. At `n=128`, `max`-over-25-rows against `0.05` fails a policy whose true
rate is `0.02` about 31 % of the time. Raising `--num_envs` to `512` shrinks the
binomial noise by 2× without touching the `0.05` threshold, without pooling, and
without dropping worst-shape acceptance — a true `0.02` policy would then pass
essentially always and a true `0.05` policy would still fail. If the remediated
policy lands near `0.015–0.020` and the 128-environment suite reports scattered,
non-reproducible failures, that is the measurement, not the policy, and increasing
`n` is the correct response. Decide this **before** running 4b, not after seeing
its result.

## 5. What must not change, and has not

* `0.05` termination gate, `0.20 m` clearance limit, `40 m/s²` acceleration limit —
  untouched in `runtime/curriculum.py`.
* Six-factor order, ranges (`e_comp [-0.125, 0.020]`, `e_flow [-0.20, 0.20]`,
  `d_seal_j [0, 1]`), independent per-environment uniform sampling, and the
  `health_scale` mapping — untouched in `runtime/rom_b_runtime.py`.
* ROM outputs, pressure, velocity, extension and PPO actions are not multiplied by
  `h`; `h` remains an evaluation-only coordinate and is not in the observation.
* Factors stay private: `predict_delta` and `step` still raise if handed factors,
  and the actor/critic remain factor-blind on 42 observations.
* Residual authority unchanged (`0.002 m`, cylinder-4 `0.00025 m`).
* No new `e_pressure`, valve-delay, mechanical-clearance or pressure-limit
  randomization.
* `model_53` is not overwritten; remediation does not start from `model_607`.
* ROM-C and AAM untouched; no AAM data regenerated.
