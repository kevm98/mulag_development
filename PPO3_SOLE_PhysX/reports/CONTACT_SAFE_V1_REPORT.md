# Contact-safe v1 — PPO3 (ROM-A) and PPO4 (ID/FD)

Zero-contact-loss work on branch `fix/ppo3-ppo4-zero-contact-loss`.

Every number below is read from a stored artifact under
`reports/contact_safe_v1/`; the file is named next to it. Where a claim was
made and then disproved, both are kept.

**Status of this document.** Sections 1–7 are complete and measured. Section 8
(training) and Section 9 (final gates) are filled in as the runs land; anything
not yet measured says so explicitly rather than being predicted.

---

## 0. Root cause, in one paragraph

The cutter is not sliding off the verge — it is being **thrown off it**. The
reference integrator `cylinder_target_max_lead_m` is saturated at its inherited
25 mm bound on **both** actuators. With the cylinders at ~5e6 N/m, contact
force *is* commanded position, so 25 mm of stored command is ~125 kN of
commanded force; measured peaks are 119 kN (ROM-A) and 170 kN (ID/FD). The
integrator winds up against terrain the cylinders cannot follow, releases the
stored command as an impulse, the tool is ejected upward, the integrator
re-winds during the flight, and the cycle repeats. Three things let this
survive every previous campaign: the binary contact reward pays a 119 kN
impact exactly what it pays 6 kN of stable support; the policy's force
observation clips at 10 kN so it cannot see any of the range it operates in;
and `ground_contact_fraction` is a duty cycle that scores 0.9486 — a PASS — on
a policy measured 45 consecutive steps and 297 mm off the ground.

---

## 1. Reproduced failure

Full detail, including the mechanism attribution and a step-by-step event
trace, is in
[`contact_safe_v1/PREFIX_DIAGNOSTIC.md`](contact_safe_v1/PREFIX_DIAGNOSTIC.md).
Summary, held-out mesh, stage 5, `s_curve`, relief 1.0, 960 steps × 32 envs:

| | PPO3 / ROM-A | PPO4 / ID-FD |
|---|---:|---:|
| `ground_contact_fraction` (gate ≥ 0.90) | **0.9486 PASS** | 0.8945 FAIL |
| contact-loss events | 428 | 458 |
| visible lifts (> 5 mm geometric) | 334 | 359 |
| impact/ejection events | 201 | 212 |
| hard impacts (> 30 kN) | 286 | 278 |
| max consecutive airborne steps | 45 (0.75 s) | 79 (1.32 s) |
| max airborne gap | 0.297 m | 0.675 m |
| upward separations | 73.4 % | 73.6 % |
| max upward cutter velocity | 4.41 m/s | 7.93 m/s |
| max contact force | 119 kN | 170 kN |
| max cylinder target lead | 0.0300 m | 0.0406 m |

Across three distinct held-out meshes (`PREFIX_SUMMARY.md`):

| | PPO3 events / lifts / max gap | PPO4 events / lifts / max gap |
|---|---|---|
| mesh 1337 (the pinned gate mesh) | 428 / 334 / 0.297 m | 458 / 359 / 0.675 m |
| mesh 909 | 408 / 289 / 0.292 m | 392 / 311 / 0.823 m |
| mesh 4242 | 398 / 296 / 0.303 m | *(see artifact)* |

The failure is a property of the policy and the plant, not of one mesh.

### 1.1 Mechanism attribution

| mechanism | PPO3 | PPO4 |
|---|---:|---:|
| impact / ejection | 804 | 848 |
| actuator lag | 776 | 820 |
| path geometry | 132 | 164 |
| **invalid sensing** | **0** | **0** |
| **cylinder / workspace saturation** | **0** | **0** |

Two of the five candidate mechanisms are eliminated by measurement, not by
argument: minimum scan validity 0.889, limit shield active on 0.00–0.05 % of
samples, minimum cylinder headroom 15 mm. Impact/ejection and actuator lag
split the remainder almost evenly because they are the same loop measured
before and after the impact.

### 1.2 A methodological finding that invalidates past multi-seed claims

`--seed` **cannot vary a held-out evaluation.** The held-out mesh is pinned at
seed 1337, `randomize_start_phase` is False, and the actor runs at its mean, so
the rollout is deterministic. Seeds 17 / 909 / 1337 / 4242 returned
**bit-identical** reports; those four runs are preserved in
`contact_safe_v1/prefix_identical_seed_artifact/` as the evidence.
`terrain_eval_seed_override` (`--eval_terrain_seed`) is new and is the only way
to move the held-out mesh. Passing `--eval_terrain_seed 1337` reproduces the
pinned mesh exactly, which is the control proving the override is correct.

---

## 2. Files changed

| file | what changed |
|---|---|
| `terrain_env.py` | contact-support state machine, `contact_safety_terminal`, contact observation channels, synchronized contact termination, `terrain_eval_seed_override` in `_mow_strip_cfg`, contact metrics in the reward log |
| `terrain_env_cfg.py` | 30 contact-safety config fields, `terrain_eval_seed_override`, width derivation follows the contact flag |
| `terrain_rewards.py` | `compute_contact_safety_reward` (7 terms), `airborne_task_reward_bound` |
| `terrain_observations.py` | 6 contact channels, `observation_space_for_cfg`, history depth honoured in the width check |
| `terrain_convergence.py` | `CONTACT_SAFETY_GATE_SPECS`, `contact_safety_promotion_blockers`, `summarize_convergence(contact_safety=…)` |
| `scripts/evaluate.py` | contact metrics, `--strict_contact`, `--eval_terrain_seed`, plant/observation overrides, never terminates on contact loss |
| `scripts/train.py` | `--contact_focus`, `--contact_stage`, `--height_schedule`, contact and impact-energy overrides |
| `scripts/train_until_converged.py` | forwards every width-changing and plant flag to BOTH children |
| `scripts/migrate_shared_policy.py` | generalized, width-generic, behaviour-preserving migration |
| `scripts/probe_contact_loss.py` | **new** — event-level diagnosis |
| `scripts/smoke_contact_safe.py` | **new** — integration smoke test, both actuators |
| `scripts/summarize_contact_reports.py` | **new** — report aggregation |
| `scripts/run_*.sh` | **new** — diagnosis, sweep and training drivers |
| `tests/test_contact_safety.py` | **new** — 42 contracts |
| `PPO4_ID_FD_PhysX/tests/test_ppo4_contract.py` | contact-safety fields added to the forbidden-assignment list; shared target-lead contract |

PPO4 gained **no** task logic. Every behaviour above is defined in
`PPO3_SOLE_PhysX` and inherited, and the PPO4 contract test forbids PPO4 from
assigning any of the new fields — if it could set its own gap tolerance, "PPO4
lifts more than PPO3" would stop being a statement about the actuator.

---

## 3. Contact-support state machine

`TerrainSolePolicyEnv._update_contact_safety`, cached on
`common_step_counter`. `DirectRLEnv.step` increments that counter once and then
calls `_get_dones`, `_get_rewards`, `_reset_idx` and `_get_observations` in
that order; all consumers go through the same update, so one transition is
counted once rather than three times. Verified on the real environment, not a
stub: the smoke test records the marker delta per step and asserts it is
exactly `{1}`.

Tracked: `_contact_acquired`, `_contact_acquisition_steps`,
`_contact_loss_steps`, `_previous_contact_force_n`, `_previous_contact_state`,
`_previous_cutter_z`, `_airborne_gap_m`, `_takeoff_event`,
`_impact_ejection_event`, `_hard_impact_event`, `_visible_lift_event`, plus
per-episode accumulators.

**It writes nothing.** No cylinder target, valve command or joint reference.
Enforced by an AST test over the four methods, so PPO remains the sole scored
controller.

Hysteresis, all configurable:

| field | default | meaning |
|---|---|---|
| `terrain_contact_acquire_steps` | 3 | consecutive contact steps before support is claimed |
| `terrain_contact_loss_grace_steps` | 2 | consecutive missing steps tolerated (curriculum A=4, B=2, C=1, strict=0) |
| `terrain_max_airborne_gap_m` | 0.005 | geometric gap that counts as a visible lift |
| `terrain_takeoff_velocity_threshold_mps` | 0.02 | upward velocity at separation |
| `terrain_hard_impact_force_n` | 30000 | hard-impact force |
| `terrain_impact_ejection_force_n` | 30000 | force before a takeoff that makes it an ejection |

Support is never acquired and no event is raised while
`_preposition_counter > 0` or `_settle_counter > 0`.

**A visible lift requires a geometric gap AND a loss that outlived the grace
window.** That is what makes "zero visible lifts" a claim about where the tool
is rather than about the contact sensor, and it is what lets a demonstrably
isolated sensor miss with zero gap pass.

---

## 4. Termination — and why it is not the primary mechanism

`contact_safety_terminal()` covers sustained post-acquisition loss, excessive
airborne gap, hard impact and repeated ejection, with
`terrain_lift_terminal_reward = -100.0` applied inside the `active` mask.

**Resets in this task are all-or-nothing.** `_run_reset_preposition` raises
when handed fewer than `num_envs` environments, because the physical
deployment ramp steps the whole scene. A contact termination therefore has to
synchronize like every other termination the parent produces — and the first
version of this work did not, which would have crashed the run with a
`RuntimeError` the first time the policy lifted. Fixed, and a test asserts the
parent's guard string so the reasoning is re-derived if that restriction ever
changes.

The consequence sets the training design. One environment lifting resets all
of them, and the pre-fix policy lifts ~430 times per 32-env/960-step rollout;
at 256 training environments that is continuous reset work rather than
training — the same trap `defer_safety_terminations` exists to escape. So
**the reward carries the signal and termination is the belt-and-braces**:
default off, enabled only in curriculum stage C and only behind
`STAGE_C_TERMINATION=true`.

---

## 5. Reward

`terrain_rewards.compute_contact_safety_reward`, seven terms, every one
bounded, the whole group zeroed while inactive:

```
r_support  = +w_support * contact
r_loss     = -w_loss * acquired * (1 - contact)
r_force    = +w_force * contact * exp(-((F - F_target)/F_std)^2)
r_excess   = -w_excess * clamp(relu((F - F_soft)/F_soft), 0, 5)^2
r_hard     = -w_hard * hard_impact
r_takeoff  = -w_takeoff * weak_support * clamp(relu(v_z)/v_scale, 0, 5)^2
r_gap      = -w_gap * clamp(gap/gap_scale, 0, 4)^2
```

| parameter | value | why this value |
|---|---:|---|
| `w_support` | 4.0 | replaces the binary bonus; `terrain_contact_reward_weight` goes to 0 so support is not paid twice |
| `w_loss` | 20.0 | must exceed every task reward an airborne env can collect (cut 8.0 + lateral 3.0 + progress 1.5 = 12.5). Asserted in tests, not commented |
| `F_target` | **6000 N** | **not** the 2 kN design figure — measured stable contact runs at 18.9 kN mean / 7.2 kN p05, and a band centred on 2 kN is flat everywhere the policy lives. That is the reward-saturation defect the head penalty already had (§3.5 of `HANDOVER_PPO3_PPO4.md`: alignment 0.9477, 0.80 and 0.66 all scored −40.00). 6 kN is gentler than today and inside the reachable set |
| `F_std` | 8000 N | wide enough that the measured 18.9 kN mean retains gradient |
| `F_soft` | 20000 N | just above the measured in-contact mean, so the excess penalty presses down on the operating point |
| `w_hard` | 20.0 | at the 30 kN threshold |
| `w_takeoff` | 4.0, `v_scale` 0.10 | only under weak support — penalizing upward motion while firmly supported would fight the tool following a rising verge, which is the task |
| `w_gap` | 8.0, `gap_scale` 0.005 | continuous in the gap: with a binary term, 4 mm and 40 mm score identically and PPO has no gradient back down |

Signature defaults mirror the config exactly, and a test asserts they still do
— a default that drifts from the config makes every unit test measure a term
the environment never computes.

---

## 6. Observation

Six channels appended after the model-error block, so every migrated column
keeps its meaning:

| index | channel | encoding |
|---|---|---|
| +0 | `contact_boolean` | physical PhysX contact |
| +1 | `log_contact_force` | `log1p(F / 1000)` |
| +2 | `contact_force_rate` | `(F − F_prev)/dt / 1e6`, clipped ±5 |
| +3 | `airborne_gap` | gap / 0.02, clipped, zero while supported |
| +4 | `contact_acquired` | episode has had real support |
| +5 | `contact_loss_steps` | consecutive missing steps / grace |

The pre-existing `cutter_contact_force_normalized` clips `F/1000` at 10, i.e.
at 10 kN, while the in-contact distribution has a 19.7 kN mean and a 119–170 kN
maximum. **Every force the policy has ever had to react to was the same number
to it.** The clipped channel is retained rather than replaced so migrated
checkpoints keep their learned weights on it. `cutter_vertical_velocity` was
already present at index 11 and is not duplicated.

Model-error history raised 1 → 3 (Phase 5). Note that
`build_verge_observation` previously ignored history depth in its own
dimension check, so depth > 1 could never have been enabled at all.

Width: 54 base + 36 model error (12 × 3) + 6 contact = **96**.

Seven entry points each rebuilt this width from the same flags; they now share
`observation_space_for_cfg`.

### Checkpoint compatibility

Both routes from the task were taken. `migrate_shared_policy.py --mode
widen_observation` copies compatible weights and zero-initializes new input
columns with a neutral normalizer (mean 0, var 1, std 1), so the migrated
network computes exactly what the old one computed until training moves them;
and the results live in **new** experiments, `ppo3_sole_rom_contact_safe_v1`
and `ppo4_sole_idfd_contact_safe_v1`. Old checkpoints, reports and experiment
folders are untouched. Migration sets
`migration_requires_optimizer_reset`, which `train.py` refuses to ignore.

```
logs/rsl_rl/ppo3_sole_rom_contact_safe_v1/model_seed_96.pt   (from shared_v2_rom/model_migrated.pt)
logs/rsl_rl/ppo4_sole_idfd_contact_safe_v1/model_seed_96.pt  (from shared_v2_idfd/model_validated.pt)
```

---

## 7. Gates

Nine strict gates in `CONTACT_SAFETY_GATE_SPECS`, kept **out** of `GATE_SPECS`
so the historical 17-gate denominator of every stored report is unchanged:

| gate | metric | threshold |
|---|---|---|
| `visible_lift_events` | `visible_lift_event_count` | ≤ 0 |
| `impact_ejection_events` | `impact_ejection_event_count` | ≤ 0 |
| `hard_impacts` | `hard_impact_event_count` | ≤ 0 |
| `consecutive_airborne_steps` | `maximum_consecutive_airborne_steps` | ≤ 0 |
| `airborne_gap` | `maximum_airborne_gap_m` | ≤ 0.005 |
| `post_acquisition_contact` | `post_acquisition_contact_fraction` | ≥ 1.0 |
| `contact_acquired` | `contact_acquired_fraction` | ≥ 1.0 |
| `takeoff_events` | `takeoff_event_count` | ≤ 0 |
| `contact_force_peak` | `maximum_contact_force_n` | ≤ 30000 |

Plus two non-numeric promotion blockers: `teacher_mode` must be `"off"` and the
direct target-writing contact demo must be inactive. A run that violates either
is not a sole-controller result whatever its contact numbers say.

The evaluator **measures** contact loss and never terminates on it
(`terrain_terminate_on_contact_loss = False` is forced in `build_env`): a gate
that truncates its own evidence cannot size an excursion.

New metrics in every report: `contact_acquired_fraction`,
`post_acquisition_contact_fraction`, `contact_loss_event_count`,
`contact_loss_steps`, `maximum_consecutive_airborne_steps`,
`maximum_airborne_gap_m`, `p95_airborne_gap_m`, `takeoff_event_count`,
`impact_ejection_event_count`, `hard_impact_event_count`,
`mean_contact_force_n`, `p95_contact_force_n`, `maximum_contact_force_n`,
`mean_force_rate_nps`, `p95_force_rate_nps`,
`maximum_positive_cutter_vertical_velocity_mps`, `minimum_scan_valid_fraction`,
`target_lead_p95_m`, `target_lead_max_m`, `shield_active_fraction`,
`min_cylinder_headroom_m`, plus `teacher_mode` and
`direct_target_contact_demo`.

---

## 8. Impact-energy sweep (Phase 6)

Artifacts: `contact_safe_v1/sweep/`. 960 steps × 16 envs, held-out mesh,
`s_curve`, relief 1.0, **shipped policy held fixed**. This measures how much
impact energy a plant configuration admits, not what a policy trained for it
could do.

One knob at a time from the shipped baseline, not a cross product: the
repository's own notes record `terrain_arm_stiffness_scale` as **non-monotonic**
(0.5 leaves a 19.7 mm hop, 0.2/0.1 lose contact, 0.05 is worse than 0.2), so a
coordinate sweep is the only reading these knobs support. 960 steps, not 480 —
the events concentrate late in the path, so a half-horizon sweep would compare
configurations on the easy part.

### PPO3 / ROM-A

| config | contact | events | lifts | peak F (kN) | max gap (m) | max lead (m) | max v_z (m/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.9575 | 188 | 146 | 146.2 | 0.2861 | 0.0303 | 2.65 |
| `action_scale` 0.0035 | **0.9585** | **152** | **114** | 125.0 | 0.2733 | 0.0303 | **2.40** |
| `action_scale` 0.0030 | 0.9537 | 171 | 130 | 113.8 | **0.2447** | 0.0303 | 2.97 |
| `rate` 0.20 | 0.9507 | 192 | 156 | 153.9 | 0.2878 | 0.0314 | 3.03 |
| `rate` 0.15 | 0.9452 | 214 | 174 | 153.3 | 0.2435 | 0.0319 | 2.92 |
| `stiffness` 0.25 | 0.9444 | 257 | 198 | **96.0** | 0.2554 | 0.0293 | 1.93 |
| `stiffness` 0.20 | 0.9202 | 346 | 266 | 113.5 | 0.2992 | 0.0332 | 3.68 |
| `lead` 0.015 | 0.9329 | 321 | 238 | 115.3 | 0.3142 | 0.0197 | 3.43 |
| `lead` 0.010 | 0.9205 | 336 | 213 | 107.3 | 0.3048 | **0.0166** | 3.18 |

### PPO4 / ID-FD

| config | contact | events | lifts | peak F (kN) | max gap (m) | max lead (m) | max v_z (m/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.9048 | 154 | 125 | 186.0 | 0.5531 | 0.0320 | 6.48 |
| `action_scale` 0.0035 | 0.9147 | 202 | 154 | 146.3 | 0.5562 | 0.0320 | 6.53 |
| `action_scale` 0.0030 | 0.9159 | 174 | 126 | 148.2 | 0.6470 | 0.0320 | 6.80 |
| `rate` 0.20 | 0.9079 | 210 | 182 | **386.2** | **1.0926** | 0.0403 | **7.03** |
| `rate` 0.15 | **0.9398** | **136** | **98** | 130.8 | 0.6456 | 0.0290 | 2.26 |
| `stiffness` 0.25 | 0.8875 | 240 | 189 | 164.2 | 0.6389 | 0.0348 | 4.60 |
| `stiffness` 0.20 | 0.8885 | 262 | 196 | 96.6 | 0.4449 | 0.0364 | 4.04 |
| `lead` 0.015 | 0.8934 | 248 | 168 | 180.0 | 0.3610 | 0.0208 | 3.10 |
| `lead` 0.010 | 0.8704 | 299 | 218 | 126.5 | **0.3685** | **0.0164** | 2.15 |

### Reading it honestly

* **Every knob that reduces impact energy also costs the FIXED policy
  contact.** That is expected and is not evidence the knob is wrong: this
  policy was trained with 25 mm of command authority and is co-adapted to it,
  the same "off-distribution plant change for a co-adapted checkpoint" pattern
  `terrain_kinematic_drive_substep_interpolation` already documents (enabling
  that fix under v5 took 16/17 to 15/17 while improving every quantity it
  targets). The retrained policy is the test.
* **`lead` is the only knob that moves the mechanism itself.** It is the only
  one that takes measured max lead off its bound (0.030 → 0.0166), and it cuts
  peak force on both actuators (146 → 107 kN, 186 → 127 kN).
* **`rate` points in opposite directions on the two actuators.** PPO4 at 0.20
  is a warning, not a candidate: 386 kN peak force, 1.09 m max gap, 7.03 m/s
  ejection — worse than baseline on every axis — while 0.15 is its best row and
  PPO3's second worst. These knobs are non-monotonic exactly as the existing
  notes say. It is therefore **not set**; setting it per-actuator would destroy
  the attribution PPO4 exists to provide.
* **`stiffness` 0.25 gives PPO3 its lowest peak force (96 kN) and lowest
  ejection velocity (1.93 m/s)** but costs contact, and 0.20 is worse on both
  actuators. Left at 0.35.

### Chosen for training

| field | value |
|---|---|
| `action_scale` | 0.0035 |
| `cylinder_target_max_lead_m` | 0.010 |
| `terrain_arm_stiffness_scale` | 0.35 (unchanged) |
| `sole_max_action_rate` | not overridden (curriculum 0.25) |
| `cylinder_target_leak` | 0.0 (lead reduction first, per the task) |
| `terrain_head_joint_stiffness_scale` | 1.0 (unchanged) |

`terrain_head_joint_stiffness_scale` stays at 1.0 throughout: softening the
head orientation joint with the cylinders was already measured to cost
`min_head_alignment` 0.9500 → 0.9150 while buying none of the bounce reduction.

PhysX compliant contacts (`terrain_compliant_contact_stiffness/damping`) are
**not** used and no claim here rests on them. A previous 10× sweep that should
have cut force ~10× moved median peak force 50.4 → 46.9 kN, i.e. the setting
is not honoured for this contact pair.

### A bug this sweep found in itself

The first pass returned three **bit-identical** `action_scale` rows on both
actuators. That was not the knob having no effect — the override was written
as `sole_action_scale` and the real field is `action_scale`. `configclass`
accepts arbitrary attribute assignment, so the misspelling was a silent no-op
producing a perfectly plausible "this knob does nothing" result; only the exact
duplication gave it away. Two fixes: the name, and
`terrain_observations.override_cfg`, which refuses to create a config field
that does not exist and through which every CLI override now passes. A second
ordering bug surfaced while tracing it — `apply_sole_curriculum_stage` rewrites
`action_scale` and `sole_max_action_rate` per stage, and `train.py` applied the
curriculum *after* the CLI overrides — so `apply_plant_overrides()` is now
re-asserted after every curriculum application.

---

## 9. Training and final evaluation

*Filled in when the runs land.*

---

## 10. Rejected and failed experiments

*Recorded as they occur. Nothing is omitted for being unflattering.*

1. **`--seed` as a multi-mesh robustness check.** Ran, produced four
   bit-identical reports, rejected. Artifacts retained in
   `contact_safe_v1/prefix_identical_seed_artifact/`.
2. **Contact termination without synchronization.** Implemented, then found to
   crash the run on the first lift (`_run_reset_preposition` raises on a
   partial reset). Fixed by synchronizing.
3. **Contact termination as the primary training signal.** Rejected on the
   measured reset arithmetic in §4, not on principle: at 256 environments and
   ~430 lifts per rollout it is continuous reset work. The reward carries the
   signal instead.
4. **`F_target = 2000 N`.** The design figure, and the obvious starting point.
   Rejected: measured stable contact is 18.9 kN mean / 7.2 kN p05, so a band
   centred at 2 kN has no gradient anywhere the policy operates. Caught by a
   unit test before any GPU time was spent on it.

---

## 11. Limitations

* The held-out mesh is pinned by design, so multi-mesh claims require
  `--eval_terrain_seed` and are a different (weaker) statement than the gate
  mesh score. Both are reported separately.
* The strict gates use absolute counts, so a longer horizon is a strictly
  harder test. Every strict number here is at 960 steps.
* `terrain_arm_stiffness_scale` is non-monotonic; the coordinate sweep in §8
  bounds behaviour at the swept points and says nothing between them.
* The head-alignment problem that dominates `HANDOVER_PPO3_PPO4.md` is
  untouched by this work. It did not appear in any contact-loss trace examined
  (alignment 0.9996–1.0000 throughout), so the two are treated as independent
  — that is an observation, not a proof.

---

## 12. Confirmation: PPO is the sole scored controller

* No PD, DLS, force, direct-target or privileged-height controller was added.
* The contact state machine writes no plant command; enforced by an AST test
  over its four methods.
* `sole_teacher_mode = "off"` and `enable_contact_tracking_demo = False` in
  every reported run, recorded in each report's header and asserted by
  `assert_teacher_disabled` before the run starts.
* `contact_safety_promotion_blockers` refuses to promote a checkpoint whose
  report shows either of those was active.
* The smoke test verifies on the real plant that PPO3 constructs ROM-A and no
  ID runtime, and PPO4 constructs ID → FD and **no ROM runtime at all**.
