# Stage 6 — flat road with raised chunks, wide tracking line

Branch `work/ppo3-final-terrain-policy`. Covers the terrain pivot, the head
fix, the PPO4 windup fix, and the adaptivity result.

Sections marked **[IN FLIGHT]** are running as this is written; every other
number is read from a stored artifact and the file is named next to it.

---

## 1. What the terrain is now

Stage 5 asked the arm to follow ground that undulated *everywhere*. That made
cut height the dominant difficulty and left the lateral path exercising barely
a third of the available verge — the opposite of the stated objective, which is
tracking.

Stage 6 inverts it: **flat at road level for long runs, with discrete raised
sections between**, and the tracking line swung across most of the half-strip.
The chunks are the disturbance; the line is the task.

`terrain.generate_flat_patches` ([terrain.py](../terrain.py)):

| property | value |
|---|---|
| chunks per strip | 3–6, drawn per environment |
| chunk height | 0.10–0.38 m, uniform |
| chunk placement | evenly slotted, then jittered ±0.35 slot-width |
| max surface slope | 0.26 (hard cap) |
| tracking amplitude | 2.2× — mean peak-to-peak ~0.20 m → ~0.44 m |

**Width is derived from height, not drawn independently.** A chunk's minimum
width is `0.607 * height / (0.26 * length)`, then widened by 1.05–1.6×. Drawing
the two independently would eventually produce a tall narrow draw — a wall the
arm cannot follow at any skill level, which would make the stage unlearnable
for a reason unrelated to control. The cap is what keeps the stage hard but
fair.

### 1.1 The randomization bug (found and fixed)

The layout was drawn **once** and shared by all 256 environments. Every
parallel environment, and every resumed block, saw the *same* chunks in the
same places. A policy could memorise one mesh and score well, which is exactly
what the stage exists to prevent — and it explains why adaptivity across unseen
meshes was sitting at coin-flip (9/18).

Cause: the relief profile was applied with the base seed, and the result was
then re-seeded per environment — so the per-env seed changed the path family
but never reached the chunk layout, which had already been baked in.

Fix ([terrain_env.py](../terrain_env.py), `_mow_strip_cfg_unprofiled` +
`_spawn_mow_strip`): re-seed **first**, draw the profile **second**.

```python
env_strip_cfg = replace(unprofiled_cfg, seed=base + env_index * stride, ...)
env_strip_cfg = apply_relief_profile(env_strip_cfg, relief_profile)
```

Locked by two tests in [test_terrain_paths.py](../tests/test_terrain_paths.py):
`test_every_environment_gets_its_own_chunk_layout` and
`test_the_spawner_draws_the_profile_after_the_per_env_seed`. The second test
asserts the *ordering*, not just the outcome, because the outcome alone would
still pass if someone reintroduced a shared layout with per-env noise on top.

---

## 2. Stage-5 checkpoints do not transfer

Measured before committing to any retraining:

| | stage 5 | stage 6 |
|---|---|---|
| PPO3 cells passing | 32/32 | **0/3** |

The dominant new failure was lateral tracking, not cut height — consistent with
the stage being a different task rather than a harder one. So the amplitude is
**ramped, not jumped**:

| sub-stage | amplitude | relief heights |
|---|---|---|
| S6a | 1.4 | 0.50, 0.70 |
| S6b | 1.8 | 0.70, 0.85, 1.00 |
| S6c | 2.2 | 0.50, 0.70, 0.85, 1.00 |

---

## 3. The head fix

**Symptom.** q5 was identically zero on PPO3 — the head action did nothing.

**Root cause.** The correction measured tilt in **world Y-Z**. As the arm
slews, the correctable component of tilt rotates into world **X** (measured
correlation +0.814), so the controller was regulating a projection that no
longer contained the error it was trying to remove.

**Fix.** Solve the tilt in the **arm frame** (`--head_tilt_arm_frame`), and give
the policy the head frame directly as two observation channels
(`--head_frame_observation`, width 96 → 98).

**Result** — PPO3 seed checkpoint, stage 6 held-out, 2.2×, 32 envs × 960 steps
(`stage_S6a/reports/heldout_initial.json`), **all 17 gates pass**:

| metric | value |
|---|---|
| min_head_alignment | 0.9744 |
| p01_head_alignment | 0.9984 |
| max_abs_head_axis_x | 0.0096 |
| ground_contact_fraction | 0.9475 |
| mean_abs_cut_height_error_m | 0.0038 |
| p95_abs_cut_height_error_m | 0.0 |
| mean_abs_lateral_error_m | 0.0740 |
| p95_tool_acceleration_mps2 | 44.55 |

The head fix worked. It is a **separate** problem from contact safety (§5) and
the two must not be conflated.

---

## 4. The PPO4 q5 windup fix

PPO4's q5 carries a DC bias of **0.324**. The correction is a velocity
integrator (`correction += 0.010 * action`, clamped ±0.35 rad), so that bias
saturates the bound in ~108 of 960 steps and holds a permanent ~20° head
offset. PPO3's q5 sits at 0.0114 and never winds up — that is the entire
difference between their head alignment, 0.9875 vs **0.5303**.

Two contributing defects, both real:

1. no leak on the integrator, so any DC bias walks to the bound and stays;
2. `_sole_applied` is 4-wide, so **q5 was never penalised** by the reward.

Fix: `sole_head_correction_leak`, which settles a persistent bias at
`scale*action/leak` instead of at the bound, plus
`terrain_head_correction_weight` to tax the standing offset during training.

**The leak value is not yet chosen by measurement.** [IN FLIGHT] —
`run_q5_windup_sweep.sh` sweeps 0.0 / 0.010 / 0.020 / 0.040 / 0.080 on the
fixed checkpoint with the policy held constant. 0.0 is the control and must
reproduce the failing baseline exactly. A previous attempt reached 2 of 5 cells
before being stopped and is not reported here.

---

## 5. What is NOT fixed: strict contact safety

**The headline requirement — zero visible lift — is not met, on any cell, on
either actuator.** This has been true across every campaign in this branch and
it is not improving.

PPO3 seed checkpoint on stage-6 held-out terrain, same run that passes all 17
functional gates:

| contact metric | value |
|---|---|
| visible_lift_event_count | 152 |
| hard_impact_event_count | 1049 |
| impact_ejection_event_count | 34 |
| takeoff_event_count | 304 |
| maximum_contact_force_n | 141 019 |
| maximum_consecutive_airborne_steps | 43 |
| max positive cutter vertical velocity | 9.33 m/s |

8 of 9 strict contact gates are blockers; only `contact_acquired` passes.

The mechanism is established (V1 report §0): the cylinders are stiff enough
that **commanded position is commanded force**, the reference integrator winds
up against terrain the cylinders cannot follow, and the stored command releases
as an impulse that ejects the tool. `cylinder_target_max_lead_m` was reduced
25 mm → 18 mm on measurement (§6), which reduced but did not remove it.

**Assessment.** This is a plant property, not a policy deficiency. A 141 kN
peak against a ~5e6 N/m cylinder is ~28 mm of commanded interference; no
policy operating through that actuator can produce gentle contact, because it
has no authority that is soft enough. The remedy is mechanical — see
[PASSIVE_CUTTER_COMPLIANCE_DESIGN.md](PASSIVE_CUTTER_COMPLIANCE_DESIGN.md),
which is design-only and deliberately unimplemented pending a decision.

---

## 6. Rejected parameter combinations

Recorded because a rejected setting is a result.

| parameter | tried | outcome |
|---|---|---|
| `cylinder_target_max_lead_m` | 0.010 | **Rejected.** Claimed to remove saturation; audit showed it *doubled* it (43.4% vs 22.3% of steps at the bound). |
| `cylinder_target_max_lead_m` | 0.025 (inherited) | Rejected — ~125 kN of stored command. |
| `cylinder_target_max_lead_m` | **0.018** | **Accepted.** Zero-action control shows the plant's natural lead is ~5 mm; 18 mm leaves headroom without storing an impulse. |
| `terrain_arm_stiffness_scale` | 1.0 (rigid) | Rejected — hop 43 mm, p95 accel 91.7. |
| `terrain_arm_stiffness_scale` | **0.35** | **Accepted.** Hop 4.6 mm, p95 accel 48.7. Knob is **non-monotonic** — do not extrapolate. |
| accel gate | 80 m/s² | **Rejected as vacuous** — below the measured floor of 100.06. Re-derived to 120. |
| head teacher, proportional over action scale | — | Rejected — saturated at 0.57° of tilt, i.e. bang-bang. Replaced by proportional over a 0.05 rad band + rate damping. |
| teacher gates (contact 0.90, p95 height 0.08, head 0.95) | — | Rejected — fail on the compliant arm for reasons unrelated to teacher health; re-derived to 0.70 / 0.25 / 0.65. |

---

## 7. Can the teacher track stage 6?

Asked because distillation is the natural fallback if PPO cannot learn the
stage. Three findings:

1. **The policy is not learning from the teacher.** `train.py` has no DAgger
   plumbing; only `distill_terrain.py` does. The teacher produced a seed
   checkpoint once. Every scored run since has `sole_teacher_mode="off"`, and
   `evaluate.py` refuses to run otherwise.
2. **Every teacher number on record is stage 4 on the old relief**, and the
   teacher degraded badly when the arm became compliant:

   | stage-4 metric | rigid arm | compliant arm (current) |
   |---|---|---|
   | ground_contact_fraction | 0.972 | 0.888 |
   | p95 cut height error | 0.0 m | 0.114 m |
   | min_head_alignment | 0.970 | 0.739 |

3. **The student already beats it on harder terrain** (§3: 0.947 contact,
   3.8 mm mean height error, p95 0.0, tool accel 44.5 vs the teacher's 61–72).

So re-seeding from the teacher would move the policy backwards, and since the
teacher's tool acceleration is *higher*, it cannot be a source of a lift-free
demonstration either. [IN FLIGHT] `run_teacher_probe.sh` measures the teacher
on stage 6 directly rather than arguing from stage-4 numbers; it is chained to
start when the GPU frees.

---

## 8. Process defects that cost real time

Kept because they were the dominant cost of this work, not the control problem.

- **Editing live code during a run.** A reward term whose definition landed
  *after* its use raised `UnboundLocalError` and destroyed **33 of 36**
  evaluation cells. Static checks, unit tests and the width contract all passed.
  Only a real rollout catches it — hence `preflight.py` Phase 3b, a 30-step
  rollout on both actuators, which catches it in two minutes.
- **Trusting exit codes.** Isaac masks non-zero exits; the supervisor evaluated
  a stale checkpoint and reported it as a result. Fixed by artifact
  verification, then by the 7-criterion provenance ledger
  ([block_provenance.py](../block_provenance.py)).
- **Silent no-op flags.** `--action_scale` was read as `sole_action_scale` and
  did nothing, producing three bit-identical sweep rows. Fixed, plus
  `override_cfg` now refuses to create a field that does not exist.
- **Counting samples instead of edges.** "1559 hard impacts" was a 5% duty
  cycle. Fixed with rising-edge detection.
- **A gate score saturated at 0.0**, discarding a strictly better checkpoint
  (alignment 0.9911 vs 0.9826). Fixed with `_tie_break_metrics`.

---

## 9. Status

| item | state |
|---|---|
| flat_patches terrain + stage 6 | done, 8 tests |
| per-environment layout randomization | done, 2 tests |
| head-frame fix | done — 17/17 gates, alignment 0.9744 |
| PPO3 retrain through 1.4 → 1.8 → 2.2 | [IN FLIGHT] |
| PPO4 retrain through the ramp | [IN FLIGHT] |
| PPO4 q5 leak chosen by measurement | [IN FLIGHT] |
| adaptivity, 18 cells × 2 actuators | [IN FLIGHT] |
| teacher probe on stage 6 | [IN FLIGHT] |
| **strict zero-contact-loss** | **NOT MET — 0/N on every cell, both actuators** |

249 tests pass. Preflight passes on both actuators including the 30-step
rollout.
