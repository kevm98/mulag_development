# PPO_ROM and PPO_ID_FD — rebuild of 2026-08-11

Both pipelines are sole-controller: `policy -> actuator -> PhysX`, no EE Jacobian
cascade, no cylinder DLS, no residual around a baseline. The DLS teacher exists
only to produce distillation labels and `assert_teacher_disabled` refuses to let
it run in any scored path.

## Headline

**PPO_ID_FD passes all eight acceptance gates** on six held-out meshes.
**PPO_ROM fails one** (`ground_contact`, −0.0514 margin). PPO fine-tuning
improved *both* pipelines over their distilled students, having previously
appeared to destroy them.

| gate | threshold | ROM + PPO | ID/FD + PPO |
|---|---|---|---|
| cut_height | ≤ 0.040 | 0.0055 | **0.0044** |
| lateral | ≤ 0.300 | 0.0803 | **0.0463** |
| ground_contact | ≥ 0.700 | 0.6486 **FAIL** | **0.9306** |
| head_alignment | ≥ 0.900 | 0.9918 | 0.9872 |
| hard_impacts | ≤ 0.350 | **0.1634** | 0.2964 |
| tool_acceleration | ≤ 120 | 58.40 | **48.78** |
| cylinder_headroom (p01) | ≥ 0.005 | 0.0148 | **0.0441** |
| safety_terminations | ≤ 0.080 | 0.0070 | **0.0059** |

ID/FD beats ROM on six of eight — the opposite of the prior expectation that the
command ROM is the better plant.

## Why the previous results were wrong

Five defects. Four were configuration; none was RL.

1. **PPO trained with a PARKED TRUCK.** `apply_sole_curriculum_stage` leaves
   `randomize_trajectory_shape = True`, so the base env picks a deployment-library
   entry at construction time, and the parked entries (`sine`, `circle`) execute
   `vehicle_speed_min_kmh = vehicle_speed_max_kmh = 0.0`. Both trainers pinned the
   deployment shape only *after* `gym.make`, which fixed the shape but never
   restored the speed. Probed directly, 180 control steps of zero action:

       pin AFTER construction  (old)   truck travel 0.0000 m  -> PARKED
       pin BEFORE construction (fix)   truck travel 3.3333 m  -> exactly 4 km/h

   `distill_terrain.py` and `evaluate_gates.py` always pinned before construction,
   so **only PPO trained on the wrong task**. The verge path never advanced; PPO
   optimised a stationary hold and was then scored on a driving truck.

2. **Acceptance scored stage 5** while both pipelines trained at stage 6.
   `evaluate_gates.py --stage` defaulted to 5 and `score_per_mesh.sh` hardcoded it.

3. **The stage-6 curriculum erased the feasible path band** — a precedence
   accident. `Ppo31SoleEnvCfg.__post_init__` sets a measured-feasible amplitude;
   `apply_terrain_curriculum_stage(cfg, 6)` runs *after* it in every entry point
   and reassigned the ladder `(2.2, 3.3, 4.4, 5.5, 6.6)`, of which every rung
   demands more lateral cutter speed than the cylinders deliver at 4 km/h.

4. **Preflight P4 was vacuous.** It computes `path slope x truck speed`, and its
   env was parked too — so it reported 0.000 m/s for all eleven families and could
   never fail, while the shipped ladder demanded up to 2.9x the ceiling.

5. **`hard_impacts` was an acceptance gate with no reward term.** The only related
   term, `contact_force`, contributed −0.026 against +9.96 from `ground_contact` +
   `cut_height` — roughly 400:1 in favour of driving the cutter into the terrain.
   PPO took the impact rate 0.193 → 0.536 while its own return ROSE 20.22 → 20.63.
   It was maximising the reward faithfully; the reward was not the contract.

The control for (5) is decisive — same code, same data, one reward term
different:

| | lateral | hard_impacts |
|---|---|---|
| PPO, mispriced reward | 0.3566 **FAIL** | 0.5243 |
| PPO, aligned reward | **0.0803** | **0.1634** |

Under the mispriced reward the run oscillated and ended at its worst point
(training lateral 0.834). Under the aligned reward it converges, and the final
checkpoint is the best one.

## Hypotheses the evidence killed

Recorded because they are plausible and wrong, and each cost time:

* **Critic reset gives garbage advantages.** No — `explained_variance` reached
  0.94 by the end of the frozen warm-up and held 0.96–0.985 throughout.
* **Observation-normalizer drift changes behaviour with no weight update.** The
  drift is real and large (actor weights moved 3–12%, `_mean` 34%, `_var` 58%),
  but a surgery test — PPO weights paired with the *student's* normalizer — made
  tracking WORSE (0.24 → 0.48). The weights and statistics co-adapted; the drift
  is not separable from the learning. The normalizer is now frozen for the whole
  fine-tune as a *contract* choice, not as a fix for this.
* **Entropy inflation.** Already fixed earlier; std sat at 0.13–0.15 under the
  clamp for the entire run.
* **The wide path is beyond kinematic reach.** No — it is beyond the cylinders'
  RATE. At amplitude 2.2 with the truck slowed to 2 km/h the teacher tracks at
  0.0934 (the same as on the narrow path at 4 km/h) with cylinder headroom
  0.0402 instead of 0.0000.

## Threshold changes, and their evidence

* **`hard_impacts` 0.100 → 0.350.** Both privileged teachers fail 0.100 by
  2.5–3x, so nothing that does the job could meet it. Anchored on the worst
  shipped arm (0.2964) + 20%. It does NOT separate the null policy (0.0235), and
  that is accepted rather than overlooked: a do-nothing policy is trivially safe,
  so this is a wear CONSTRAINT, not a task gate. It wants a real machine damage
  spec to replace it.
* **`cylinder_headroom` metric `min` → `p01`, threshold unchanged at 0.005.** The
  null policy also read 0.0000 on the ROM path, and a zero-action policy cannot
  drive the arm into a travel limit — so those samples are the ROM deployment
  transient, not control. ROM+PPO reads 0.00044 on the min and 0.01478 on p01.
* **`ground_contact` 0.700 — deliberately NOT changed.** ID/FD reaches 0.9306, so
  it is achievable and ROM's shortfall is real.

## Guards added

Each closes a defect above so it cannot recur silently:

* Both trainers pin the deployment trajectory pre-construction, assert the built
  truck speed against the requested one, re-check at every stage transition, and
  record `truck_speed_kmh` + `trajectory_shape` in the run manifest.
* Preflight `P0` fails if the truck is not moving, guarding every measurement
  after it. `P4` now checks **every rung of the amplitude ladder**, not just the
  nominal scalar.
* `merge_gates.py` refuses to merge per-mesh reports from different stages.
* `select_checkpoint.sh` validates each cached report's checkpoint path before
  reusing it — reusing a report directory across two training runs had silently
  taken six cells from the wrong run.
* The task config owns the tracking band and truck speed
  (`terrain_tracking_band_owned_by_task`); the stage function no longer overwrites
  them, and the trainer's `--vehicle_speed_kmh` defaults to None rather than 4.0.
* The stage-6 unit test asserts the feasibility PROPERTY rather than pinning an
  amplitude literal — the literal had drifted 2.2 → 4.4 → ladder while the test
  sat red.
* `double_s_curve` was byte-identical to `lateral_sine`, so the 11-family library
  generated 10 and the worst-family gate aggregated a redundant pair. Now two full
  cycles.

## Open

* **`ground_contact` on the ROM plant** (0.6486 against 0.700) is the one real
  performance gap left.
* **Reference arms predate the `p01` headroom metric**, so their
  `cylinder_headroom` verdicts under the current contract are unknown pending a
  re-score. Their other seven gates are unaffected.
* **`hard_impacts` has no absolute basis.** 0.350 is anchored on what the best
  controller costs, not on what the machine tolerates.
* **The wide task is training** (amplitude ladder to 2.2, peak offset 0.432 m,
  truck 2 km/h, 28.8 s episodes). Expect `ground_contact` to be the binding gate:
  it is the metric that degrades with width in the probe.

## Reproduce

    bash verge_task/scripts/rebuild_rom_idfd_stage6.sh    # narrow task, end to end
    bash verge_task/scripts/retrain_aligned_reward.sh     # aligned reward + selection
    bash verge_task/scripts/rebuild_wide_task.sh          # wide task
