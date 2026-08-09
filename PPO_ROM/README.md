# PPO3_1 — verge line tracking over ROM-A (the canonical PPO3)

**This package supersedes `verge_task` as the PPO3 pipeline.** PPO3 remains
the shared library — terrain generator, observation contract, ROM plant, DLS
teacher, distillation and training scripts all still live there and are imported
unchanged. PPO3_1 owns the parts that were rebuilt: the reward, the acceptance
gates, the c4 command authority, and the terrain.

Status as of **2026-08-06**: shipped policy passes **all eight gates on three
held-out meshes with no thin margins**.

| gate | threshold | student v2 | margin |
|---|---|---|---|
| lateral (worst-family) | ≤ 0.300 m | **0.0794** | +0.2206 |
| ground_contact (worst-family) | ≥ 0.700 | **0.8347** | +0.1347 |
| head_alignment (p01) | ≥ 0.900 | 0.9877 | +0.0877 |
| safety_terminations | ≤ 0.080 | — | +0.0776 |
| cylinder_headroom | ≥ 0.005 m | — | +0.0424 |
| hard_impacts | ≤ 0.100 | — | +0.0391 |
| cut_height | ≤ 0.040 m | 0.0065 | +0.0335 |
| tool_acceleration | ≤ 120 m/s² | — | +54.76 |

Deliverable: `logs/rsl_rl/ppo3_1_sole/student_distilled_v2.pt`
Report: [`reports/frontier/student_v2_gates.json`](reports/frontier/student_v2_gates.json)

---

## The finding that made it work

Every earlier attempt failed on **lateral**, and so did the privileged DLS
teacher. The cause was not the policy, the reward, or the critic — **the task
was physically impossible**.

The cylinders can move the cutter sideways at most

    0.004 m per control step / 0.02 s = 0.200 m/s

and that bound is *optimistic* (it assumes a lateral Jacobian of 1.0 and the
entire cylinder budget spent sideways). At the shipped settings, **9 of 11 path
families demanded more than that**, `boom_sweep` by 3.7×. Error then accumulates
by construction and no amount of reward tuning removes it.

The fix is to set the terrain from measured capability, then re-distill on it.
`scripts/probe_path_feasibility.py` and preflight check **P4** both measure this,
so it cannot silently regress.

**Lateral and contact are separable knobs** — this is the key to satisfying both
gates at once:

| knob | drives | gate it moves |
|---|---|---|
| `terrain_tracking_amplitude_scale` | every family's lateral swing | lateral |
| `terrain_tracking_height_sweep_gain` | `boom_sweep` only (its offset is height-coupled) | lateral |
| `terrain_height_scales` | how much ground there is to touch | ground_contact |

Lowering the heights alongside the amplitude fixed tracking and broke contact
(teacher 0.5417 against a 0.700 gate). Restoring the heights bought contact back
(0.8229) with tracking untouched (0.0623).

---

## Terrain we train on

    relief profile                        flat_patches
    terrain_tracking_amplitude_scale      0.7
    terrain_tracking_height_sweep_gain    0.5
    terrain_height_scales                 (0.50, 0.70, 0.85, 1.00)
    truck speed                           4.0 km/h (1.111 m/s), scripted

Domain randomization, sampled per environment (coprime lengths, so 12
environments cover all 12 pairs exactly once):

    terrain_tracking_amplitude_scales      (0.30, 0.45, 0.60, 0.75)
    terrain_tracking_height_sweep_gains    (0.25, 0.40, 0.55)

A policy generalises over the distribution it saw and essentially nothing else.
Measured: a student distilled at amplitude 2.2 tracked at 0.303 m on
amplitude-0.7 terrain that permits 0.058 m — five times worse than the plant
allows, purely from the shift. Hence the band, not a point.

### The eleven path families

All eleven are inside the actuator's rate ceiling at the shipped settings
(peak demanded lateral cutter speed, worst of meshes 1337/17/909):

| family | swing (m) | demand (m/s) | % of ceiling |
|---|---|---|---|
| `lateral_chirp` | 0.18 | 0.135 | 68% |
| `double_s_curve` | 0.18 | 0.113 | 57% |
| `random_spline` | 0.16 | 0.103 | 51% |
| `fourier_meander` | 0.15 | 0.094 | 47% |
| `lateral_sine` | 0.18 | 0.093 | 46% |
| `boom_sweep` | 0.10 | 0.091 | 45% |
| `s_curve` | 0.23 | 0.084 | 42% |
| `lane_change_hold` | 0.11 | 0.082 | 41% |
| `gaussian_slalom` | 0.17 | 0.067 | 33% |
| `offset_return` | 0.13 | 0.028 | 14% |
| `straight` | 0.00 | 0.000 | 0% |

For reference, at the old settings `boom_sweep` demanded **0.734 m/s** — 3.7×
the ceiling — and was the family the teacher could not track either.

---

## Commands

### Watch a policy in the viewer

```bash
PPO_ROM/visualize.sh                    # shipped student, mesh 1337
MESH=17  PPO_ROM/visualize.sh           # a different held-out mesh
ENVS=4   PPO_ROM/visualize.sh           # four side by side
STEPS=1920 PPO_ROM/visualize.sh         # run longer
PPO_ROM/visualize.sh --teacher          # the privileged DLS teacher
PPO_ROM/visualize.sh logs/rsl_rl/ppo3_1_sole/polish/model_598.pt
```

Each environment draws its own path family, so `ENVS=11` shows the whole set at
once. The families cycle in the order listed in
`verge_task/terrain.py:TRACKING_PATH_FAMILIES`.

> Do **not** visualize through the registered `...-PLAY-v0` task. It builds
> `TerrainC4AuthorityEnvCfg`, which does not carry the feasibility-derived
> terrain — you would watch the policy fail on amplitude-2.2 ground it was never
> trained for. `visualize.sh` goes through the same config path the scoring used.

### Preflight (always, before spending GPU hours)

```bash
PPO_ROM/scripts/preflight.py --num_envs 32 --steps 120
```

Ten checks. **P4** asserts every path family is inside the cylinder rate
ceiling; **S5** asserts every shaping term still has gradient past 2× its gate.

### Distil a student from the DLS teacher

```bash
verge_task/scripts/distill_terrain.py \
    --task RRLAB-Mulag-Verge-ROM-v0 \
    --stage 6 --num_envs 128 --seed 17 --iterations 120 \
    --out logs/rsl_rl/ppo3_1_sole/student_distilled_v2.pt \
    --report PPO_ROM/reports/distillation_v2.json \
    --headless
```

~50 min. **Distillation is the endpoint for this task, not an initialisation** —
see "PPO fine-tuning" below.

### PPO fine-tune (optional; measured to lose)

```bash
NUM_ENVS=256 STAGE_ITERATIONS=600 FREEZE_ITERATIONS=150 \
  STUDENT=logs/rsl_rl/ppo3_1_sole/student_distilled_v2.pt \
  LOG_DIR=logs/rsl_rl/ppo3_1_sole/polish \
  PPO_ROM/scripts/supervise_training.sh
```

Runs under a stall watchdog that resumes from the newest checkpoint (see
"Known environment bug"). ~2 h for 600 iterations.

### Score on the eight gates

```bash
PPO_ROM/scripts/score_per_mesh.sh <name> <ckpt.pt> 480 32
PPO_ROM/scripts/score_per_mesh.sh teacher_check --teacher 480 32
PPO_ROM/scripts/score_per_mesh.sh null_check    --null    480 32
```

Writes `reports/frontier/<name>_m<mesh>.json` per mesh plus a merged
`<name>_gates.json`. ~4 min for three meshes.

### Training progress and charts

```bash
PPO_ROM/scripts/training_status.py --plot
```

Leads with the three signals a run exists to produce: does the critic explain
the returns (explained variance), does exploration std hold, does tracking
survive unfreeze.

---

## PPO fine-tuning: measured to lose

Six runs on 2026-08-06, all with a healthy critic (explained variance
0.85–0.95). PPO reliably buys ground contact and cut height by **spending
lateral**, while total reward rises the whole way:

| policy | lateral | contact | cut | all gates |
|---|---|---|---|---|
| **student v2** | **0.0794** | 0.8347 | 0.0065 | pass |
| polish model_200 | 0.1453 | 0.8736 | 0.0059 | pass |
| polish model_598 | 0.1816 | 0.9417 | 0.0031 | pass |

PPO is optimising correctly — the reward's optimum simply is not where the
tracking objective is. The reward rebalance (`lateral_fine_scale_m` 0.15 → 0.60,
`lateral` 10 → 12, `contact_loss` −6 → −3) reduced the drift **12–18×** but did
not reverse its sign.

**Default to the distilled student.** Fine-tune only if contact and cut height
matter more than tracking.

---

## Known environment bug: Isaac Sim deadlocks on the second env build

Building a second environment in the same process — `env.close()` then
`gym.make()` again — **deadlocks** in this Isaac Sim build. Observed four times
on 2026-08-06 (null eval, teacher eval, student eval, one training run): process
alive at ~109% CPU, no output for 15–40 minutes, exactly two PhysX construction
lines logged. A single mesh rolls out in ~50 s, so it is not slow work.

Consequences, both already handled:

* **Never loop meshes in one process.** `score_per_mesh.sh` runs one process per
  mesh and merges with `scripts/merge_gates.py`.
* **Silence is not progress.** `evaluate_gates.py` prints every 10% and
  `supervise_training.sh` treats a log that stops growing as failure, killing and
  resuming from the newest checkpoint.

---

## Layout

    gates.py                    the 8-gate contract; no Isaac Lab import, runs anywhere
    rewards.py                  nine bounded terms; marginal rates documented
    env_cfg.py                  Ppo31SoleEnvCfg — terrain, c4 authority, PPO regime
    env.py                      thin subclass; overrides `_get_rewards` only
    terrain_env{,_cfg}.py       c4 command authority over PPO3's verge task
    visualize.sh                viewer, on the terrain actually trained on
    scripts/preflight.py        10 checks incl. P4 feasibility, S5 gradients
    scripts/score_per_mesh.sh   gate scoring, one process per mesh
    scripts/supervise_training.sh  PPO under a stall watchdog
    scripts/training_status.py  run report + charts
    scripts/probe_path_feasibility.py  rate demand per family, CPU only
    tests/                      21 tests; gate thresholds and reward gradients pinned

## Related

* `verge_task` — shared library: terrain, observations, ROM plant, DLS
  teacher, distillation and training scripts.
* `PPO_ID_FD` — the identical task and settings with ID→FD in place of
  ROM-A, for a like-for-like actuator comparison.
