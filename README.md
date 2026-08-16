# MULAG verge mower — RL pipelines

A MULAG FME 500 verge mower. The truck drives on the road at a constant 4 km/h;
the arm reaches over the embankment and the cutter must **follow a line along the
verge** while staying on the ground. Tracking is the objective; the terrain is a
disturbance, not the task.

---

## The active pipelines

The first two solve the same task with the same terrain, reward, 66-value
observation, four-action interface, and eight-gate contract. The AAM pipeline
adds two causal latent values to the policy observation and keeps historical
ID/FD as its safety fallback.

| | pipeline | actuator | status |
|---|---|---|---|
| **[PPO_ROM](PPO_ROM/)** | ROM-A | policy → command ROM → PhysX | **CERTIFIED — all 8 gates, 6 meshes, no thin margins** |
| **[PPO_ID_FD](PPO_ID_FD/)** | ID→FD | policy → ID (target→valve) → FD (valve→Δx) → PhysX | **PROMOTED — six-mesh selection pipeline** |
| **[PPO_ID_AAM_FD](PPO_ID_AAM_FD/)** | causal AAM + ID adapter + FD | policy → AAM context → ID adapter → FD → PhysX | interface verified; nonzero adapter in safe shadow mode |

Start with each pipeline's own README — they carry the results, the commands and
the reasoning.

### Head-to-head (historical comparison)

| metric | ROM-A | ID→FD | gate |
|---|---|---|---|
| lateral error | 0.0817 m | **0.0628 m** | ≤ 0.300 |
| ground contact | **0.7854** | 0.6882 (historical run) | ≥ 0.700 |
| meshes failing | **0 of 6** | 1 of 6 (historical run) | — |

The historical comparison predates the final wide-task ID-FD pipeline. Use
`PPO_ID_FD/README.md` for the current validated commands and artifacts.

---

## Two findings that govern everything here

**1. The task has to be inside the plant's capability.** The cylinders move the
cutter sideways at most `0.004 m / 0.02 s = 0.200 m/s`. At the old settings
**9 of 11 path families demanded more than that**, `boom_sweep` by 3.7×, so error
accumulated by construction and the privileged DLS teacher failed too. No reward,
critic or PPO change can fix a task the plant cannot execute. Terrain is now
*derived* from that ceiling, and preflight check **P4** asserts it every run.

**2. Distillation is the endpoint, not an initialisation.** PPO fine-tuning was
run **seven times** on 2026-08-06, always with a healthy critic (explained
variance 0.85–0.95). It lost every time: it reliably buys ground contact and cut
height by spending lateral, because the reward's optimum is not where the
tracking gate is. The worst case degraded tracking **12.9×**. Prefer the
distilled student; fine-tune only if contact matters more than tracking.

---

## Repository layout

### Pipelines (start here)

    PPO_ROM/         ROM-A pipeline. The certified one.
    PPO_ID_FD/       current ID/FD pipeline.
    PPO_ID_AAM_FD/   causal online AAM + ID adapter + FD pipeline.

### Shared libraries (imported by the pipelines — not pipelines themselves)

    verge_task/       terrain generator, 66-value observation contract,
                           DLS teacher, distillation and training scripts,
                           actuator resolution
    mulag_plant/        base environment (ResidualPolicyEnv), DLS solver,
                           ROM runtime, cylinder adapters

These are **live dependencies**, not legacy. `verge_task`'s task class inherits
from `mulag_plant.policy.env`; both pipelines import both. They are libraries
because they are *shared*, not because they are old.

### Actuator models (loaded at runtime, one folder each)

    rom_model/      ROM-A command ROM   -> PPO_ROM
      ROM_pipeline/                        model DEFINITIONS (put on sys.path)
      plant_reachable_v1_finetune_blend25/best_rom.pt   the deployed ROM
    id_fd_model/    ID and FD models    -> PPO_ID_FD and PPO_ID_AAM_FD
      vendor_models/                       model DEFINITIONS (imported)
      best_id_cylinder_control.pt          ID: target -> valve
      best_fd_cylinder_strict.pt           FD: valve -> delta x  (STRICT, not control)

    outputs/        AAM and latent-adapter checkpoints used by PPO_ID_AAM_FD

Each folder holds **both** the weights and the model definitions, because the
runtimes import the definitions from beside the weights. Each has its own README
with the traps (the trailing space in two `vendor_models` directory names is
real; the control FD was rank ~2 and under-fit the valve).

### Everything else

    archive/superseded_pipelines/  historical and superseded policy pipelines
    archive/model_training_trees/  historical actuator-model training trees
    archive/reference_repos/       teacher_student, DLS_solver, Simscape DR
    archive/handovers/             historical handover and review documents
    logs/rsl_rl/                   checkpoints and TensorBoard runs
    thesis /                       thesis sources

Anything under `archive/`, including the numbered PPO predecessors, is retained
only for provenance and is **not** a current pipeline. The active pipelines are
the three top-level PPO directories listed above.

---

## Common commands

Each pipeline README is the command reference. Shared scripts remain available
for cross-pipeline scoring, while the AAM pipeline also provides task-pinned
wrappers in its own folder.

```bash
# watch the shipped policy
PPO_ROM/visualize.sh

# preflight before committing GPU hours (10 checks)
PPO_ROM/scripts/preflight.py --num_envs 32 --steps 120
PPO_ID_FD/scripts/preflight.py --num_envs 32 --steps 120
PPO_ID_AAM_FD/scripts/preflight.py --num-envs 4 --device cpu

# score on the 8 gates (one process per mesh -- see the Isaac Sim note below)
PPO_ROM/scripts/score_per_mesh.sh <name> <ckpt.pt> 480 32
MESHES="1337 17 909 4242 555 2024" \
  PPO_ROM/scripts/score_per_mesh.sh <name> <ckpt.pt> 480 32

# training report and charts
PPO_ROM/scripts/training_status.py --plot
PPO_ID_AAM_FD/scripts/training_status.py --plot
```

---

## Two environment traps

**Isaac Sim deadlocks on the second environment build in one process.**
`env.close()` then `gym.make()` again wedges — observed four times on
2026-08-06, each with exactly two PhysX construction lines, process alive at
~109% CPU, silent for 15–40 minutes. A single mesh rolls out in ~50 s, so it is
not slow work. **Never loop meshes in one process**; `score_per_mesh.sh` runs one
process per mesh and merges via `merge_gates.py`.

**Silence is not progress.** Long runs print progress every 10%, and
`supervise_training.sh` treats a log that stops growing as failure — killing and
resuming from the newest checkpoint. Before that existed, one hang cost four
hours overnight.

---

## Certifying a result

`gates.py` requires **at least three** meshes, which is enough to detect gross
failure but **not** to certify a thin pass. Ground contact varies **±0.042 across
meshes**; a three-mesh run once reported ID→FD passing contact by +0.0069 and six
meshes showed it failing at 0.6882. Treat any contact margin under ~0.05 as
unresolved until measured on six.
