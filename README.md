# MULAG verge mower — RL pipelines

A MULAG FME 500 verge mower. The truck drives on the road at a constant 4 km/h;
the arm reaches over the embankment and the cutter must **follow a line along the
verge** while staying on the ground. Tracking is the objective; the terrain is a
disturbance, not the task.

---

## The two pipelines

Both solve the **same task** with the same terrain, reward, observation (66),
action (4) and eight-gate acceptance contract. They differ in **one thing**: what
converts a cylinder command into motion.

| | pipeline | actuator | status |
|---|---|---|---|
| **[PPO_ROM](PPO_ROM/)** | ROM-A | policy → command ROM → PhysX | **CERTIFIED — all 8 gates, 6 meshes, no thin margins** |
| **[PPO_ID_FD](PPO_ID_FD/)** | ID→FD | policy → ID (target→valve) → FD (valve→Δx) → PhysX | tracks 23% better, **fails ground_contact** |

Start with each pipeline's own README — they carry the results, the commands and
the reasoning.

### Head-to-head (6 held-out meshes, worst-mesh worst-family)

| metric | ROM-A | ID→FD | gate |
|---|---|---|---|
| lateral error | 0.0817 m | **0.0628 m** | ≤ 0.300 |
| ground contact | **0.7854** | 0.6882 **FAIL** | ≥ 0.700 |
| meshes failing | **0 of 6** | 1 of 6 (mesh 555) | — |

ID→FD is the better *tracker*; ROM-A is the only one that meets the *contract*.
Deliverable today is `logs/rsl_rl/ppo3_1_sole/student_distilled_v2.pt`.

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

    PPO_ROM/       ROM-A pipeline. The certified one.
    PPO_ID_FD/     ID/FD pipeline. Subclasses PPO_ROM; only the actuator differs.

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
    id_fd_model/    ID and FD models    -> PPO_ID_FD
      vendor_models/                       model DEFINITIONS (imported)
      best_id_cylinder_control.pt          ID: target -> valve
      best_fd_cylinder_strict.pt           FD: valve -> delta x  (STRICT, not control)

Each folder holds **both** the weights and the model definitions, because the
runtimes import the definitions from beside the weights. Each has its own README
with the traps (the trailing space in two `vendor_models` directory names is
real; the control FD was rank ~2 and under-fit the valve).

### Everything else

    archive/superseded_pipelines/  PPO4 (superseded by PPO_ID_FD), PPO2 ROM-B/
                                   ROM-C, ROM_B_Randomized, Universal_Policy
    archive/model_training_trees/  PPO1_ID_FD_PhysX, ROM-cylinders_pipeline --
                                   where rom_model/ and id_fd_model/ came from
    archive/reference_repos/       teacher_student, DLS_solver, Simscape DR
    archive/handovers/             historical handover and review documents
    logs/rsl_rl/                   checkpoints and TensorBoard runs
    thesis /                       thesis sources

---

## Common commands

All scripts live under `PPO_ROM/scripts/` and take `--task` / `TASK=`
to select the pipeline. There is deliberately **no second copy** for PPO_ID_FD — a
copy would drift, and the moment it did a ROM-vs-ID/FD comparison would be
measuring the drift instead of the actuator.

```bash
# watch the shipped policy
PPO_ROM/visualize.sh

# preflight before committing GPU hours (10 checks)
PPO_ROM/scripts/preflight.py --num_envs 32 --steps 120
PPO_ID_FD/scripts/preflight.py --num_envs 32 --steps 120

# score on the 8 gates (one process per mesh -- see the Isaac Sim note below)
PPO_ROM/scripts/score_per_mesh.sh <name> <ckpt.pt> 480 32
MESHES="1337 17 909 4242 555 2024" \
  PPO_ROM/scripts/score_per_mesh.sh <name> <ckpt.pt> 480 32

# training report and charts
PPO_ROM/scripts/training_status.py --plot
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
