# Stale code archive — 2026-08-17

Follow-up to `archive/README_CLEANUP_2026-08-16.md`, which retired *artifacts*
(checkpoints, outputs, logs). This one retires *code*. **Nothing was deleted** —
every path below was moved with `git mv`, so history follows the file and a
plain `git mv` back restores it.

## Selection criterion

A file was moved only if it satisfied **both** tests:

1. **Zero references** anywhere in the live tree — no import, no invocation by
   another script, no mention in any live `.md`.
2. **Outside the import closure of the three active pipelines.** The closure was
   computed by parsing the AST of every `.py` under `PPO_ROM/`, `PPO_ID_FD/` and
   `PPO_ID_AAM_FD/` and following repo-internal imports transitively to a fixed
   point: **102 files**, of which 49 are the shared `mulag_plant` / `verge_task` /
   `rrlab_assets` libraries. Nothing below is in that set.

Zero-reference alone was **not** treated as staleness. Test files score zero by
construction (pytest collects by convention) and standalone probes are run by
hand, so both were excluded from consideration.

## What was moved

### `extensions/` — foreign-robot task code, 30 files

Vendored IsaacLab template code for robots this project does not use. Untouched
since 2026-05-19/22, and **none of its registered gym IDs are referenced
anywhere in the repo** — not even the MULAG one.

| path | what it is |
|---|---|
| `rrlab_tasks/.../locomotion/` | the Unitree Go2 / ANYmal velocity-tracking stack (24 files) — a quadruped, not a hydraulic manipulator. Registers `RRLAB-Velocity-{Flat,Rough}-Unitree-Go2-v0`. |
| `rrlab_tasks/.../obstacle_avoidance/config/franka/` | Franka arm config. Registers `RRLAB-Obstacle-Avoidance-Franka-v0`. |
| `rrlab_tasks/.../obstacle_avoidance/config/saugbagger/` | Saugbagger config. Registers `RRLAB-Obstacle-Avoidance-Saugbagger-v0`. |
| `rrlab_tasks/.../obstacle_avoidance_env_cfg_backup.py` | a `_backup` copy of a live sibling. |
| `control_policy/` | whole package. Was pip-installed editable from `rrlab_mulag.worktrees/agents-grep-simulation-config-search/` — a **stale worktree path**, not this checkout. Referenced only by its own `setup.py` and `extension.toml`. |

`rrlab_tasks` still imports: `manager_based/__init__.py` uses
`import_packages(__name__)`, which auto-discovers subpackages rather than naming
them, so removing directories cannot break it.

**Deliberately kept:** `extensions/rrlab_assets/rrlab_assets/saugbagger.py`.
It looks equally foreign, but `rrlab_assets/__init__.py` imports it
unconditionally, which puts it **inside the pipeline import closure** — archiving
it would break `import rrlab_assets` for all three pipelines. It stays until
that `__init__` is changed.

### `verge_task/` — PPO3/PPO4 `_sole` lineage drivers, 13 files

Queue and sweep drivers for the *previous* sole-arm lineage. Every run name they
target (`ppo3_sole_rom_direct_q5_v1`, `ppo3_sole_rom_contact_safe_v1`,
`ppo3_sole_rom_headfix_v1`, `ppo3_sole_rom_generalization_v1`,
`ppo3_sole_rom_four_action_recovery_v1`, and the `ppo4_sole_idfd_*` mirrors) was
retired to `archive/checkpoints_2026-08-16/` yesterday. The live lineage is
`ppo3_1_sole` / `ppo4_1_sole`, which none of these scripts mention.

```
scripts/run_contact_loss_diagnosis.sh     scripts/run_headfix_all.sh
scripts/run_contact_safe_gates.sh         scripts/run_headfix_training.sh
scripts/run_contact_safe_v2.sh            scripts/run_plant_ab.sh
scripts/run_direct_q5_queue.sh            scripts/run_queue.sh
scripts/run_four_action_recovery_queue.sh scripts/run_stage6_training.sh
scripts/run_generalization_queue.sh       scripts/run_strict_matrix.sh
visualize_direct_q5.sh
```

**Deliberately kept**, though they drive the same retired lineage — each is
cited by a live document, and a doc that names a script in `archive/` is worse
than a stale script:

| kept | cited by |
|---|---|
| `run_contact_safe_training.sh` | `verge_task/README.md` |
| `run_impact_energy_sweep.sh` | `verge_task/README.md` |
| `run_q5_windup_sweep.sh` | `verge_task/reports/STAGE6_TERRAIN_REPORT.md` |
| `run_teacher_probe.sh` | `verge_task/reports/STAGE6_TERRAIN_REPORT.md` |

`verge_task/train.sh` and `verge_task/visualize.sh` also still default to the old
run names, but they are the **entry points** for the task and were left alone.
Their defaults are stale; the scripts are not.

## What was considered and NOT moved

**AAM stage22–28 tooling** (`run_stage22_v{4,5,6}*.sh`, `run_stage25_*.sh`,
`run_stage26_aam.sh`, `run_stage27_rolling20.sh`, `check_stage22_gate.py`,
`validate_stage22_collection.py`, `validate_stage27_chain.py`,
`report_aam_stage26.py`, `evaluate_stage28_id_transfer.py`). Superseded by
stage29/31, and each has exactly one reference — its own `AAM_module/stage*.md`
lineage document. That is the same "cited by a stage doc" rule under which the
2026-08-16 cleanup **kept** the `aam_stage25` → `stage30` outputs, so keeping
the tools is what makes those docs reproducible. They are superseded, not stale.

## Verification

- `python -m compileall` over the whole remaining tree: clean.
- `pytest PPO_ROM/tests PPO_ID_AAM_FD/tests AAM_module/tests mulag_plant/trajectories/tests`:
  **156 passed**.
- `import rrlab_assets` could not be exercised — it needs Isaac Sim's `pxr`,
  which is absent from a bare interpreter. This is a pre-existing property of the
  environment, not a consequence of the move; the static closure check above is
  what establishes that `rrlab_assets` was left intact.

## Restoring anything

```bash
git mv archive/stale_code_2026-08-17/<path> <path>
```
