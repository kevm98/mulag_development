# PPO-(AAM)-ID-FD stale material — 2026-08-19

Moved with `git mv` (plain `mv` for the six untracked files), never deleted.
Same selection rule as `archive/README_CLEANUP_2026-08-16.md` and
`archive/stale_code_2026-08-17/README.md`.

**A file was moved only if BOTH held:**

1. zero references from the live tree outside its own cluster — code, scripts
   and live `.md` alike;
2. outside the transitive import closure of `PPO_ROM`, `PPO_ID_FD`,
   `PPO_ID_AAM_FD`.

Tests and `__init__.py` score zero by convention, so none were moved. Gate after
the move: `pytest PPO_ROM/tests PPO_ID_AAM_FD/tests AAM_module/tests
mulag_plant/trajectories/tests` — 156 passed, unchanged from before.

## What moved, and why

### `PPO_ID_AAM_FD/experiments/` (whole directory)

Superseded by `experiments/E1..E5`, which the top-level `experiments/README.md`
declares the current structure ("material predating this structure is in
`legacy/`"). Two independent reasons:

- **Five files were byte-identical duplicates of live code.** `diff` reports no
  difference against `experiments/E3/code/` (`analyze_adaptation_trace.py`,
  `run_e3_plant_change.sh`) and `experiments/E4/code/`
  (`run_openloop_e2e.sh`, `run_openloop_seeds.sh`,
  `summarise_openloop_seeds.py`). The E-tree copies are the ones the E READMEs
  invoke.
- **The rest is a closed, self-referential Phase-era cluster.** Every reference
  to `factor_recon_sweep`, `mass_observability`, `latent_width_ceiling`,
  `delay_identification`, `evaluate_zhat_identification`, the `run_phase*`
  scripts and both `HANDOVER_*.md` files came from another file inside the same
  directory. Nothing in the thesis, `reports/`, `deployment_adaptation/` or
  `AAM_module/` cites any of them.

Its `README.md` was also factually stale: it lists "Experiment 2 — in-episode
dynamics change" as planned with "**no** supporting code — nothing changes
$\theta$ mid-episode yet", but E3 has since run exactly that experiment to a
CLOSED verdict, and `PPO_ID_AAM_FD/plant_perturbation.py` is the code it says
does not exist.

`results/` moved with the code: those JSON/log files are the outputs of the
archived probes, not of any live experiment.

### `PPO_ID_AAM_FD/scripts/blend_actor_checkpoints.py`

Actor-interpolation utility for the `latent_teacher_blends_4129_4148` campaign.
Zero references anywhere in the live tree; the campaign is over.

### `PPO_ID_AAM_FD/reports/selection.json`

Actively misleading, not merely superseded. It records
`"selected_iteration": 375, "promotable": true` for
`stage31_promotable_v2/model_375.pt`, scored over the 480-step selection window
(8.9 m of the 26 m path). The full-path rescore in
`experiments/E1/results/fullpath_reselection.json` refutes it: **every**
`stage31_promotable_v2` checkpoint fails the `ground_contact` gate over the
whole path. The current checkpoint is
`logs/rsl_rl/ppo_id_aam_fd/headroom_priced_v1/model_299.pt`, which is what
`experiments/E3`, `E4` and `E5` actually run.

### `PPO_ID_AAM_FD/reports/speed_stress/`

Nineteen per-mesh JSON reports for superseded checkpoints (`model_375`, `474`,
`650`, `675`, `700`, `724`, `725`, `749`, and a teacher row). No citation in the
thesis, `reports/`, `experiments/` or `deployment_adaptation/`.

## Live-tree edits this required

One real cross-dependency existed and was repointed rather than kept:

- `experiments/E4/code/run_openloop_seeds.sh:78` invoked
  `$repo/PPO_ID_AAM_FD/experiments/summarise_openloop_seeds.py`. It now calls
  its own byte-identical local copy, `experiments/E4/code/summarise_openloop_seeds.py`,
  which makes E4 self-contained.
- `experiments/E4/code/run_openloop_e2e.sh:29` had a usage comment naming the
  old path; updated to `experiments/E4/code/run_openloop_e2e.sh`.

`PPO_ID_AAM_FD/reports/` was left empty and removed; `evaluate_gates.py` writes
reports with `mkdir(parents=True, exist_ok=True)`, so both `score_per_mesh.sh`
and `select_checkpoint.sh` recreate their default output directories on demand.

## What deliberately stayed

`env.py`, `env_cfg.py`, `online_runtime.py`, `plant_perturbation.py`,
`trace_recorder.py`, `__init__.py`, all of `tests/`, and the remaining
`scripts/` tooling. `plant_perturbation.py` and `trace_recorder.py` look
orphaned to a filename grep but are imported as modules — by the tests and by
`verge_task/terrain_env.py` respectively. This is the same trap as
`saugbagger.py` in the 2026-08-17 cleanup: run the closure check, not a
name-plausibility judgement.
