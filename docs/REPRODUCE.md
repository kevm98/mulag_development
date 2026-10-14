# Reproduce the MULAG results

Run commands from the repository root. Each pipeline README contains its full training, visualisation, and evaluation contract: [R](../PPO_ROM/README.md), [IF](../PPO_ID_FD/README.md), and [AF](../PPO_AAM_FD/README.md).

## Environment

The project uses Isaac Sim and Isaac Lab with the pinned Python packages below. The examples assume the repository is at `~/rrlab_mulag` and the environment's Python is at `~/miniconda3/envs/env_isaaclab/bin/python`.

```bash
cd ~/rrlab_mulag
PY=~/miniconda3/envs/env_isaaclab/bin/python
```

Keep `torch==2.7.0`, `numpy==1.26.0`, and `rsl-rl-lib==5.0.1` pinned. Import RSL-RL config classes from `rrlab_compat.rsl_rl`, not `isaaclab_rl`.

## Watch a shipped R policy

```bash
PPO_ROM/visualize.sh logs/rsl_rl/ppo3_1_sole/speed_range_2_7_pathfix_v2_rom/model_promoted_2_7kmh.pt \
  --unimog_speed 7 --trajectory-shape double_s_curve --num-envs 4 --mesh 17 --steps 2808
```

## Preflight before a longer run

```bash
$PY PPO_ROM/scripts/preflight.py --num_envs 32 --steps 120
```

## Score across the eight gates

Run one mesh per process. The evaluation script can hang when several meshes are supplied in one process; the wrapper also provides a stall watchdog.

```bash
MESHES="1337 17 909" PPO_ROM/scripts/score_per_mesh.sh <name> <ckpt.pt> 2808 32
```

## Repository map

| Path | Contents |
|---|---|
| [`PPO_ROM/`](../PPO_ROM/) · [`PPO_ID_FD/`](../PPO_ID_FD/) · [`PPO_AAM_FD/`](../PPO_AAM_FD/) | The three control stacks. |
| [`verge_task/`](../verge_task/) · [`mulag_plant/`](../mulag_plant/) | Terrain, environment, and DLS teacher. |
| [`rrlab_compat/`](../rrlab_compat/) | RSL-RL configuration classes. |
| [`learned_models/`](../learned_models/) | ROM, inverse/forward, and state-dynamics models. The state-dynamics model is outside the thesis. |
| [`experiments/`](../experiments/) | Chapter 7 experiment code, results, and figures. |
| [`reports/`](../reports/) · [`outputs/`](../outputs/) | Pre-registrations and scored reports. |
| [`AAM_module/`](../AAM_module/) | Datasets; thesis results use the `_frozen` copies. |
| [`logs/rsl_rl/`](../logs/rsl_rl/) | Runs and checkpoints. |
| [`archive/`](../archive/) | Retired material, including files still used by figure scripts. |

Read a folder's README before changing it. **Do not delete `archive/` wholesale**: Chapter 4–6 figure scripts read `model_training_trees/` and `ppo_rom_retired_2026-08-16/`, and `ppo_rom_stage6_tools/` is used by `rebuild_rom_idfd_stage6.sh`.
