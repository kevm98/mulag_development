# MULAG verge mower — learned hydraulics and online adaptation

A MULAG FME 500 verge mower on a Unimog carrier. The truck drives the road at a
constant speed; the arm reaches over the embankment and the cutter follows a line
along the verge. Tracking is the objective, terrain is the disturbance.

Reinforcement-learning pipelines, the learned actuator models they act through,
the experiments, and the thesis that reports them. All evidence is
simulation-to-simulation; nothing here is a claim about hardware.

## Setup

```bash
PY=~/miniconda3/envs/env_isaaclab/bin/python
```

Pinned, do not upgrade: `torch==2.7.0`, `numpy==1.26.0`, `rsl-rl-lib==5.0.1`.
Import RSL-RL config classes from `rrlab_compat.rsl_rl`, never `isaaclab_rl`.

## The three stacks

| | stack | command path |
|---|---|---|
| **R** | [PPO_ROM](PPO_ROM/README.md) | policy → command ROM → PhysX |
| **IF** | [PPO_ID_FD](PPO_ID_FD/README.md) | policy → inverse → forward model → PhysX |
| **AF** | [PPO_AAM_FD](PPO_AAM_FD/README.md) | policy → valve command → forward model → PhysX |

R and IF differ only in the actuator path. AF emits the valve command instead of
a cylinder increment, so it is not scored against them.

**Each pipeline README holds that stack's full commands, including training.**

## Run

```bash
cd ~/rrlab_mulag

# watch a shipped policy
PPO_ROM/visualize.sh logs/rsl_rl/ppo3_1_sole/speed_range_2_7_pathfix_v2_rom/model_promoted_2_7kmh.pt \
  --unimog_speed 7 --trajectory-shape double_s_curve --num-envs 4 --mesh 17 --steps 2808

# preflight before spending GPU hours
$PY PPO_ROM/scripts/preflight.py --num_envs 32 --steps 120

# score on the eight gates, one process per mesh
MESHES="1337 17 909" PPO_ROM/scripts/score_per_mesh.sh <name> <ckpt.pt> 2808 32

# build the thesis
cd thesis/thesis/hydraulic_manipulator_thesis && latexmk -pdf main.tex
```

## Layout

    PPO_ROM/ PPO_ID_FD/ PPO_AAM_FD/   the three stacks
    verge_task/ mulag_plant/          terrain, environment, DLS teacher
    rrlab_compat/                     RSL-RL config classes
    learned_models/                   the learned actuator models
      rom_model/                        command ROM        -> PPO_ROM
      id_fd_model/                      inverse + forward  -> PPO_ID_FD, PPO_AAM_FD
      sd_model/                         state dynamics. Not in the thesis.
    experiments/                      one folder per Chapter 7 section
    reports/ outputs/                 pre-registrations, scored reports
    AAM_module/                       datasets; results use the _frozen copies
    logs/rsl_rl/                      runs and checkpoints
    thesis/                           sources and figure scripts
    archive/                          retired material

Read a folder's own README before touching it.

**`archive/` is load-bearing**, despite the name: the Chapter 4/5/6 figure
scripts read `model_training_trees/` and `ppo_rom_retired_2026-08-16/`, and
`ppo_rom_stage6_tools/` is executed by `rebuild_rom_idfd_stage6.sh`. Grep before
deleting anything there.
