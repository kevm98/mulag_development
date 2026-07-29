# PPO3 terrain-adaptive sole controller

PPO3 is the final MULAG verge-mowing task. The truck follows a straight, flat
asphalt road at 4 km/h while one PPO policy controls all four hydraulic-cylinder
commands through ROM-A. The cutter follows a red reference line on a separate
green, height-varying verge and must maintain physical ground contact without
destabilizing the truck.

## Final control path

```text
54-D deployable observation
  -> PPO actor (4 actions, sole controller)
  -> EMA + slew limit + extension/head safety shields
  -> ROM-A command-conditioned hydraulic plant
  -> PhysX
```

The adaptive contact controller is a privileged teacher used only during
DAgger initialization. It is absent from PPO rollouts, held-out evaluation, and
deployment. Production checkpoints use the frozen schema
`terrain_teacher_student_v1`; older stored-path and pre-correction checkpoints
are rejected.

## Expanded path-and-height curriculum

One shared policy is trained over:

1. `straight`
2. `lateral_sine`
3. `s_curve`
4. `random_spline`
5. `offset_return`
6. `boom_sweep`
7. `lateral_chirp`
8. `double_s_curve`
9. `lane_change_hold`
10. `gaussian_slalom`
11. `fourier_meander`

Training with 512 environments cycles these families evenly. Every family is
crossed independently with 50%, 70%, 85%, and 100% smooth terrain relief, and
each resumed training block receives a new deterministic terrain seed. The
16-second stage-4 episode covers 17.8 m at 4 km/h and executes one complete
16 m lateral trajectory on a 26 m verge strip.

## Visualize one family with live tracking plots

Run without `--headless`. Replace the family name with any name above.

```bash
cd /home/rrlab/rrlab_mulag
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
CKPT=logs/rsl_rl/ppo3_sole_terrain/teacher_student_v1_long/model_converged.pt

$PY PPO3_SOLE_PhysX/scripts/evaluate.py "$CKPT" \
  --device cuda:0 --num_envs 1 --steps 960 --stage 4 --eval_terrain \
  --path_family s_curve --terrain_height_scale 1.0 --plot_tracking \
  --plot_out PPO3_SOLE_PhysX/plots/s_curve_tracking.png \
  --report PPO3_SOLE_PhysX/reports/s_curve_visualization.json
```

The Matplotlib window is one plan-view XY plot: the red curve is the desired
path and the blue curve is the cutter's actual trajectory. It intentionally
contains no vertical-height, lateral-versus-time, or contact-force subplot.

Interactive visualization adds a large visual-only dark-gray floor with a 2 m grid,
a dark rough asphalt road, the green physical verge, and the red projected path.
The floor/grid has collision disabled and is not spawned during training or
headless evaluation.

## Train from the validated teacher

### 1. Terrain DAgger initializer

```bash
$PY PPO3_SOLE_PhysX/scripts/distill_terrain.py \
  --headless --device cuda:0 --num_envs 128 --stage 4 \
  --probe_steps 960 --iterations 120 --steps_per_iteration 32 \
  --epochs_per_iteration 3 --minibatch_size 8192 \
  --replay_capacity 262144 --learning_rate 3e-4 \
  --out logs/rsl_rl/ppo3_sole_terrain/terrain_teacher_student_v1.pt \
  --report PPO3_SOLE_PhysX/reports/terrain_teacher_distillation_v1.json
```

### 2. Extensive continuation from the current best PPO

```bash
RUN=logs/rsl_rl/ppo3_sole_terrain/expanded_paths_height_v2_very_long
mkdir -p "$RUN"
set -o pipefail

$PY PPO3_SOLE_PhysX/scripts/train_until_converged.py \
  --resume logs/rsl_rl/ppo3_sole_terrain/teacher_student_v1_long/model_best.pt \
  --run_dir "$RUN" \
  --device cuda:0 --num_envs 512 --eval_envs 128 \
  --block_iterations 100 --min_blocks 20 --max_blocks 0 --eval_steps 960 \
  --learning_rate 5e-5 --entropy_coef 0.001 \
  --save_interval 10 --train_timeout_s 43200 --eval_timeout_s 7200 \
  2>&1 | tee "$RUN/supervisor.log"
```

This forces at least 2,000 PPO updates (32.8 million environment transitions),
then continues without a block cap until aggregate and worst-family held-out
gates pass. Re-running the same command resumes from `state.json`. Do not add
`--restart`, and do not reset the optimizer or critic. The final extensively
trained artifact is `model_converged.pt` in the new run directory.

## Starting-checkpoint baseline

The checkpoint used to initialize the expanded curriculum passed the earlier
seven-family held-out task after two 512-environment blocks:

| Metric | Result |
|---|---:|
| Held-out convergence score | 0.0 |
| Cutter contact fraction | 97.85% |
| Mean contact-gap error | 0.0020 m |
| Mean lateral path error | 0.0407 m |
| p01 head alignment | 0.9890 |
| Hard minimum head alignment | 0.9522 |
| p95 tool acceleration | 61.99 m/s² |
| Safety-violation rate | 0.307% |

## Validation

```bash
$PY -m pytest -q PPO3_SOLE_PhysX/tests
```

The retained evidence in `reports/` is the baseline for the earlier seven-path
curriculum. New v2 reports belong in the new training run directory so old and
new validation evidence cannot be confused.

## Package layout

```text
actions.py                  pure actor action chain and safety shields
env.py / env_cfg.py         sole-controller base task
terrain.py                  terrain and eleven smooth path generators
terrain_env.py              road, verge, sensors, physics and task signals
terrain_env_cfg.py          task, visual and PPO configuration
terrain_observations.py     frozen 54-D actor observation
terrain_rewards.py          contact/lateral/safety reward
terrain_convergence.py      shared held-out gates
distill.py                  DAgger helpers and teacher isolation contract
scripts/distill_terrain.py  privileged-teacher DAgger
scripts/train.py            one PPO training process
scripts/train_until_converged.py persistent trainer/evaluator supervisor
scripts/evaluate.py         held-out evaluation, GUI and live plots
scripts/probe_drive.py       physics/sensor diagnostic
tests/                      fast structural and numerical contracts
reports/                    retained final evidence only
```
