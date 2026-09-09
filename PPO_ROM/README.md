# PPO-ROM wide verge task

PPO is the sole arm controller: `policy -> cylinder-command ROM -> PhysX`. The
truck drives straight at one constant speed sampled per episode from 2--7 km/h;
the cutter follows a 26 m white line over stage-6 terrain.

Promoted policy -- all eight gates at 2/3/4/5/6/7 km/h on meshes 1337/17/909.
Stable copy of `model_7364.pt`, which resumed runs overwrite:

```text
logs/rsl_rl/ppo3_1_sole/speed_range_2_7_pathfix_v2_rom/model_promoted_2_7kmh.pt
```

## Trajectory shapes

The cutter's tracked white line, not the truck path. Eleven families, defined as
`TRACKING_PATH_FAMILIES` in `verge_task/terrain.py` and re-exported by
`PPO_ROM/path_shapes.py`:

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

The order is load-bearing: each environment is assigned
`TRACKING_PATH_FAMILIES[(env_index + offset) % 11]`, so a run with eleven or
more environments covers every family deterministically.

## Visualize

`--unimog_speed` pins one constant truck speed; omit it to sample 2--7 km/h per
episode. `--trajectory-shape` sets the cutter's white line, not the truck path
(11 shapes, listed in `PPO_ROM/path_shapes.py`); or `--trajectory-shapes all`.

```bash
cd ~/rrlab_mulag
PPO_ROM/visualize.sh logs/rsl_rl/ppo3_1_sole/speed_range_2_7_pathfix_v2_rom/model_promoted_2_7kmh.pt \
  --unimog_speed 7 --trajectory-shape double_s_curve --num-envs 4 --mesh 17 --steps 2808
```

Hardest gated case: mesh 17 has the tallest hill, `double_s_curve` the most
lateral travel. Close the Isaac window to stop.

## Train

```bash
PY=~/miniconda3/envs/env_isaaclab/bin/python
$PY PPO_ROM/scripts/train.py \
  --resume logs/rsl_rl/ppo3_1_sole/speed_range_2_7_pathfix_v2_rom/model_promoted_2_7kmh.pt \
  --stages 6 --stage-iterations 300 --num-envs 128 --trajectory-shapes all \
  --vehicle_speed_min_kmh 2.0 --vehicle_speed_max_kmh 7.0 \
  --trajectory_regularized --trajectory_local_coef 0.10 \
  --trajectory_max_update_kl 0.01 --trajectory_max_reference_kl 0.05 \
  --trajectory_actor_learning_rate 1e-5 --trajectory_critic_learning_rate 1e-4 \
  --reset_optimizer_on_resume --reset_critic_head_on_resume \
  --freeze_actor_iterations 100 --reset_actor_optimizer_on_resume \
  --gamma 0.97 --lam 0.95 --fixed_std 0.15 --entropy_coef 0.001 \
  --obs_normalizer freeze --save-interval 25 \
  --log_dir logs/rsl_rl/ppo3_1_sole/my_rom_run --headless
```

Each environment keeps its sampled speed until reset; fast ones hold at 26 m
with zero reward until the batch completes.

