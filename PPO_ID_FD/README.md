# PPO-ID-FD wide task

PPO is the sole arm controller: `policy -> ID (target -> valve) -> FD (valve ->
motion) -> PhysX`. Truck, terrain, observations, rewards, path shapes, gates and
the PPO update are identical to PPO-ROM -- only the actuator differs.

Promoted policy -- all eight gates at 2/3/4/5/6/7 km/h on meshes 1337/17/909.
Stable copy of `model_2322.pt`, which resumed runs overwrite:

```text
logs/rsl_rl/ppo4_1_sole/speed_range_2_7_pathfix_v2_idfd/model_promoted_2_7kmh.pt
```

## Visualize

`--unimog_speed` pins one constant truck speed; omit it to sample 2--7 km/h per
episode. `--trajectory-shape` sets the cutter's white line, not the truck path
(11 shapes, listed in `PPO_ROM/path_shapes.py`); or `--trajectory-shapes all`.

```bash
cd ~/rrlab_mulag
PPO_ID_FD/visualize.sh logs/rsl_rl/ppo4_1_sole/speed_range_2_7_pathfix_v2_idfd/model_promoted_2_7kmh.pt \
  --unimog_speed 7 --trajectory-shape double_s_curve --num-envs 4 --mesh 17 --steps 2808
```

The hardest gated case: mesh 17 has the tallest hill (0.53 m), `double_s_curve`
the most lateral travel. Close the Isaac window to stop.

## Train

```bash
PY=~/miniconda3/envs/env_isaaclab/bin/python
PPO_ID_FD/scripts/train.sh \
  --resume logs/rsl_rl/ppo4_1_sole/speed_range_2_7_pathfix_v2_idfd/model_promoted_2_7kmh.pt \
  --stages 6 --stage-iterations 600 --num-envs 128 --trajectory-shapes all \
  --vehicle_speed_min_kmh 2.0 --vehicle_speed_max_kmh 7.0 \
  --vehicle_speed_sampling active_time_balanced \
  --trajectory_regularized --trajectory_local_coef 0.10 \
  --trajectory_max_update_kl 0.05 --trajectory_max_reference_kl 0.10 \
  --trajectory_actor_learning_rate 3e-6 --trajectory_critic_learning_rate 1e-4 \
  --reset_optimizer_on_resume --reset_critic_head_on_resume \
  --freeze_actor_iterations 0 --reset_actor_optimizer_on_resume \
  --gamma 0.97 --lam 0.95 --fixed_std 0.05 --entropy_coef 0.001 \
  --obs_normalizer freeze --save-interval 25 \
  --log_dir logs/rsl_rl/ppo4_1_sole/my_idfd_run --headless
```

`active_time_balanced` samples faster speeds more often, since a fixed-length
path keeps them active for fewer steps than slow ones.
