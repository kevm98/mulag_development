# PPO-ID(AAM)-FD wide verge task

PPO is the sole arm controller:
`policy -> causal AAM -> frozen ID + latent adapter -> frozen FD -> PhysX`. The
truck drives straight at one constant speed sampled per episode from 2--7 km/h;
the cutter follows a 26 m white line over stage-6 terrain.

The policy has 68 observations and four desired cylinder-motion outputs; it
never emits valve commands. The AAM reads the observable history
`[arm q, arm qdot, measured dP, previous valve command]`, estimates `z_hat` at
20 Hz, and conditions both the policy and the inverse model. FD is deliberately
not conditioned. Invalid or slow inference falls back to frozen ID.

Current policy -- all eight gates on meshes 1337/17/909 at 4 km/h:

```text
logs/rsl_rl/ppo_id_aam_fd/headroom_priced_v1/model_299.pt
```

Requires these frozen runtime models:

```text
outputs/stage31_rom/aam_cnn/L20_d64/aam.pt
outputs/stage31_rom/id_adapter_cnn_l20_s5/zhat_adapter.pt
id_fd_model/best_id_cylinder_control.pt
id_fd_model/best_fd_cylinder_strict.pt
```

## Visualize

`--unimog_speed` pins one constant truck speed; omit it to sample 2--7 km/h per
episode. `--trajectory-shape` sets the cutter's white line, not the truck path
(11 shapes, listed in `PPO_ROM/path_shapes.py`); or `--trajectory-shapes all`.

```bash
cd /home/rrlab/rrlab_mulag
PPO_AAM_FD/visualize.sh logs/rsl_rl/ppo_id_aam_fd/headroom_priced_v1/model_299.pt \
  --unimog_speed 7 --trajectory-shape double_s_curve --num-envs 4 --mesh 17 --steps 2808
```

Hardest gated case: mesh 17 has the tallest hill, `double_s_curve` the most
lateral travel. `--steps 2808` is the full 26 m path; the script's default of
960 stops a third of the way along. Close the Isaac window to stop.

## Verify

CPU-only, no simulator -- checks the causal warm-up, reset isolation, bounded
adapter, and frozen-ID fallback:

```bash
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
$PY PPO_AAM_FD/scripts/preflight.py --num-envs 4 --device cpu
$PY PPO_AAM_FD/scripts/minimum_online_adaptation_check.py --device cpu --num-envs 2 --ticks 72
```

## Train

```bash
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
$PY PPO_AAM_FD/scripts/train.py \
  --resume logs/rsl_rl/ppo_id_aam_fd/headroom_priced_v1/model_299.pt \
  --stages 6 --stage_iterations 400 --num_envs 256 --seed 17 \
  --vehicle_speed_min_kmh 2.0 --vehicle_speed_max_kmh 7.0 \
  --learning_rate 1e-5 --fixed_learning_rate --entropy_coef 0.0 \
  --reset_optimizer_on_resume --reset_critic_head_on_resume \
  --freeze_actor_iterations 100 --obs_normalizer freeze \
  --online_aam_residual_scale 0.25 --save_interval 25 \
  --log_dir logs/rsl_rl/ppo_id_aam_fd/my_aam_run --headless
```

Each environment keeps its sampled speed until reset; fast ones hold at 26 m
with zero reward until the batch completes. The actor is frozen for the first
100 iterations so the critic adapts before any policy update.

## Score

```bash
MESHES="1337 17 909" SPEED_MIN_KMH=2 SPEED_MAX_KMH=7 \
  PPO_AAM_FD/scripts/score_per_mesh.sh final \
  logs/rsl_rl/ppo_id_aam_fd/headroom_priced_v1/model_299.pt 2808 32
```

Score the full 2808-step path. Selecting on a short window hides
`ground_contact` failures that only appear later along the 26 m line.
