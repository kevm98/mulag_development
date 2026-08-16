# PPO-ID(AAM)-FD wide verge task

PPO is the sole arm controller. The truck drives at one constant speed sampled
per episode from 2--7 km/h, and the cutter tracks a 26 m white line over
stage-6 terrain.

```text
policy -> causal AAM -> frozen ID + latent adapter -> frozen FD -> PhysX
```

The policy has 68 observations and four desired cylinder-motion outputs. It
does not output valve commands. The causal AAM uses the observable history
`[arm q, arm qdot, measured dP, previous valve command]`, estimates `z_hat`,
and updates at 20 Hz. The promoted estimator is a causal 1-D CNN over the
latest 20 ROM rows (1.0 s); its five-prediction output smoother adds only
250 ms. The PPO plant contract also keeps an 8 mm solver-enforced c3
extension reserve while scoring headroom against the original stroke.
Invalid or slow inference falls back to frozen ID.

```bash
cd /home/rrlab/rrlab_mulag
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
CKPT=logs/rsl_rl/ppo_id_aam_fd/stage31_promotable_v2/model_375.pt
RUN=logs/rsl_rl/ppo_id_aam_fd/stage31_promotable_v2
```

Required runtime models:

```text
outputs/stage31_rom/aam_cnn/L20_d64/aam.pt
outputs/stage31_rom/id_adapter_cnn_l20_s5/zhat_adapter.pt
id_fd_model/best_id_cylinder_control.pt
id_fd_model/best_fd_cylinder_strict.pt
```

## Visualize

One shape in six environments:

```bash
PPO_ID_AAM_FD/visualize.sh "$CKPT" \
  --trajectory-shape s_curve --num-envs 6 --mesh 1337 --steps 2808
```

Several shapes in six environments:

```bash
PPO_ID_AAM_FD/visualize.sh "$CKPT" \
  --trajectory-shapes straight,s_curve,boom_sweep \
  --num-envs 6 --mesh 17 --steps 2808
```

All shapes in eleven environments:

```bash
PPO_ID_AAM_FD/visualize.sh "$CKPT" \
  --trajectory-shapes all --num-envs 11 --mesh 1337 --steps 2808
```

Available shapes:

```text
straight              lateral_sine       s_curve
random_spline         offset_return       boom_sweep
lateral_chirp         double_s_curve      lane_change_hold
gaussian_slalom       fourier_meander
```

The visualizer uses the same natural road, grass, soil, and markings as
PPO-ROM. Close Isaac Sim to stop.

During a randomized rollout each environment keeps its sampled speed until
reset. A fast environment stops at 26 m and is unscored while slower
environments finish; the batch resets only after all paths complete.

## Verify the pipeline

```bash
$PY PPO_ID_AAM_FD/scripts/preflight.py --num-envs 4 --device cpu

$PY PPO_ID_AAM_FD/scripts/minimum_online_adaptation_check.py \
  --device cpu --num-envs 2 --ticks 72

$PY PPO_ID_AAM_FD/scripts/smoke.py \
  --meshes 1337 --num-envs 1 --steps 8
```

The CPU checks validate the causal warm-up, reset isolation, bounded adapter,
and frozen-ID fallback. The smoke only checks Isaac/PhysX initialization and
stepping; it is not a policy-quality result.

## Train or continue training

Once the CPU preflight and minimum online check pass, run the longer PPO
fine-tune against the frozen CNN/ID/FD stack:

```bash
PPO_ID_AAM_FD/scripts/promotion_finetune.sh \
  logs/rsl_rl/ppo_id_aam_fd/stage31_promotable_v2/model_375.pt \
  logs/rsl_rl/ppo_id_aam_fd/promotion_finetune_l20_cnn
```

The default is 1500 stage-6 iterations. Override with
`PPO_FINETUNE_ITERATIONS` or `PPO_FINETUNE_ENVS` for a declared resource
budget; the script refuses to run if the AAM artifact is not CNN/L20.

The selected checkpoint already includes the completed deployment warm-up. A
speed-range run still performs a fresh critic-only adaptation: the actor is
held fixed for 100 PPO iterations, then the saved critic warm-up checkpoint is
followed by actor updates.

```bash
$PY PPO_ID_AAM_FD/scripts/train.py \
  --resume "$CKPT" --stages 6 --stage_iterations 400 \
  --num-envs 256 --seed 17 \
  --vehicle_speed_min_kmh 2.0 --vehicle_speed_max_kmh 7.0 \
  --learning_rate 1e-5 --fixed_learning_rate --entropy_coef 0.0 \
  --reset_optimizer_on_resume --reset_critic_head_on_resume \
  --freeze_actor_iterations 100 --obs_normalizer freeze \
  --online_aam_residual_scale 0.25 --save_interval 25 \
  --log_dir logs/rsl_rl/ppo_id_aam_fd/continued_run --headless
```

For a fresh 68-D policy initialized from the validated ID-FD policy:

```bash
$PY PPO_ID_AAM_FD/scripts/migrate_idfd_checkpoint.py \
  logs/rsl_rl/ppo4_1_sole/idfd_promotable_v1/promoted_policy.pt \
  logs/rsl_rl/ppo_id_aam_fd/idfd_init/model_init.pt \
  --zhat outputs/stage31_rom/aam_cnn/L20_d64/zhat_all.json
```

Resume the migrated checkpoint with `--reset_optimizer_on_resume`.

## Score and select

Score one checkpoint on the six held-out meshes:

```bash
MESHES="1337 17 2024 4242 555 909" \
SPEED_MIN_KMH=2 SPEED_MAX_KMH=7 \
REPORT_ROOT=PPO_ID_AAM_FD/reports/score \
  PPO_ID_AAM_FD/scripts/score_per_mesh.sh final "$CKPT" 2808 32
```

The ordered campaign performs the same final sweep for the promoted AAM-FD
checkpoint: fixed `unimog_speed` 2, 3, 4, 5, 6, and 7 km/h. Reports and the checkpoint
summary are under
`experiments/speed_range_2_7_pathfix_v2/post_training_eval/summary.json`.

Select checkpoints after their mesh reports are complete:

```bash
MESHES="1337 17 2024 4242 555 909" \
REQUIRED_MESHES=6 REPORT_ROOT=PPO_ID_AAM_FD/reports/selection \
  PPO_ID_AAM_FD/scripts/select_checkpoint.sh \
  "$RUN" aam_promotable 100 375 550
```

Training status and drift diagnosis:

```bash
PPO_ID_AAM_FD/scripts/training_status.sh --run "$RUN" --plot

PPO_ID_AAM_FD/scripts/monitor_training_drift.sh \
  --run id_aam_fd="$RUN" \
  --output "$RUN/drift_monitor.jsonl"
```

The selected full six-mesh checkpoint is `model_375.pt`.

## Promotion gate

After fine-tuning, collect identical estimated/zero/shuffled PPO rollouts and
run the single promotion command:

```bash
PPO_ID_AAM_FD/scripts/evaluate_promotion_ablation.sh \
  logs/rsl_rl/ppo_id_aam_fd/promotion_finetune_l20_cnn/model_<final>.pt
```

Promotion is allowed only if: (1) true and estimated SD latents improve
dynamics, (2) estimated AAM retains the true-latent advantage, and (3) the
estimated PPO latent beats both zero and shuffled latents on all paired
continuous metrics (lateral error, cut-height error, impacts, acceleration,
and headroom). A failed criterion leaves the verdict unpromoted.
