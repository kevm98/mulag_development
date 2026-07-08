# Stage-1 Adapter Pipeline: Execution Plan

## Quick Reference

Execute these commands in sequence. **Do not skip validation gates.**

---

## Phase 1: Data Collection and Model Training

### 1.1 Collect Dataset (~30-60 min)

```bash
cd ~/rrlab_mulag

./rrlab.sh -p kevin_integration/new_observations_pipeline/scripts/ppo1_adapter/collect_adapter_dataset.py \
  --num_envs 128 \
  --num_steps 2000 \
  --max_delta_cylinder 0.0025 \
  --action_smoothing_beta 0.85 \
  --action_rate_limit 0.08 \
  --settle_steps 30 \
  --headless \
  --device cuda:0 \
  --backend explicit_effort_pid_bias \
  --kp_scale 0.1 \
  --kd_scale 0.5 \
  --bias_calibration_time_s 5.0 \
  --output_dir outputs/ppo1_adapter \
  --seed 42
```

**Check**: `outputs/ppo1_adapter/adapter_dataset.pt` exists and has > 150k samples

---

### 1.2 Train Forward Model (~5-10 min)

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  kevin_integration/new_observations_pipeline/scripts/ppo1_adapter/train_forward_response_model.py \
  --dataset_path outputs/ppo1_adapter/adapter_dataset.pt \
  --output_dir outputs/ppo1_adapter \
  --hidden_dim 64 \
  --num_layers 2 \
  --batch_size 256 \
  --num_epochs 50 \
  --lr 1e-3 \
  --device cuda:0
```

**Check**: Best val MAE < 10mm, sign accuracy > 85%, R² > 0.7

---

### 1.3 Train Adapter (~5-10 min)

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  kevin_integration/new_observations_pipeline/scripts/ppo1_adapter/train_task_to_cylinder_adapter.py \
  --dataset_path outputs/ppo1_adapter/adapter_dataset.pt \
  --forward_checkpoint outputs/ppo1_adapter/forward_response_model_best.pt \
  --output_dir outputs/ppo1_adapter \
  --hidden_dim 64 \
  --num_layers 2 \
  --batch_size 256 \
  --num_epochs 50 \
  --lr 1e-3 \
  --lambda_consistency 1.0 \
  --lambda_command 0.01 \
  --lambda_limit 0.1 \
  --device cuda:0
```

**Check**: `outputs/ppo1_adapter/task_to_cylinder_adapter_best.pt` exists

---

## Phase 2: Validation Gates

### Gate 1: Offline Validation (~1 min)

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  kevin_integration/new_observations_pipeline/scripts/ppo1_adapter/validate_adapter_offline.py \
  --dataset_path outputs/ppo1_adapter/adapter_dataset.pt \
  --forward_checkpoint outputs/ppo1_adapter/forward_response_model_best.pt \
  --adapter_checkpoint outputs/ppo1_adapter/task_to_cylinder_adapter_best.pt \
  --device cuda:0
```

**Must show**: ✓ PASS

**If FAIL**: Stop. Improve dataset or model.

---

### Gate 2: Direction Tests (~5-10 min)

```bash
./rrlab.sh -p kevin_integration/new_observations_pipeline/scripts/ppo1_adapter/debug_adapter_direction_tests.py \
  --adapter_checkpoint outputs/ppo1_adapter/task_to_cylinder_adapter_best.pt \
  --num_envs 64 \
  --num_tests_per_direction 20 \
  --ppo_action_scale 0.02 \
  --headless \
  --device cuda:0 \
  --backend explicit_effort_pid_bias \
  --kp_scale 0.1 \
  --kd_scale 0.5 \
  --bias_calibration_time_s 5.0 \
  --settle_steps 30
```

**Must show**: ✓ PASS (Axis: 4/4, Diagonal: 3/4, Safety: pass)

**If FAIL**: Stop. Adapter directions incorrect. Fix adapter.

---

### Gate 3: Greedy Evaluation (~10-15 min)

```bash
./rrlab.sh -p kevin_integration/new_observations_pipeline/scripts/ppo1_adapter/evaluate_adapter_greedy.py \
  --adapter_checkpoint outputs/ppo1_adapter/task_to_cylinder_adapter_best.pt \
  --num_envs 64 \
  --num_episodes 50 \
  --max_steps 200 \
  --greedy_gain 0.5 \
  --max_delta_ee 0.02 \
  --trajectory_mode sequence_easy \
  --target_hold_steps 50 \
  --headless \
  --device cuda:0 \
  --backend explicit_effort_pid_bias \
  --kp_scale 0.1 \
  --kd_scale 0.5 \
  --bias_calibration_time_s 5.0 \
  --settle_steps 30
```

**Must show**: ✓ PASS (Dist < 100mm, Success > 80%, Safety pass)

**If FAIL**: Stop. Adapter insufficient for tracking. Do not proceed to PPO.

---

## Phase 3: PPO Training (ONLY if all gates pass)

### 3.1 Train PPO (~30-60 min)

```bash
./rrlab.sh -p kevin_integration/scripts/train_stage1_adapter_ppo.py \
  --adapter_checkpoint outputs/ppo1_adapter/task_to_cylinder_adapter_best.pt \
  --num_envs 128 \
  --max_iterations 50 \
  --headless \
  --device cuda:0 \
  --trajectory_mode sequence_easy \
  --target_hold_steps 50 \
  --backend explicit_effort_pid_bias \
  --kp_scale 0.1 \
  --kd_scale 0.5 \
  --bias_calibration_time_s 5.0 \
  --ppo_action_scale 0.02 \
  --adapter_max_delta_cylinder 0.0025 \
  --action_smoothing_beta 0.8 \
  --action_rate_limit 0.1 \
  --reward_mode distance_only \
  --ppo_lr_scale 0.05 \
  --ppo_clip_param 0.05 \
  --ppo_entropy_coef 0.001 \
  --ppo_num_learning_epochs 2 \
  --save_interval 5 \
  --experiment_name ppo1_adapter_sequence_easy
```

**Check during training**:
- Mean distance should decrease
- Success rate should increase
- Distance should NOT stay > 1m for many iterations
- If stuck: Stop and debug adapter/lower controller

**Output**: `logs/rsl_rl/ppo1_adapter_sequence_easy/model_best.pt`

---

### 3.2 Evaluate PPO (~15-20 min)

```bash
./rrlab.sh -p kevin_integration/new_observations_pipeline/scripts/ppo1_adapter/evaluate_stage1_adapter_ppo.py \
  --checkpoint logs/rsl_rl/ppo1_adapter_sequence_easy \
  --adapter_checkpoint outputs/ppo1_adapter/task_to_cylinder_adapter_best.pt \
  --num_envs 64 \
  --num_episodes 100 \
  --max_steps 200 \
  --trajectory_mode sequence_easy \
  --headless \
  --device cuda:0 \
  --backend explicit_effort_pid_bias \
  --kp_scale 0.1 \
  --kd_scale 0.5 \
  --ppo_action_scale 0.02
```

**Success criteria**:
- PPO beats zero
- PPO beats random
- PPO matches or beats adapter_greedy
- Safety metrics pass

---

## Quick Diagnostics

### Check dataset quality
```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python -c "
import torch
data = torch.load('outputs/ppo1_adapter/adapter_dataset.pt')
print(f'Samples: {data[\"metadata\"][\"accepted_samples\"]}')
print(f'Acceptance: {data[\"metadata\"][\"acceptance_rate\"]*100:.1f}%')
"
```

### Check model outputs
```bash
ls -lh outputs/ppo1_adapter/*.pt
```

### Check PPO training logs
```bash
ls -lh logs/rsl_rl/ppo1_adapter_sequence_easy/
```

---

## Timeline Estimate

- Data collection: 30-60 min
- Forward model training: 5-10 min
- Adapter training: 5-10 min
- Gate 1 (offline): 1 min
- Gate 2 (direction): 5-10 min
- Gate 3 (greedy): 10-15 min
- PPO training: 30-60 min
- PPO evaluation: 15-20 min

**Total**: ~2-3 hours for full pipeline

---

## Stop Conditions

**STOP and debug if**:
- Gate 1 fails → Improve dataset/model
- Gate 2 fails → Fix adapter directions
- Gate 3 fails → Improve adapter before PPO
- PPO distance > 1m after 10 iterations → Debug adapter/controller
- PPO does not beat greedy → Accept greedy as best; investigate PPO design

**DO NOT**:
- Skip validation gates
- Tune PPO blindly if adapter is broken
- Claim success without external evaluation
- Continue if direction tests fail
