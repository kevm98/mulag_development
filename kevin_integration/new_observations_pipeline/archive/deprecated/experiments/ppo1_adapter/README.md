# Stage-1 Adapter Pipeline: Hierarchical Control with Learned Task-to-Cylinder Bridge

## Overview

This pipeline implements correct hierarchical control for the Mulag hydraulic manipulator:

```
Target Y-Z trajectory
    ↓
PPO upper-level controller
    ↓
desired EE motion [ΔY_des, ΔZ_des]
    ↓
learned task-to-cylinder adapter (frozen)
    ↓
desired cylinder increment [Δcyl₁, Δcyl₂, Δcyl₃, Δcyl₄]
    ↓
lower-level cylinder controller (PID + effort bias)
    ↓
PhysX plant
```

### Key Architecture Decisions

1. **PPO action**: Task-space EE motion (2D), NOT cylinder commands
2. **Adapter**: Learned inverse model trained in IsaacLab
3. **Separation**: PPO learns strategy; adapter handles kinematics; lower controller handles actuation
4. **Validation gates**: Strict progression through offline → direction tests → greedy → PPO

## Why This Architecture

See [RESEARCH_LITERATURE_REVIEW.md](RESEARCH_LITERATURE_REVIEW.md) for full justification.

**Summary**:
- Hierarchical control is standard for hydraulic manipulators
- Task-space PPO enables better generalization and transfer
- Learned adapter avoids Simscape/Isaac coordinate frame mismatches
- Forward-consistency loss ensures physically plausible predictions
- Frozen adapter during PPO simplifies credit assignment

## Pipeline Execution

### Prerequisites

- IsaacLab environment configured
- Conda environment: `env_isaaclab`
- GPU: CUDA device
- Workspace: `~/rrlab_mulag`

### Step-by-Step Execution

#### 1. Collect Adapter Dataset

Collect paired (cylinder_command, EE_motion) samples in IsaacLab:

```bash
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

**Expected output**: `outputs/ppo1_adapter/adapter_dataset.pt`

**Validation**:
- Accepted samples: > 150,000
- Acceptance rate: > 60%
- Sign balance: +Y/-Y and +Z/-Z should be roughly equal

#### 2. Train Forward Response Model

Train forward model: (state, command) → EE motion

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
  --val_split 0.2 \
  --device cuda:0
```

**Expected output**: `outputs/ppo1_adapter/forward_response_model_best.pt`

**Pass criteria**:
- Val MAE Y < 10mm
- Val MAE Z < 10mm
- Sign accuracy Y > 85%
- Sign accuracy Z > 85%
- R² Y > 0.7
- R² Z > 0.7

#### 3. Train Task-to-Cylinder Adapter

Train adapter with supervised + forward-consistency loss:

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
  --val_split 0.2 \
  --device cuda:0
```

**Expected output**: `outputs/ppo1_adapter/task_to_cylinder_adapter_best.pt`

#### 4. Validation Gate 1: Offline Metrics

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  kevin_integration/new_observations_pipeline/scripts/ppo1_adapter/validate_adapter_offline.py \
  --dataset_path outputs/ppo1_adapter/adapter_dataset.pt \
  --forward_checkpoint outputs/ppo1_adapter/forward_response_model_best.pt \
  --adapter_checkpoint outputs/ppo1_adapter/task_to_cylinder_adapter_best.pt \
  --device cuda:0
```

**Pass criteria**:
- Forward model MAE < 10mm, sign accuracy > 85%, R² > 0.7
- Adapter forward-consistency MAE < 10mm
- Adapter sign accuracy > 90%
- Command saturation < 5%
- Limit violations < 1%

**If FAIL**: Improve dataset coverage or model capacity. Do NOT proceed.

#### 5. Validation Gate 2: Direction Tests

One-step closed-loop validation in Isaac:

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

**Pass criteria**:
- Axis tests: 4/4 pass (+Y, -Y, +Z, -Z correct signs)
- Diagonal tests: 3/4 pass
- Mean jump < 30mm
- Max jump < 80mm

**If FAIL**: Adapter does not produce correct motion directions. Fix adapter/dataset. Do NOT proceed.

#### 6. Validation Gate 3: Greedy Evaluation

Multi-step tracking with greedy proportional controller:

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

**Pass criteria**:
- Mean distance < 100mm
- Success@0.10 > 80%
- Mean jump < 30mm
- Max jump < 80mm
- Beats zero baseline by > 25%

**If FAIL**: Adapter insufficient for tracking. Improve adapter before PPO. Do NOT proceed.

#### 7. Train PPO with Frozen Adapter

Only if all validation gates pass:

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

**Expected output**: `logs/rsl_rl/ppo1_adapter_sequence_easy/model_best.pt`

**Good training trends**:
- Mean distance decreases
- Success@0.10 increases
- PPO action norm stable (not saturated)
- Adapter output norm stable
- Cylinder limit violations < 5%

**Bad trends**:
- Distance stays > 1m
- Success near zero
- Progress always negative
- Adapter saturates
- Cylinder limits hit frequently

**If bad**: Stop training. Debug adapter or lower controller. Do NOT tune PPO hyperparameters blindly.

#### 8. External PPO Evaluation

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

**Pass criteria**:
- PPO beats zero baseline
- PPO beats random baseline
- PPO matches or beats adapter_greedy
- Mean distance < 100mm
- Success@0.10 > 80%
- Mean jump < 30mm
- Max jump < 80mm
- Cylinder limits < 5%

**If PPO does not beat greedy**:
- Report that adapter_greedy is the current best controller
- Do NOT claim PPO improved tracking
- Investigate PPO observation/reward design

## Success Criteria

Pipeline is successful when:

1. ✓ Adapter offline validation passes
2. ✓ Direction tests pass (4/4 axis, 3/4 diagonal)
3. ✓ Greedy adapter beats zero and old Jacobian
4. ✓ PPO with frozen adapter externally tracks targets
5. ✓ PPO does not output cylinder commands
6. ✓ Final path is: PPO [ΔY, ΔZ] → adapter → cylinder_target → lower controller → PhysX

## Architecture Verification

Confirm these are true:

- [ ] PPO action_dim = 2 (not 4)
- [ ] PPO action = [desired_delta_ee_y, desired_delta_ee_z]
- [ ] Adapter input includes state (ee, cylinders, velocities)
- [ ] Adapter output = [Δcyl₁, Δcyl₂, Δcyl₃, Δcyl₄]
- [ ] Adapter is frozen during PPO training
- [ ] No pinned J_pinv in final runtime path
- [ ] Lower controller is explicit_effort_pid_bias
- [ ] ID/FD models are bypassed (for now)

## Next Steps After Success

After all validation gates pass and PPO trains successfully:

1. **Compare**: PPO vs greedy vs old Jacobian vs zero
2. **Document**: Real metrics from all evaluation stages
3. **Analyze**: Where does PPO add value beyond greedy?
4. **Integrate ID/FD**: Optionally improve lower-level actuator tracking
5. **Extend**: Train on more complex trajectories
6. **Transfer**: Test on hardware (if available)

## Troubleshooting

### Dataset collection fails
- Check: Environment resets properly
- Check: Cylinder limits not hit immediately
- Increase: settle_steps
- Decrease: max_delta_cylinder

### Forward model validation fails
- Increase: Dataset size or coverage
- Increase: Model capacity (hidden_dim, num_layers)
- Check: Normalization statistics

### Adapter validation fails
- Increase: lambda_consistency weight
- Improve: Forward model first
- Check: Cylinder limits are set correctly

### Direction tests fail
- Check: Adapter signs in offline validation
- Check: Lower controller gains (kp_scale, kd_scale)
- Increase: bias_calibration_time_s

### Greedy evaluation fails
- Check: greedy_gain is reasonable
- Check: max_delta_ee is reasonable
- Check: Direction tests passed first

### PPO training fails
- Do NOT proceed if validation gates failed
- Check: Adapter is frozen
- Check: PPO observation includes target error
- Check: Reward signal is reasonable
- Start with low learning rate (ppo_lr_scale=0.05)

## Files

### Core Modules
- `forward_response_model.py`: Forward model (state, command) → EE motion
- `task_to_cylinder_adapter.py`: Adapter (desired EE, state) → cylinder command
- `stage1_adapter_env_wrapper.py`: RL environment wrapper with frozen adapter

### Scripts
- `collect_adapter_dataset.py`: Data collection in Isaac
- `train_forward_response_model.py`: Train forward model
- `train_task_to_cylinder_adapter.py`: Train adapter
- `validate_adapter_offline.py`: Gate 1 validation
- `debug_adapter_direction_tests.py`: Gate 2 validation
- `evaluate_adapter_greedy.py`: Gate 3 validation
- `train_stage1_adapter_ppo.py`: PPO training (main script)
- `evaluate_stage1_adapter_ppo.py`: External PPO evaluation

### Documentation
- `RESEARCH_LITERATURE_REVIEW.md`: Literature justification
- `README.md`: This file
- `EXECUTION_PLAN.md`: Step-by-step execution guide
