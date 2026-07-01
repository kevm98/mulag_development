# New Observations Pipeline

Horizon displacement control with learned hydraulic ID/FD models.

## Selected Low-Level Controller

**current_horizon_idfd_ff**

```
delta_cmd_H = fd_gain[c] * delta_fd_H
```

| Parameter | Value |
|-----------|-------|
| fd_gain_per_cylinder | [2.0, 3.0, 3.0, 3.0] |
| tracking_gain | 0.0 |
| horizon_steps | 10 |
| horizon_action_scale_m | 0.01 m |
| ID model | Fine-tuned (97.2% sign accuracy) |
| FD model | Original sequence FD (88% sign accuracy) |

## PPO Action

```
action[0:4] in [-1, 1]
→ delta_x_des_H = action * horizon_action_scale_m
→ closed-loop ID/FD rollout
→ delta_fd_H
→ fd_gain[c] * delta_fd_H
→ smooth PhysX ramp
→ PhysX
```

## Directory Structure

```
new_observations_pipeline/
├── README.md
├── learned_models/         # Model loading and inference
│   ├── velocity_to_valve_id.py
│   ├── scalers.py
│   └── checkpoints.md
├── low_level_controller/   # Deterministic horizon displacement logic
│   ├── controller_config.py
│   ├── idfd_rollout.py
│   ├── horizon_idfd_controller.py
│   └── horizon_gain_calibration.py
├── ppo1/                   # PPO environment adapters
│   ├── horizon_idfd_env_wrapper.py
│   ├── observations_adapter.py
│   ├── actions_adapter.py
│   └── rewards_adapter.py
├── tasks/                  # Task-specific references
└── scripts/                # Runnable test/calibration scripts
```

## Commands

All commands run from `~/rrlab_mulag` with the `env_isaaclab` conda environment.

### Smoke test (PPO import + 1 iteration)

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd ~/rrlab_mulag

./rrlab.sh -p kevin_integration/scripts/train_upper_level_ppo_horizon_idfd.py \
  --num_envs 1 \
  --max_iterations 1 \
  --headless \
  --device cuda:0 \
  --horizon_steps 10 \
  --horizon_action_scale_m 0.01 \
  --fd_horizon_gain_per_cylinder "2.0,3.0,3.0,3.0"
```

### Laptop training (safe RTX 3060 6 GB baseline)

```bash
./rrlab.sh -p kevin_integration/scripts/train_upper_level_ppo_horizon_idfd.py \
  --num_envs 4 \
  --max_iterations 200 \
  --headless \
  --device cuda:0 \
  --horizon_steps 10 \
  --horizon_action_scale_m 0.01 \
  --fd_horizon_gain_per_cylinder "2.0,3.0,3.0,3.0"
```

### Lab GPU training (increase only after smoke tests)

```bash
./rrlab.sh -p kevin_integration/scripts/train_upper_level_ppo_horizon_idfd.py \
  --num_envs 16 \
  --max_iterations 1000 \
  --headless \
  --device cuda:0 \
  --horizon_steps 10 \
  --horizon_action_scale_m 0.01 \
  --fd_horizon_gain_per_cylinder "2.0,3.0,3.0,3.0"
```

### Play checkpoint

```bash
./rrlab.sh -p kevin_integration/scripts/play_upper_level_ppo_horizon_idfd.py \
  --num_envs 1 \
  --headless \
  --device cuda:0 \
  --checkpoint <PATH_TO_CHECKPOINT> \
  --horizon_steps 10 \
  --horizon_action_scale_m 0.01 \
  --fd_horizon_gain_per_cylinder "2.0,3.0,3.0,3.0"
```

### Low-level validation

```bash
./rrlab.sh -p kevin_integration/new_observations_pipeline/ppo1/test_horizon_displacement_pipeline.py \
  --num_envs 1 --headless --device cuda:0
```

### Gain calibration

```bash
./rrlab.sh -p kevin_integration/new_observations_pipeline/ppo1/calibrate_current_horizon_idfd_ff.py \
  --num_envs 1 --headless --device cuda:0
```

## Rules

- **Do not** use position_ramp_pi for PPO (diagnostic only).
- **Do not** use horizon_idfd_ff_pi unless future metrics prove it improves both sign and displacement ratio.
- **Current selected controller**: current_horizon_idfd_ff.
- Per-step velocity control failed and was replaced by horizon displacement control.
- PPO is real only when RSL-RL `runner.learn()` is running.
