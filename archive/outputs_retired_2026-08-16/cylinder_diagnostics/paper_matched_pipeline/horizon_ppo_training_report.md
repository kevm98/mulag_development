# Horizon PPO Training Report

**Date**: 2026-07-01

---

## 1. Pre-PPO Validation

See: `horizon_pre_ppo_validation_report.md`

- 92.6% horizon sign accuracy in full pipeline sweep (1140 trials)
- 8/8 FD sign correct — FD always predicts correct direction
- 0.40m EE drift over 0.83s zero-command baseline
- Single-cylinder tests confirm pipeline functionality

## 2. PPO Action Definition

```
action[0:4] in [-1, 1]
delta_x_des_H = action * horizon_action_scale_m
```

| Parameter | Value |
|-----------|-------|
| horizon_action_scale_m | 0.01 m (conservative) |
| horizon_steps | 10 (0.167s at 60Hz) |
| fd_horizon_gain | 1.0 (characterized) |
| tracking_gain | 0.0 |

## 3. Reward Terms

| Term | Weight | Purpose |
|------|--------|---------|
| EE error improvement | +w_improve | Positive reward for reducing EE error |
| Final EE error | -w_final | Penalize remaining error |
| Action magnitude | -w_action | Penalize large displacements |
| Clipping fraction | -w_clip | Penalize joint limit clipping |
| Oscillation index | -w_osc | Penalize oscillatory motion |
| Hydraulic consistency | -w_hydraulic | Penalize measured vs predicted deviation |

## 4. Training Stages

**Stage 1** (current):
- Static reachable target
- cylinder_displacement action
- num_envs: 64-128
- max_iterations: 200
- action_scale: 0.01 m

**Stage 2** (next):
- Increased target offset and action scale
- max_iterations: 1000

**Stage 3** (future):
- EE displacement action + DLS mapper

## 5. Safety Rules

Training script `train_upper_level_ppo_horizon_idfd.py` enforces:
- Refuses to run if position_ramp_pi enabled
- Refuses to run if ID disabled
- Refuses to run if FD disabled
- Validates fine-tuned ID checkpoint exists

## 6. Training Command

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab && cd ~/rrlab_mulag
timeout 3600s python kevin_integration/scripts/train_upper_level_ppo_horizon_idfd.py \
  --num_envs 128 --max_iterations 200 --headless \
  --horizon_steps 10 --horizon_action_scale_m 0.01 \
  --fd_horizon_gain 1.0 --tracking_gain 0.0
```

## 7. Remaining Limitations

1. Gravity bias on cylinders 1,2 — PPO can learn coordinated compensation
2. No running tracking error integral — uses single-step prev error
3. Model dt (0.05s) differs from physics dt (0.0167s)
4. Per-cylinder fd_gain not yet calibrated individually
