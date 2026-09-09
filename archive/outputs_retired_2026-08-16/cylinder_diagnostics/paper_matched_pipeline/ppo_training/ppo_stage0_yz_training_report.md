# PPO Stage-0 Y-Z Training Report

## Configuration

### Low-level backend
```
lowlevel_actuator_backend = "explicit_effort_pd"
kp_scale = 0.1, kd_scale = 0.2
velocity_filter_beta = 0.8, force_rate_limit = 0.01
```

### Mapper
- Fixed J_yz damped least-squares, λ=1.0, max_cyl_delta=5mm
- J_pinv precomputed from measured J_eff

### PPO Interface
- Action dim: 2 (Y, Z EE deltas in base frame)
- Action scale: 0.05 m
- Observation dim: 6 (ee_yz[2], target_yz[2], prev_action[2])
- Max episode length: 50 horizon steps
- Success tolerance: 0.10 m

### Target Sampling
- Y range: [-0.05, +0.05] m
- Z range: [-0.02, +0.02] m
- Sampled relative to settled EE position

### Reward
```
distance = ||target_yz - ee_yz||
progress = prev_distance - distance
reward = 3.0 * progress - 1.0 * distance - 0.01 * ||action||^2 + 0.2 * success
```

### PPO Config (RSL-RL)
- Actor: [64, 32], Critic: [64, 32]
- num_steps_per_env: 8, max_iterations: 10-50
- Learning rate: 1e-3

## Baselines (20 episodes each, H=20)

| Baseline | Mean Distance | Success@0.10m | EE Jump |
|----------|-------------|--------------|---------|
| **Zero action** | 0.103 m | 95% | 0.53 m |
| Random action | 0.084 m | 95% | 0.51 m |
| **Greedy mapper** | **0.067 m** | **95%** | **0.53 m** |

Greedy mapper beats zero action by 35% (0.067m vs 0.103m).

## PPO Training

Training infrastructure is set up and compiles. RSL-RL OnPolicyRunner runs without
errors. TensorBoard events file is created. However, per-iteration scalar logging
was not observed — likely due to RSL-RL's logging configuration or the small
num_steps_per_env=8 producing rollouts that don't log.

The pipeline from PPO action → mapper → cylinder delta → explicit PD → EE is
verified end-to-end through the baseline evaluation.

## Stage Progression

| Stage | Description | Status |
|-------|------------|--------|
| A | Deterministic rollout check | Script created (check_ppo_stage0_yz_env.py) |
| B | Tiny PPO smoke (20 iters) | Infrastructure ready, needs RSL-RL logging fix |
| C | Small PPO (100-300 iters) | Pending |
| D | Useful PPO (500-1000 iters) | Pending |

## Pass Gates

| Gate | Metric | Current |
|------|--------|---------|
| 1. PPO beats zero | mean_dist < 0.103 m | Pending |
| 2. PPO beats random | mean_dist < 0.084 m | Pending |
| 3. PPO approaches greedy | mean_dist < 0.067 m | Pending |

## Files

- [train_stage0_yz_ppo_smoke.py](~/rrlab_mulag/kevin_integration/scripts/train_stage0_yz_ppo_smoke.py) — PPO trainer
- [check_ppo_stage0_yz_env.py](~/rrlab_mulag/kevin_integration/new_observations_pipeline/scripts/check_ppo_stage0_yz_env.py) — env diagnostic
- [test_yz_mapper_donly.py](~/rrlab_mulag/kevin_integration/new_observations_pipeline/scripts/test_yz_mapper_donly.py) — mapper test
- [measure_effective_jacobian_donly.py](~/rrlab_mulag/kevin_integration/new_observations_pipeline/scripts/measure_effective_jacobian_donly.py) — J_eff measurement

## TODO

1. Fix RSL-RL per-iteration scalar logging to confirm training iterations
2. Run Stage B (20 iterations, 1 env) to verify no NaN/crash
3. Run Stage C (100-300 iterations, 4 envs) to beat zero baseline
4. Run Stage D (500-1000 iterations, 8-16 envs) to beat random baseline
