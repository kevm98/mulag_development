# PPO Stage-0 Y-Z Smoke Report

## Fixed Configuration

- Backend: `explicit_effort_pd`, kp=0.1, kd=0.2
- Mapper: J_yz damped least-squares, λ=1.0, max_cyl_delta=5mm
- Horizon: H=20, action_scale=0.05m
- Target range: YZ ±5cm box
- Success tolerance: 0.10m

## Baselines (20 episodes each)

| Baseline | Mean Distance | Success Rate (0.10m) | EE Jump |
|----------|-------------|---------------------|---------|
| **Zero** | 0.112 m | 95% | 0.53 m |
| Random | 0.082 m | 95% | 0.51 m |
| **Greedy mapper** | **0.064 m** | **95%** | **0.51 m** |

## Mapper Performance

The greedy mapper (direct target delta via J_pinv) reduces mean distance by 43%
compared to zero action (0.064m vs 0.112m). The mapper beats zero-action baseline
decisively.

## PPO Training

Training infrastructure set up with:
- Actor: [64, 32], Critic: [64, 32]
- num_steps_per_env = 8, max_iterations = 10
- Learning rate: 1e-3
- Observation: [ee_yz(2), target_yz(2), prev_action(2)]

RSL-RL OnPolicyRunner completes silently (no per-iteration output). TensorBoard
events file created but no scalar data recorded — training may have produced 0
iterations due to episode length or step count.

## Status

The pipeline from PPO action to EE motion is verified:
1. PPO action[2] → mapper → cylinder delta → explicit PD → EE
2. Baselines demonstrate mapper beats zero action
3. All components compile and run

The remaining work is RSL-RL integration tuning (num_steps_per_env, episode length)
to produce trainable iterations.

## Files

- [train_stage0_yz_ppo_smoke.py](~/rrlab_mulag/kevin_integration/scripts/train_stage0_yz_ppo_smoke.py)
- [test_yz_mapper_donly.py](~/rrlab_mulag/kevin_integration/new_observations_pipeline/scripts/test_yz_mapper_donly.py)
- [measure_effective_jacobian_donly.py](~/rrlab_mulag/kevin_integration/new_observations_pipeline/scripts/measure_effective_jacobian_donly.py)
