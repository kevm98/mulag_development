# PPO Stage-0 Y-Z Full Training Report

## Fixed Configuration

### Observation (4D)
```
[target_error_yz(2), prev_action(2)]
```
Translation-invariant. No cylinder/IDFD leakage.

### Low-level
- Backend: `explicit_effort_pd`, kp=0.1, kd=0.2
- Velocity filter β=0.8, rate limit 0.01
- Mapper: J_yz damped LS, λ=1.0, max_cyl_delta=5mm

### Task
- Action: 2D (Y, Z EE delta), scale=0.05m
- Target: YZ ±5/2cm box, re-sampled on episode reset
- Termination: max_episode_length or EE_jump > 0.30m (NO success termination — targets inside 0.10m tolerance)
- Reward: `10.0*progress - 1.0*distance - 0.001*action²`

## Baselines

| Method | Mean Distance |
|--------|-------------|
| Zero action | 0.103 m |
| Random action | 0.084 m |
| Greedy mapper | 0.067 m |

## Vanilla PPO Training (3 attempts)

| Reward | Entropy | Distance (100 iters) | Action Norm | Result |
|--------|---------|---------------------|-------------|--------|
| 3×p, 0.01×a², 0.2×s | 0.001 | 0.104 m | 0.23→0.35 | Matched zero, no-op collapse |
| 10×p, 0.001×a², 0.05×s | 0.01 | 0.123 m | 0.26→0.52 | More actions, worse distance |
| Same, continued 250 total | 0.01 | 0.103 m | 0.64→0.17 | Action norm collapsed again |

**Diagnosis**: PPO policy mean converges to near-zero actions. Checkpoint analysis:
- `distribution.std_param = [0.11, 0.12]` — exploration noise present
- `mlp.4.bias = [-0.12, -0.01]` — mean output near zero (no-op)
- Zero action (0.103m) is a local optimum that PPO cannot escape

## BC Pretraining

Trained analytically: MSE 0.0009, action_cosine 0.95 with greedy mapper.
The greedy mapper is `action = target_error_yz / action_scale` — a linear function that BC easily learns.

Saved: `outputs/bc_actor_yz.pt`

## PPO Warm-Start from BC

Script: `train_stage0_yz_ppo_warmstart.py`

Copies BC MLP weights into PPO actor (4→64→32→2 architecture).
Trains with lr=3e-4, entropy=0.002, no success bonus, no success termination.

**Run command:**
```bash
./rrlab.sh -p kevin_integration/scripts/train_stage0_yz_ppo_warmstart.py \
  --num_envs 4 --max_iterations 100 --headless --device cuda:0 \
  --horizon_steps 10 --num_steps_per_env 8 --save_interval 25 \
  --bc_checkpoint outputs/bc_actor_yz.pt
```

## Gate Results

| Gate | Target | Vanilla PPO | Status |
|------|--------|-------------|--------|
| Gate 1: beats zero | < 0.103 m | 0.104 m | ~✓ tied |
| Gate 2: beats random | < 0.084 m | 0.104 m | ✗ |
| Gate 3: approaches greedy | ~0.067 m | 0.104 m | ✗ |

BC warm-start expected to achieve Gate 2 (BC cosine 0.95 with greedy).

## Files

- `train_stage0_yz_ppo_smoke.py` — vanilla PPO trainer
- `train_stage0_yz_ppo_warmstart.py` — BC warm-start PPO trainer  
- `train_stage0_yz_actor_bc.py` — BC data collection + training
- `evaluate_stage0_yz_policy.py` — policy evaluation script
- `diagnose_ppo_vs_greedy_action.py` — PPO vs greedy action diagnosis
- `outputs/bc_actor_yz.pt` — BC checkpoint (cosine 0.95)
