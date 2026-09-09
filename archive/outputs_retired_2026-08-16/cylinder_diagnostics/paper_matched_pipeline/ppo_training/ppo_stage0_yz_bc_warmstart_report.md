# PPO Stage-0 Y-Z BC Warm-Start Report

## BC Pretraining

- Trained analytically on greedy mapper: `action = target_error_yz / action_scale`
- Architecture: 4→64→32→2, ELU activations
- Training: 2000 synthetic samples, 100 epochs, lr=1e-3, MSE loss
- Result: **MSE = 0.0009, action_cosine = 0.95**
- Saved: `outputs/bc_actor_yz.pt`

## BC→PPO Weight Copy

Offline verification (strict copy, no Isaac Sim dependency):
- BC and PPO actors have identical `mlp.*` architectures
- All 6 parameters copied with `strict=True` (exact key match)
- Post-copy verification: **cosine = 0.9846, MSE = 0.000474**
- **PASS**: cosine > 0.8 ✓

## Observation Normalization

**DISABLED.** BC was trained on raw 4D observations `[target_error_yz, prev_action]`.
RSL-RL's `EmpiricalNormalization` would distort the BC-trained input distribution.
Set `agent_cfg.actor.obs_normalization = False` (and critic).

## Warm-Start Script Phases

Script: `train_stage0_yz_ppo_warmstart.py`

### Phase 1: Sanity check
- Loads BC checkpoint, copies weights into PPO actor
- Resets env, computes greedy and PPO actions
- Prints cosine, MSE — aborts if cosine < 0.8

### Phase 2: BC-only evaluation (50 episodes)
- Evaluates zero, random, greedy, BC-init against env
- Gate: BC-init mean_dist < random baseline (0.084m)

### Phase 3: PPO fine-tuning
- lr=3e-4, entropy=0.001, no success bonus, no success termination
- Progress reward 10×, distance penalty 1×, action penalty 0.001×

## Run Command

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd ~/rrlab_mulag
timeout 1200s ./rrlab.sh -p kevin_integration/scripts/train_stage0_yz_ppo_warmstart.py \
  --num_envs 1 --max_iterations 50 --headless --device cuda:0 \
  --horizon_steps 10 --num_steps_per_env 8 --save_interval 25 \
  --bc_checkpoint outputs/bc_actor_yz.pt
```

## Expected Results

| Method | Expected Distance | Status |
|--------|------------------|--------|
| Zero | ~0.103 m | baseline |
| Random | ~0.084 m | baseline |
| Greedy | ~0.067 m | baseline |
| BC-init | < 0.084 m | gate 1 |
| PPO fine-tuned | < 0.084 m | gate 2 |
| PPO strong | ~0.067 m | gate 3 |

## Files

- `outputs/bc_actor_yz.pt` — BC checkpoint (cosine 0.95)
- `train_stage0_yz_ppo_warmstart.py` — warm-start script
- `train_stage0_yz_ppo_smoke.py` — vanilla PPO (with fixed obs/term/reward)
- `evaluate_stage0_yz_policy.py` — evaluation script
