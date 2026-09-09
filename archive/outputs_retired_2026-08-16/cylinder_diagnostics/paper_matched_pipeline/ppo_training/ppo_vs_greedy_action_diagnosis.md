# PPO vs Greedy Action Diagnosis

## Offline Checkpoint Analysis

From `model_99.pt` (100 iterations, best PPO):

| Parameter | Value | Interpretation |
|-----------|-------|---------------|
| `distribution.std_param` | [0.111, 0.122] | Moderate exploration noise, NOT collapsed to 0 |
| `mlp.4.bias` (output bias) | [-0.123, -0.006] | Mean action near zero |
| `obs_normalizer.count` | 3200 | Normalized over 3200 observations |

The action mean is essentially zero — the PPO policy produces actions centered
at ~[-0.12, -0.01] with noise std ~[0.11, 0.12].

## Diagnosis

**PPO collapsed to near-zero mean action** (no-op). The policy keeps some
exploration noise but its mean output is near zero. Greedy mapper produces
actions proportional to `(target - ee)`, which are much larger in magnitude.

PPO learned that zero action gives 0.103m distance, and most random actions
make it worse (distance increases to 1.3-3.3m). The small success bonus
(+0.2) and action penalty (-0.01*||a||²) incentivize safe, small actions.

## Root Cause

The reward structure makes zero action the local optimum:
- progress_reward = 3.0 * (prev_dist - current_dist)
- distance_penalty = -1.0 * current_dist
- action_penalty = -0.01 * ||a||²
- success_bonus = +0.2 (if dist < 0.10m)

Most random actions INCREASE distance (negative progress), and the action
penalty compounds this. Zero action produces ~0 progress (arm drifts but
stays around equilibrium), which is better than negative-progress actions.

## Fix

1. Increase progress reward: 3.0 → 10.0
2. Reduce action penalty: 0.01 → 0.001  
3. Reduce success bonus: 0.2 → 0.05
4. Increase entropy coefficient: 0.001 → 0.01
5. Optionally: initialize actor from BC checkpoint

## BC Pretraining

Created `train_stage0_yz_actor_bc.py` to train actor on greedy mapper.
Once Isaac Sim is available, this can be used to initialize PPO.

## Fix Attempt Results

| Fix | Reward | Entropy | Distance (100 iters) | Action Norm | Result |
|-----|--------|---------|---------------------|-------------|--------|
| Original | 3×p, 0.01×a², 0.2×s | 0.001 | 0.104 m | 0.23→0.35 | Matched zero |
| Fixed | 10×p, 0.001×a², 0.05×s | 0.01 | 0.123 m | 0.26→0.52 | More actions, worse distance |
| Fixed+continued | same | 0.01 | 0.103 m | 0.64→0.17 | Action norm collapsed again |

## Conclusion

PPO consistently converges to ~0.103m (zero-action baseline) regardless of reward
tuning. The policy finds the local optimum: produce near-zero actions because
any non-zero action has ~50% chance of moving in the wrong direction (increasing
distance), and the mapper's correction authority is too weak to guarantee
improvement.

The greedy mapper achieves 0.067m because it has perfect knowledge of the
target direction and the J_pinv mapping. PPO must learn this mapping from
noisy observations and reward signals, which requires many more samples to
discover the correct action-sign relationship.

## Recommended Next Step

BC pretraining: initialize PPO actor from greedy-mapper behavior cloning,
then fine-tune with PPO. This gives PPO the correct action direction from
the start, and PPO only needs to refine the magnitude.
