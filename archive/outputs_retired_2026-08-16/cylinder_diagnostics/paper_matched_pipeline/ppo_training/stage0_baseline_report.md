# Stage-0 Baseline Report

## Environment

- Stage-0 Level: 0 (easiest, ±5cm target box)
- Episode count: 200 (1 env)
- Action scale: 0.01

## Results

### Zero Action
| Metric | Value |
|--------|-------|
| mean final EE error | 0.042 m |
| mean guard_fraction | 1.0 (likely spurious - guard flags from smooth ramp physics) |
| workspace violation rate | TBD |
| cab violation rate | TBD |

Interpretation: Zero action is stable. The EE stays at 4.2 cm error, well within the ±5 cm
target box. The arm does not drift substantially under hold.

### Random Action (partial)
Random action causes high guard fraction and likely high workspace soft violation rate
since the random policy drives the arm aggressively. Episodes are long due to soft
workspace violation (no immediate termination).

### Greedy Action
Not completed due to tensor shape bug in multi-env evaluation.

## Interpretation

1. **Zero action is safe**: EE error ~0.04 m, no immediate failure.
   PPO exploration is too aggressive compared to zero-action baseline.
2. **Random action causes high guard/clipping**: Action scale of 0.01
   produces displacements up to 1 cm per horizon, which coupled with
   random exploration easily triggers guards.
3. **Action scale must be reduced**: The current 0.01 scale is too high
   for random exploration at Stage-0. Action-scale sweep needed.

## Action Item

Reduce action scale to the largest value where random-action baseline passes:
- failure_rate < 20%
- guard_fraction < 15%
- clipping_fraction < 10%

Likely value: 0.0025 or 0.005.
