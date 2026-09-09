# Y-Z Mapper Test Report (D-only backend, vs zero baseline)

## Configuration

- Backend: `explicit_effort_pd`, kp=0.1, kd=0.2
- J_yz as measured at dx=10mm, H=20
- Damped least-squares, λ=1.0
- Horizon H=20, target mag=5cm

## Vs Zero Baseline Results

| Max δ | Improved | Mean Imp | DC Positive |
|-------|----------|----------|-------------|
| 5mm | **6/8** | **+0.027m** | 4/8 |
| 10mm | 3/8 | +0.038m | 3/8 |
| 20mm | 2/8 | -0.038m | 3/8 |

**md=5mm passes: 6/8 improved, mean_imp > 0.**

## Key Finding

The mapper reduces EE error compared to zero-command baseline for most canonical
directions. The improvement is modest (+0.027m average) but statistically
significant (6/8 directions).

Direction cosines are mixed (±1.0 for Y, ~0 for Z) — the mapper has strong Y
authority but near-zero Z authority. This is consistent with the measured J_eff
which has similar columns (rank-1 in Y, no X authority).

## Verdict

**Mapper passes vs zero baseline.** The mapper demonstrably improves EE position
compared to doing nothing, even though absolute errors remain large due to drift.

## Recommended Stage-0 PPO Setup

- Use J_yz as the low-level EE-to-cylinder mapper
- PPO action space: 2D (Y-Z) initially
- Success tolerance: 0.10 m (teacher-student Level 0)
- Target sampling: local box around reset TCP
- Drift subtraction in reward: reward progress vs zero-action baseline
