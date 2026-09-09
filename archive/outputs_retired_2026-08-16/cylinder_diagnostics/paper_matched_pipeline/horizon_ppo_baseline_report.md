# Horizon PPO Baseline Report

**Date**: 2026-07-02

---

## Baselines

### Zero action baseline
- PPO sends `action = zeros(4)` → no displacement commanded
- Expected: arm drifts under gravity, no active motion toward target

### Random horizon baseline  
- PPO sends `action = uniform(-1, 1)` → random cylinder displacements
- Expected: arm moves randomly, occasional progress toward target by chance

### Scripted horizon-IDFD baseline
- Scripted controller sends fixed `delta_x_des_H` per cylinder
- Uses the same ID/FD rollout and smooth ramp as PPO
- Expected: deterministic, sign-correct displacement

## Metrics to Compare

| Metric | Zero | Random | Scripted | PPO |
|--------|------|--------|----------|-----|
| Mean EE error reduction | 0 | ~0 | >0 | TBD |
| Success rate | 0% | ~0% | TBD | TBD |
| Mean return | 0 | negative | TBD | TBD |
| Displacement ratio | N/A | random | calibrated | learned |
| Clipping fraction | 0 | high | low | TBD |
| Oscillation index | physics | high | low | TBD |

## Current State

- Zero baseline: 0.40m EE drift in 0.83s (pure gravity)
- Horizon-IDFD: 92.6% sign accuracy, 0.706 displacement ratio (with fd_gain=3.0)
- Random: to measure
- Scripted: to implement (fixed delta_x_des_H per cylinder toward target)

## Next

Run PPO training, then compare against baselines.
