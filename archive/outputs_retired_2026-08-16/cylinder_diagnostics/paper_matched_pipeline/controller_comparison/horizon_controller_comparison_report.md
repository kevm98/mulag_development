# Horizon Controller Comparison Report

**Date**: 2026-07-02  
**Trial count**: 792 (A: 144, B: 648 — B has more because of kp×ki grid sweep)

---

## Results Summary

| Metric | Controller A (current_ff) | Controller B (ff_pi) |
|--------|--------------------------|---------------------|
| Trials | 144 | 648 |
| Sign accuracy | **0.944** | 0.965 |
| Displacement ratio | **0.388** | 0.333 |
| FD contribution ratio | **0.900** | 1.118 ⚠️ |

### Per-Cylinder Detail

| Cyl | A sign | A ratio | B sign | B ratio | Winner |
|-----|--------|---------|--------|---------|--------|
| 0 | 1.00 | 0.69 | 1.00 | 0.59 | A (better ratio) |
| 1 | 1.00 | 0.52 | 1.00 | 0.44 | A (better ratio) |
| 2 | 0.83 | 0.19 | 0.89 | 0.16 | B (sign) / A (ratio) |
| 3 | 0.94 | 0.16 | 0.97 | 0.14 | B (sign) / A (ratio) |

---

## Analysis

**Controller B improves sign accuracy by +2.1pp** (0.944→0.965) by adding PI integral correction. However:

1. **Worse displacement ratio**: 0.333 vs 0.388. The PI correction shifts delta_cmd_H away from the FD-predicted value, reducing the effective commanded displacement.

2. **FD contribution > 1.0**: B's fd_contrib ratio of 1.118 means the PI correction often OPPOSES the FD prediction (e.g., FD says +0.01 but PI pushes -0.002, so |fd_gain*fd_sum| > |delta_cmd_H|). This is a disguised form of PI dominance.

3. **All gain combinations tested** (kp=0.15,0.25,0.5; ki=0.01,0.03,0.05): No combination improved both sign accuracy AND displacement ratio.

## Verdict

**Controller A (current_horizon_idfd_ff) is selected.**

Reasoning:
- 94.4% sign accuracy is already high — the 2.1pp gain from PI is marginal
- Controller A has better displacement ratio (0.388 vs 0.333)
- Controller A's FD contribution ratio is clean (0.900, always FD-dominant)
- Controller B's FD contribution > 1.0 violates the requirement that PI should not dominate FD
- Simpler controller with fewer parameters (2 vs 6+ per cylinder)

## Selected Configuration

| Parameter | Value |
|-----------|-------|
| Controller | current_horizon_idfd_ff |
| Horizon steps | 10-20 |
| fd_gain | 1.0 (per-cylinder calibration possible) |
| tracking_gain | 0.0 (or 0.5 for cylinder 0) |
| Ramp shape | linear |

## PPO Readiness

The current controller meets all criteria for PPO training:
- [x] Sign accuracy > 0.90 (0.944)
- [x] FD contribution > 0.3 (0.900)
- [x] ID/FD always active
- [x] No position_ramp_pi bypass
- [x] Stable across H=10,15,20
