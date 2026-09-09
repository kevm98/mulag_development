# FD Gain Calibration Report: current_horizon_idfd_ff

**Date**: 2026-07-02  
**Trials**: 360 (5 gains × 3 horizons × 4 cylinders × 6 deltas)

---

## Global Results

| fd_gain | Sign Acc | Disp Ratio | Grav Fails |
|---------|----------|------------|------------|
| 1.0 | 0.917 | 0.354 | 0 |
| 1.5 | 0.958 | 0.479 | 0 |
| 2.0 | 0.958 | 0.579 | 0 |
| 2.5 | **1.000** | 0.643 | 0 |
| 3.0 | **1.000** | **0.706** | 0 |

fd_gain=3.0 doubles displacement ratio (0.354 → 0.706) while achieving perfect sign accuracy.

## Per-Cylinder Detail

| Cyl | fg=1.0 | fg=1.5 | fg=2.0 | fg=2.5 | fg=3.0 | Best fg |
|-----|--------|--------|--------|--------|--------|---------|
| 0 | 0.646 | 0.885 | 1.077 | 1.195 | 1.308 | **2.0** (ratio 1.08) |
| 1 | 0.477 | 0.650 | 0.786 | 0.867 | 0.945 | **3.0** (ratio 0.95) |
| 2 | 0.174 | 0.210 | 0.245 | 0.270 | 0.297 | **3.0** |
| 3 | 0.121 | 0.170 | 0.209 | 0.241 | 0.274 | **3.0** |

- Cylinder 0 overshoots slightly at fg=3.0 (1.308). fg=2.0 gives 1.077 — closer to 1.0.
- Cylinder 1 reaches near-perfect tracking at fg=3.0 (0.945).
- Cylinders 2,3 remain low (0.27-0.30) — gravity-dominated, no fd_gain fully compensates.

## Recommended PPO Configuration

### Global (simple)

| Parameter | Value |
|-----------|-------|
| Controller | current_horizon_idfd_ff |
| fd_gain | **3.0** |
| tracking_gain | 0.0 |
| H | 10-20 |
| Sign accuracy | 1.000 |
| Displacement ratio | 0.706 |

### Per-cylinder (if supported)

| Cyl | fd_gain | Expected Ratio |
|-----|---------|---------------|
| 0 | 2.0 | 1.08 |
| 1 | 3.0 | 0.95 |
| 2 | 3.0 | 0.30 |
| 3 | 3.0 | 0.27 |

## PPO Action Scale

With fd_gain=3.0, displacement ratio improves to 0.706. Use:
- **horizon_action_scale_m = 0.01** (conservative, matches calibrated gain improvements)

If action_scale=0.01 gives too little motion, increase to 0.02.
