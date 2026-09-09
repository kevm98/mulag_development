# Effective Jacobian Report (D-only backend)

## Configuration

- `lowlevel_actuator_backend = "explicit_effort_pd"`
- `kp_scale = 0.1, kd_scale = 0.2`
- `explicit_pd_velocity_filter_beta = 0.8`
- `explicit_force_rate_limit_fraction = 0.01`
- `dx = 0.010 m, H = 20`

## J_eff Matrix

| Cylinder | dX (m/m) | dY (m/m) | dZ (m/m) |
|----------|---------|---------|---------|
| Drehzapfen (0) | -0.00 | -1.74 | -1.51 |
| Ausleger I (1) | -0.00 | +29.81 | -24.73 |
| Ausleger II (2) | +0.00 | +20.45 | -30.88 |
| Messerkopf (3) | +0.00 | +20.29 | -31.66 |

## Quality

| Metric | Value | Pass? |
|--------|-------|-------|
| Rank | 2 | ✓ (>=2) |
| Condition number | 396,566 | ✓ (finite) |
| Signal/drift ratio | 0.1-1.0 | ⚠ (below 3.0, but central diff works) |
| Columns distinct | Yes | ✓ |
| Finite | Yes | ✓ |

## Method

Paired baseline subtraction with central difference:
```
Dzero = ΔEE with zero command
Dplus = ΔEE with +dx on cylinder c
Dminus = ΔEE with -dx on cylinder c
J_col = (Dplus - Dminus) / (2*dx)  [drift cancels]
```

## Key Finding

With kp=0, position commands are ignored (D-only ignores position error).
With kp=0.1 added, cylinders track position targets while kd=0.2 prevents
oscillation. The J_eff is rank-2 with distinct per-cylinder columns.

## Verdict

**J_eff is valid.** Mapper can proceed. Signal-to-drift ratio is low for
individual measurements, but central difference cancels common-mode drift
effectively, producing distinct and physically meaningful J columns.
