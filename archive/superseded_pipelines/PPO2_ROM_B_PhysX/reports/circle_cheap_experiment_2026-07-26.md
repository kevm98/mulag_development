# ROM-B cheap circle experiment

Date: 2026-07-26

## Question

Can the current `model_53` circle acceleration failure be removed without PPO
training, while leaving the inner controller and deterministic truck design
unchanged?

This is a mechanism screen, not certification. Every PhysX condition below ran
in a fresh process with 128 environments, 120 scored steps, the corrected
clearance frame, the exact five-factor ROM-B build, and the locked `model_53`.

## Offline reference check

The tool-space circle is exactly closed and its first/last cylinder commands are
equal. However, differentiating the 241-point inverse cylinder command along
arc length shows its largest second-derivative spike at path fraction 0.321
(sample 77, magnitude 9.53). This is in trajectory bin 3, the same bin where
the measured tool acceleration is consistently largest.

This is evidence of a jagged inverse-cylinder reference, not a geometric gap in
the circle itself.

## Experiment 1: lower speed

Condition: `h=0.4`, seed group 3, audit seed 400030.

| base speed setting | effective circle speed | p95 acceleration | mean error | p95 error | clearance max | termination |
|---:|---:|---:|---:|---:|---:|---:|
| 0.18 | 0.126 m/s | 40.72 | 0.02646 m | 0.05029 m | 0.10346 m | 2/128 |
| 0.16 | 0.112 m/s | 40.67 | 0.02578 m | 0.04793 m | 0.11044 m | 1/128 |
| 0.14 | 0.098 m/s | 40.31 | 0.02506 m | 0.04649 m | 0.10450 m | 1/128 |

Result: reducing speed by 22% improved tracking but reduced p95 acceleration by
only 0.41 m/s2. Every setting still failed the 40 m/s2 gate. Speed is not the
main mechanism and should not be used as the fix.

## Experiment 2: PPO residual ablation

| condition | residual | p95 acceleration | bin-3 p95 | mean / p95 error | clearance max | termination |
|---|---|---:|---:|---:|---:|---:|
| h=0.4, round 3 | full | 40.72 | 57.47 | 0.02646 / 0.05029 m | 0.10346 m | 2/128 |
| h=0.4, round 3 | zero | 38.51 | 45.57 | 0.02643 / 0.04835 m | 0.11778 m | 0/128 |
| h=0.9, round 2 | full | 35.66 | 47.46 | 0.02566 / 0.04701 m | 0.14527 m | 2/128 |
| h=0.9, round 2 | zero | 37.46 | 60.85 | 0.02662 / 0.04997 m | 0.09495 m | 0/128 |

Result: zero residual removed truck-reaction terminations in both probes, but its
acceleration effect reversed between conditions. Therefore the circle should
not simply disable PPO. A half-strength residual also failed the first condition
at 40.53 m/s2.

## Experiment 3: filter only inverse-cylinder reference velocity

An EMA with `beta=0.8` was applied only to the cylinder velocity feed-forward
computed from adjacent inverse-reference seeds. Position goals, geometry, PPO,
inner-controller gains, ROM plant, and truck were unchanged.

| condition | beta | p95 acceleration | bin-3 p95 | mean / p95 error | clearance max | alignment min | termination |
|---|---:|---:|---:|---:|---:|---:|---:|
| h=0.4, round 3 | 0.0 | 40.72 | 57.47 | 0.02646 / 0.05029 m | 0.10346 m | 0.98221 | 2/128 |
| h=0.4, round 3 | 0.8 | 39.33 | 53.57 | 0.02613 / 0.04826 m | 0.10129 m | 0.98236 | 1/128 |
| h=0.9, round 2 | 0.0 | 35.66 | 47.46 | 0.02566 / 0.04701 m | 0.14527 m | 0.98280 | 2/128 |
| h=0.9, round 2 | 0.8 | 36.04 | 48.81 | 0.02551 / 0.04709 m | 0.09582 m | 0.98164 | 0/128 |

Result: the filter passed the overall acceleration gate in both probes, improved
mean tracking and clearance in both, and reduced four truck-reaction failures to
one. It improved acceleration strongly in the failing condition and regressed it
slightly in the already-passing condition. Alignment stayed above 0.98.

This is the only candidate from the cheap experiment worth a broader preflight.
It is not yet accepted because two conditions cannot establish a worst-case
claim, and the second condition shows that acceleration is not monotonically
improved.

## Decision

- Do not train PPO yet.
- Do not reduce circle speed.
- Do not disable or globally shrink PPO residuals.
- Do not change the production default yet; `dls_velocity_filter_beta` remains
  0.0 outside explicitly requested diagnostics.
- Next experiment: a process-isolated circle-only preflight of `beta=0.8` over
  fresh Monte Carlo draws and all h values. Compare it against `beta=0.0` using
  identical condition seeds. Only if it passes every arm/safety metric should
  the filter enter the ROM-B evaluation contract.

## Reproducibility change

`scripts/diagnose.py` now exposes two diagnostic-only options:

```text
--force_zero_policy
--dls_velocity_filter_beta BETA
```

Both default to the original production behavior. No controller, policy,
trajectory library, training configuration, or certification threshold was
changed.
