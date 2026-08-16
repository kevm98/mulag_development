# Pipeline ranking and promotion targets

## Current ranking

This ranking uses the six-mesh end-to-end gate reports for ROM and ID–FD, and
the Stage-30 held-out causal evaluator for AAM. AAM does not yet have a trained
68-observation PPO policy, so it cannot honestly receive an end-to-end PhysX
rank.

| rank | pipeline | reason |
|---:|---|---|
| 1 | PPO_ROM | only pipeline that passes all eight gates on six meshes with useful margins |
| 2 | PPO_ID_FD | best lateral tracking and fewer hard impacts, but fails ground contact and safety margins are worse |
| 3 | PPO_ID_AAM_FD, current | safe shadow mode equals frozen ID; nonzero causal adapter currently regresses closed-loop tracking |

## End-to-end gate comparison

Lower is better except contact and alignment. These are the measured worst-mesh
values from the six-mesh reports, not predictions.

| metric | PPO_ROM | PPO_ID_FD | per-metric leader | AAM promotion target |
|---|---:|---:|---|---:|
| worst-family lateral error (m) | 0.0817 | **0.0628** | ID–FD | `< 0.0628` |
| mean cut-height error (m) | **0.0077** | 0.0168 | ROM | `< 0.0077` |
| worst-family ground contact | **0.7854** | 0.6882 FAIL | ROM | `> 0.7854` |
| hard-impact fraction | 0.0609 | **0.0515** | ID–FD | `< 0.0515` |
| p01 head alignment | 0.9860 | **0.9879** | ID–FD | `> 0.9879` |
| safety termination rate | **0.0043** | 0.0182 | ROM | `< 0.0043` |
| p95 tool acceleration (m/s²) | **65.24** | 65.69 | ROM | `< 65.24` |

For AAM to be genuinely better overall, it should beat all seven targets on
the same six meshes and pass all eight gates. Beating only lateral error is not
enough; ID–FD already does that while failing contact.

## Stage-30 causal adapter evidence

These values are directly comparable because they come from the same 36 held-out
validation/test runs.

| configuration | closed-loop cylinder RMSE | saturation fraction | first divergence |
|---|---:|---:|---:|
| frozen ID / safe AAM scale 0.0 | **19.828** | 89.17% | **30.49 s** |
| online AAM adapter scale 0.25 | 20.438 | **59.12%** | 30.17 s |
| online AAM adapter scale 0.10 | 21.414 | 57.85% | 30.23 s |

The nonzero adapter makes the valve command look less saturated, but that did
not produce smoother or more accurate closed-loop motion. Saturation alone is
therefore not a smoothness metric. The adapter remains disabled by default.

The Stage-29 privileged true-latent adapter reached 19.322 m on the same
closed-loop cylinder metric, but the deployment-available AAM-latent adapter
reached 21.045 m. The privileged result cannot be used as the deployment rank.

## Metrics that must be used for “smooth tracking”

Rank the next trained AAM policy in this order:

1. Full closed-loop tool-path RMSE and worst-mesh/worst-family lateral error.
2. Cut-height RMSE and ground-contact fraction.
3. Overshoot and settling time on the fixed 1 s, 5 s, and 20 s horizons.
4. Valve total variation, action-rate RMSE, and action jerk RMS.
5. Hard impacts, p95 tool acceleration, and safety terminations.
6. Saturation fraction and divergence time.

The policy is promotable only if it is non-regressive against both ROM and
ID–FD on the safety metrics and beats the best baseline on the tracking
metrics. Teacher-forced AAM prediction error is diagnostic only; it must not be
used to promote an adapter that loses in closed loop.

Source reports:

- `PPO_ROM/reports/frontier/rom_contact6_gates.json`
- `PPO_ROM/reports/frontier/idfd_contact6_gates.json`
- `outputs/stage30_online_aam/evaluation_safe/summary.json`
- `outputs/stage30_online_aam/evaluation_rolling20_scale025_fixed/summary.json`
