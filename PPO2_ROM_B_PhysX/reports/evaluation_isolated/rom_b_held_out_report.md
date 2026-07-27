# PPO2 ROM-B held-out robustness audit

- Checkpoint: `/home/rrlab/rrlab_mulag/logs/rsl_rl/ppo2_rom_b_robust/2026-07-24_12-48-14/model_53_ppo2_rom_b_final.pt`
- SHA-256: `6731bac46e083d7ac99074e3a7b34752b64117a5003755e3d8296747353a69e9`
- Held-out Monte Carlo result: **FAIL**
- Held-out failures: `['p95_acceleration']`
- Correlated h seed groups per condition: `3`

## Held-out Monte Carlo worst cases

| mean error | p95 error | p95 acceleration | head alignment | clearance | truck drift | termination |
|---:|---:|---:|---:|---:|---:|---:|
| 0.03665 m | 0.06719 m | 41.02 m/s^2 | 0.98000 | 0.15336 m | 0.001000 m | 0.02344 |

## Correlated h sweep

The h sweep is evaluation-only. Its factors are applied inside the hydraulic ROM before pressure integration. Each result is the conservative worst case across all repeated seed groups and trajectory shapes.

| h | e_comp | seal | stiffness | leakage | mean error | p95 error | acc95 | term | gates |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.4 | -0.12500 | 1.00000 | 0.8825 | 1.9531 | 0.03497 | 0.06453 | 42.99 | 0.03125 | FAIL |
| 0.5 | -0.10417 | 0.83333 | 0.9011 | 1.7643 | 0.03607 | 0.06630 | 39.96 | 0.01562 | PASS |
| 0.6 | -0.08333 | 0.66667 | 0.9200 | 1.5880 | 0.03489 | 0.06455 | 41.03 | 0.04688 | FAIL |
| 0.7 | -0.06250 | 0.50000 | 0.9394 | 1.4238 | 0.03496 | 0.06200 | 40.43 | 0.01562 | FAIL |
| 0.8 | -0.04167 | 0.33333 | 0.9592 | 1.2714 | 0.03595 | 0.06737 | 40.43 | 0.03906 | FAIL |
| 0.9 | -0.02083 | 0.16667 | 0.9794 | 1.1303 | 0.03562 | 0.06312 | 40.48 | 0.02344 | FAIL |
| 1.0 | -0.00000 | 0.00000 | 1.0000 | 1.0000 | 0.03433 | 0.06227 | 44.08 | 0.02344 | FAIL |
| 1.1 | 0.01000 | 0.00000 | 1.0101 | 1.0000 | 0.03599 | 0.06411 | 42.21 | 0.02344 | FAIL |
| 1.2 | 0.02000 | 0.00000 | 1.0202 | 1.0000 | 0.03540 | 0.06583 | 41.46 | 0.03125 | FAIL |

## Artifacts

- `rom_b_held_out_report.json`: complete contract and rollout summaries
- `rom_b_held_out_rows.csv`: one row per rollout condition
- `rom_b_final_robustness.png`: all gates, per-shape curves, repeated-seed envelopes, and Monte Carlo terminations
