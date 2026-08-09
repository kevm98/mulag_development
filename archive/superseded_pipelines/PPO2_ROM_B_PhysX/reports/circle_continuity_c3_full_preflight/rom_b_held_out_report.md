# PPO2 ROM-B held-out robustness audit

- Checkpoint: `/home/rrlab/rrlab_mulag/logs/rsl_rl/ppo2_rom_b_robust/2026-07-24_12-48-14/model_53_ppo2_rom_b_final.pt`
- SHA-256: `6731bac46e083d7ac99074e3a7b34752b64117a5003755e3d8296747353a69e9`
- Held-out Monte Carlo result: **PASS**
- Held-out failures: `[]`
- Correlated h seed groups per condition: `1`

## Held-out Monte Carlo worst cases

| mean error | p95 error | p95 acceleration | head alignment | clearance | truck drift | termination |
|---:|---:|---:|---:|---:|---:|---:|
| 0.02676 m | 0.05076 m | 38.21 m/s^2 | 0.98281 | 0.10113 m | 0.001000 m | 0.00000 |

## Correlated h sweep

The h sweep is evaluation-only. Its factors are applied inside the hydraulic ROM before pressure integration. Each result is the conservative worst case across all repeated seed groups and trajectory shapes.

| h | e_comp | seal | stiffness | leakage | mean error | p95 error | acc95 | term | gates |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.4 | -0.12500 | 1.00000 | 0.8825 | 1.9531 | 0.02509 | 0.05106 | 38.76 | 0.00000 | PASS |
| 0.5 | -0.10417 | 0.83333 | 0.9011 | 1.7643 | 0.02443 | 0.04459 | 36.20 | 0.00000 | PASS |
| 0.6 | -0.08333 | 0.66667 | 0.9200 | 1.5880 | 0.02479 | 0.04743 | 38.80 | 0.01562 | PASS |
| 0.7 | -0.06250 | 0.50000 | 0.9394 | 1.4238 | 0.02456 | 0.04639 | 37.57 | 0.00000 | PASS |
| 0.8 | -0.04167 | 0.33333 | 0.9592 | 1.2714 | 0.02457 | 0.04602 | 39.65 | 0.00000 | PASS |
| 0.9 | -0.02083 | 0.16667 | 0.9794 | 1.1303 | 0.02412 | 0.04514 | 36.12 | 0.01562 | PASS |
| 1.0 | -0.00000 | 0.00000 | 1.0000 | 1.0000 | 0.02385 | 0.04373 | 35.34 | 0.00000 | PASS |
| 1.1 | 0.01000 | 0.00000 | 1.0101 | 1.0000 | 0.02407 | 0.04653 | 34.62 | 0.00000 | PASS |
| 1.2 | 0.02000 | 0.00000 | 1.0202 | 1.0000 | 0.02378 | 0.04266 | 33.71 | 0.00000 | PASS |

## Artifacts

- `rom_b_held_out_report.json`: complete contract and rollout summaries
- `rom_b_held_out_rows.csv`: one row per rollout condition
- `rom_b_final_robustness.png`: all gates, per-shape curves, repeated-seed envelopes, and Monte Carlo terminations
