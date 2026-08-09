# PPO2 ROM-B held-out robustness audit

- Checkpoint: `/home/rrlab/rrlab_mulag/logs/rsl_rl/ppo2_rom_b_robust/2026-07-24_12-48-14/model_53_ppo2_rom_b_final.pt`
- SHA-256: `6731bac46e083d7ac99074e3a7b34752b64117a5003755e3d8296747353a69e9`
- Held-out Monte Carlo result: **PASS**
- Held-out failures: `[]`
- Correlated h seed groups per condition: `5`

## Held-out Monte Carlo worst cases

| mean error | p95 error | p95 acceleration | head alignment | clearance | truck drift | termination |
|---:|---:|---:|---:|---:|---:|---:|
| 0.03747 m | 0.07188 m | 38.91 m/s^2 | 0.98011 | 0.17361 m | 0.001000 m | 0.03906 |

## Correlated h sweep

The h sweep is evaluation-only. Its factors are applied inside the hydraulic ROM before pressure integration. Each result is the conservative worst case across all repeated seed groups and trajectory shapes.

| h | e_comp | e_flow | seal | stiffness | flow | leakage | mean error | p95 error | acc95 | term | gates |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.4 | -0.12500 | 0.00000 | 1.00000 | 0.8825 | 1.0000 | 1.9531 | 0.03600 | 0.06628 | 37.71 | 0.03906 | FAIL |
| 0.5 | -0.10417 | 0.00000 | 0.83333 | 0.9011 | 1.0000 | 1.7643 | 0.03757 | 0.07199 | 38.21 | 0.04688 | PASS |
| 0.6 | -0.08333 | 0.00000 | 0.66667 | 0.9200 | 1.0000 | 1.5880 | 0.03696 | 0.06937 | 39.24 | 0.05469 | FAIL |
| 0.7 | -0.06250 | 0.00000 | 0.50000 | 0.9394 | 1.0000 | 1.4238 | 0.03811 | 0.07110 | 39.62 | 0.04688 | PASS |
| 0.8 | -0.04167 | 0.00000 | 0.33333 | 0.9592 | 1.0000 | 1.2714 | 0.03724 | 0.07038 | 41.09 | 0.07031 | FAIL |
| 0.9 | -0.02083 | 0.00000 | 0.16667 | 0.9794 | 1.0000 | 1.1303 | 0.03780 | 0.07061 | 38.79 | 0.04688 | PASS |
| 1.0 | -0.00000 | 0.00000 | 0.00000 | 1.0000 | 1.0000 | 1.0000 | 0.03696 | 0.06950 | 38.23 | 0.07812 | FAIL |
| 1.1 | 0.01000 | 0.00000 | 0.00000 | 1.0101 | 1.0000 | 1.0000 | 0.03636 | 0.06784 | 40.07 | 0.08594 | FAIL |
| 1.2 | 0.02000 | 0.00000 | 0.00000 | 1.0202 | 1.0000 | 1.0000 | 0.03656 | 0.06838 | 43.00 | 0.07031 | FAIL |

## Artifacts

- `rom_b_held_out_report.json`: complete contract and rollout summaries
- `rom_b_held_out_rows.csv`: one row per rollout condition
- `rom_b_final_robustness.png`: all gates, per-shape curves, repeated-seed envelopes, and Monte Carlo terminations
