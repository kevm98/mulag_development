# PPO2 ROM-B held-out robustness audit

- Checkpoint: `/home/rrlab/rrlab_mulag/logs/rsl_rl/ppo2_rom_b_robust/2026-07-24_12-48-14/model_53_ppo2_rom_b_final.pt`
- SHA-256: `6731bac46e083d7ac99074e3a7b34752b64117a5003755e3d8296747353a69e9`
- Held-out Monte Carlo result: **PASS**
- Held-out failures: `[]`
- Correlated h seed groups per condition: `1`

## Held-out Monte Carlo worst cases

| mean error | p95 error | p95 acceleration | head alignment | clearance | truck drift | termination |
|---:|---:|---:|---:|---:|---:|---:|
| 0.02730 m | 0.05187 m | 38.87 m/s^2 | 0.98297 | 0.08289 m | 0.001000 m | 0.00000 |

## Correlated h sweep

The h sweep is evaluation-only. Its factors are applied inside the hydraulic ROM before pressure integration. Each result is the conservative worst case across all repeated seed groups and trajectory shapes.

| h | e_comp | seal | stiffness | leakage | mean error | p95 error | acc95 | term | gates |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.5 | -0.10417 | 0.83333 | 0.9011 | 1.7643 | 0.02562 | 0.04794 | 40.26 | 0.01562 | FAIL |
| 0.6 | -0.08333 | 0.66667 | 0.9200 | 1.5880 | 0.02440 | 0.04406 | 36.13 | 0.00000 | PASS |
| 0.7 | -0.06250 | 0.50000 | 0.9394 | 1.4238 | 0.02489 | 0.04714 | 36.65 | 0.00000 | PASS |
| 1.2 | 0.02000 | 0.00000 | 1.0202 | 1.0000 | 0.02523 | 0.04551 | 35.28 | 0.00000 | PASS |

## Artifacts

- `rom_b_held_out_report.json`: complete contract and rollout summaries
- `rom_b_held_out_rows.csv`: one row per rollout condition
- `rom_b_final_robustness.png`: all gates, per-shape curves, repeated-seed envelopes, and Monte Carlo terminations
