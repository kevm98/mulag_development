# PPO2 ROM-B held-out robustness audit

- Checkpoint: `/home/rrlab/rrlab_mulag/logs/rsl_rl/ppo2_rom_b_robust/2026-07-24_12-48-14/model_53_ppo2_rom_b_final.pt`
- SHA-256: `6731bac46e083d7ac99074e3a7b34752b64117a5003755e3d8296747353a69e9`
- Held-out Monte Carlo result: **FAIL**
- Held-out failures: `['termination_rate']`
- Correlated h seed groups per condition: `3`

## Held-out Monte Carlo worst cases

| mean error | p95 error | p95 acceleration | head alignment | clearance | truck drift | termination |
|---:|---:|---:|---:|---:|---:|---:|
| 0.03676 m | 0.06883 m | 39.38 m/s^2 | 0.98002 | 0.19999 m | 0.001000 m | 0.05469 |

## Correlated h sweep

The h sweep is evaluation-only. Its factors are applied inside the hydraulic ROM before pressure integration. Each result is the conservative worst case across all repeated seed groups and trajectory shapes.

| h | e_comp | e_flow | seal | stiffness | flow | leakage | mean error | p95 error | acc95 | term | gates |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.4 | -0.12500 | -0.20000 | 1.00000 | 0.8825 | 0.8000 | 1.9531 | 0.03706 | 0.06936 | 37.47 | 0.07812 | FAIL |
| 0.5 | -0.10417 | -0.16667 | 0.83333 | 0.9011 | 0.8333 | 1.7643 | 0.03608 | 0.06816 | 38.07 | 0.03125 | PASS |
| 0.6 | -0.08333 | -0.13333 | 0.66667 | 0.9200 | 0.8667 | 1.5880 | 0.03738 | 0.07251 | 40.05 | 0.07031 | FAIL |
| 0.7 | -0.06250 | -0.10000 | 0.50000 | 0.9394 | 0.9000 | 1.4238 | 0.03727 | 0.06858 | 34.89 | 0.04688 | PASS |
| 0.8 | -0.04167 | -0.06667 | 0.33333 | 0.9592 | 0.9333 | 1.2714 | 0.03605 | 0.06646 | 38.03 | 0.04688 | PASS |
| 0.9 | -0.02083 | -0.03333 | 0.16667 | 0.9794 | 0.9667 | 1.1303 | 0.03752 | 0.07143 | 42.21 | 0.07031 | FAIL |
| 1.0 | -0.00000 | -0.00000 | 0.00000 | 1.0000 | 1.0000 | 1.0000 | 0.03730 | 0.06872 | 35.50 | 0.03906 | PASS |
| 1.1 | 0.01000 | 0.10000 | 0.00000 | 1.0101 | 1.1000 | 1.0000 | 0.03686 | 0.06791 | 37.93 | 0.05469 | FAIL |
| 1.2 | 0.02000 | 0.20000 | 0.00000 | 1.0202 | 1.2000 | 1.0000 | 0.03814 | 0.07240 | 36.18 | 0.06250 | FAIL |

## Artifacts

- `rom_b_held_out_report.json`: complete contract and rollout summaries
- `rom_b_held_out_rows.csv`: one row per rollout condition
- `rom_b_final_robustness.png`: all gates, per-shape curves, repeated-seed envelopes, and Monte Carlo terminations
