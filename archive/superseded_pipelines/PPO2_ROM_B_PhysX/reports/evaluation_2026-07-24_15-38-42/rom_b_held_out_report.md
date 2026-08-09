# PPO2 ROM-B held-out robustness audit

- Checkpoint: `/home/rrlab/rrlab_mulag/logs/rsl_rl/ppo2_rom_b_robust/2026-07-24_12-48-14/model_53_ppo2_rom_b_final.pt`
- SHA-256: `6731bac46e083d7ac99074e3a7b34752b64117a5003755e3d8296747353a69e9`
- Held-out Monte Carlo result: **FAIL**
- Held-out failures: `['termination_rate']`

## Held-out Monte Carlo worst cases

| mean error | p95 error | p95 acceleration | head alignment | clearance | truck drift | termination |
|---:|---:|---:|---:|---:|---:|---:|
| 0.03676 m | 0.06883 m | 39.38 m/s^2 | 0.98002 | 0.19999 m | 0.001000 m | 0.05469 |

## Correlated h sweep

The h sweep is evaluation-only. Its factors are applied inside the hydraulic ROM before pressure integration.

| h | e_comp | e_flow | seal | stiffness | flow | leakage | mean error | p95 error | acc95 | term | gates |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.4 | -0.12500 | -0.20000 | 1.00000 | 0.8825 | 0.8000 | 1.9531 | 0.03557 | 0.06691 | 37.47 | 0.02344 | PASS |
| 0.5 | -0.10417 | -0.16667 | 0.83333 | 0.9011 | 0.8333 | 1.7643 | 0.03492 | 0.06649 | 35.56 | 0.04688 | PASS |
| 0.6 | -0.08333 | -0.13333 | 0.66667 | 0.9200 | 0.8667 | 1.5880 | 0.03545 | 0.06550 | 37.97 | 0.05469 | FAIL |
| 0.7 | -0.06250 | -0.10000 | 0.50000 | 0.9394 | 0.9000 | 1.4238 | 0.03594 | 0.06723 | 35.45 | 0.03906 | PASS |
| 0.8 | -0.04167 | -0.06667 | 0.33333 | 0.9592 | 0.9333 | 1.2714 | 0.03506 | 0.06617 | 35.51 | 0.05469 | FAIL |
| 0.9 | -0.02083 | -0.03333 | 0.16667 | 0.9794 | 0.9667 | 1.1303 | 0.03592 | 0.06726 | 35.29 | 0.04688 | PASS |
| 1.0 | -0.00000 | -0.00000 | 0.00000 | 1.0000 | 1.0000 | 1.0000 | 0.03510 | 0.06601 | 32.94 | 0.03125 | PASS |
| 1.1 | 0.01000 | 0.10000 | 0.00000 | 1.0101 | 1.1000 | 1.0000 | 0.03530 | 0.06703 | 35.76 | 0.04688 | PASS |
| 1.2 | 0.02000 | 0.20000 | 0.00000 | 1.0202 | 1.2000 | 1.0000 | 0.03510 | 0.06616 | 32.31 | 0.05469 | FAIL |

## Artifacts

- `rom_b_held_out_report.json`: complete contract and rollout summaries
- `rom_b_held_out_rows.csv`: one row per rollout condition
- `rom_b_h_sweep.png`: correlated robustness curves
