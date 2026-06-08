# ROM Rollout Evaluation

- Split: `val`
- Rollout mode: `teacher_pressure`
- State limits: `training_range`
- Runs: `3`

## Horizons

| Horizon | Runs | Extension RMSE | dP RMSE | Max Ext Error | Unstable Fraction | NaN/Inf | Limit Violations |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.1s | 3 | 0.02055626871973442 | 0.0 | 0.05757361650466919 | 0.0 | 0 | 0 |
| 0.5s | 3 | 0.08109943173345746 | 0.0 | 0.25747624039649963 | 0.0 | 0 | 1 |
| 1.0s | 3 | 0.10461300567784745 | 0.0 | 0.2871701419353485 | 0.0 | 0 | 11 |
| 2.0s | 3 | 0.16787001704767163 | 0.0 | 0.5753825902938843 | 0.0 | 0 | 43 |
| full_run | 3 | 0.7908743611811131 | 0.0 | 2.301457405090332 | 0.0 | 0 | 1196 |