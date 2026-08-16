# Stage 28 — Frozen historical ID transfer

Status: **complete; meaningful transfer gap confirmed.** The historical
Simscape-trained ID and strict FD were evaluated without adaptation on all 118
frozen Stage-22 runs: 2 nominal, 40 hydraulic, 12 mechanical, 64 combined
(48 combined-train plus 16 held-out combined-test).

## Protocol

The exact frozen artifacts were used:

```text
id_fd_model/best_id_cylinder_control.pt
id_fd_model/best_fd_cylinder_strict.pt
```

The ID receives measured or predicted cylinder extension/velocity and the
recorded desired cylinder position/velocity. dP and force use the training
means required by the current legacy `training_mean` feature contract; force
is not present in the Stage-22 logs. The strict FD recursively advances the
predicted state at the native 50 ms timebase.

Two measurements are reported:

1. **Teacher-forced one-step transfer:** the current measured cylinder state is
   supplied at every tick and the ID→FD next-state prediction is compared with
   the measured next state.
2. **Closed-loop frozen transfer:** the predicted state is fed back into the
   frozen ID→FD pair while the recorded desired cylinder reference is supplied.

Stage-22 contains continuous DLS/PRBS trajectories rather than isolated step
labels. Therefore overshoot and settling are bounded proxies: terminal-target
signed excess on moving segments, absolute hold error on near-stationary
segments, and the first 0.5 s interval remaining within 5 mm inside each
declared segment.

There is no desired end-effector trajectory in the frozen collection. The
reported tool-path value is consequently a transfer-error proxy,
`J_tool (x_pred - x_measured)`, using the validated 3×4 cylinder-to-tool local
Jacobian; it is not a direct task tracking error.

## Results

Values below are means over complete runs in each group. `H20`, `H100` and
`H400` are 1 s, 5 s and 20 s closed-loop horizons.

| domain | runs | one-step Δx RMSE | H20 track RMSE | H100 track RMSE | H400 track RMSE | full track RMSE | first |x|>1 m | ID preclip saturation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nominal | 2 | 3.36 mm | 10.26 mm | 10.29 mm | 0.756 m | 14.49 m | 20.35 s | 92.9% |
| hydraulic | 40 | 2.43 mm | 10.03 mm | 10.43 mm | 0.262 m | 19.90 m | 32.22 s | 88.5% |
| mechanical | 12 | 2.81 mm | 9.75 mm | 9.85 mm | 0.320 m | 20.65 m | 24.07 s | 91.9% |
| combined | 64 | 2.36 mm | 10.01 mm | 10.27 mm | 0.139 m | 20.24 m | 31.87 s | 88.8% |
| held-out combined-test | 16 | 2.34 mm | 10.19 mm | 10.41 mm | 0.081 m | 21.50 m | 31.05 s | 89.4% |

Overall one-step velocity RMSE is 0.090 m/s; held-out combined-test is
0.081 m/s. The frozen model's full closed-loop valve magnitude on held-out
data is 0.821 mean absolute command versus 0.257 for the recorded ROM valve,
and 89.4% of predicted valve elements hit the historical ID preclip limits.
The recorded Stage-22 ROM valve saturation at the plant ±1.0 bound is 0% on
the held-out group.

For context, the measured Stage-22 plant itself tracks the recorded cylinder
reference at approximately 0.807 mm RMSE on the held-out group. The large
closed-loop values are therefore a model-transfer failure, not a failure of
the Stage-22 collection.

## Interpretation and downstream implication

The legacy ID→FD pair is locally plausible for roughly the first 1–5 seconds
and has millimetre-scale teacher-forced extension error, but its autoregressive
state distribution leaves the Simscape training support. Valve saturation and
state runaway then appear across every domain, including held-out combined
runs. The historical ID cannot be promoted as the unadapted deployment path.

Stage 29 is therefore required: keep the ID and FD checkpoints frozen and add
only a bounded residual adapter. The adapter must be evaluated against this
plain frozen-ID baseline, first with `z_true` and then with the causal AAM
`z_hat`; no Stage-28 result authorizes retraining or replacing the historical
ID/FD pair.

Artifacts:

- [summary.json](../outputs/stage28_id_transfer/summary.json)
- [per_run.json](../outputs/stage28_id_transfer/per_run.json)
- [rollouts.npz](../outputs/stage28_id_transfer/rollouts.npz)
- [Stage 28 evaluator](tools/evaluate_stage28_id_transfer.py)
