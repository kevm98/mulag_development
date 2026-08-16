# Stage 24 — Build the AAM dataset

Status: **complete**.

> Historical note: the original Stage 24 26-D contract excluded pressure.
> Stage 30 supersedes it for deployment. The current pressure-inclusive
> dataset is `datasets/master_v3_stage22_wide_frozen/aam_pressure/dataset/`
> and uses 30-D rows: external ROM command 8, cylinder extension/velocity 8,
> measured cylinder pressure 4, and joint position/velocity 10. Valve commands
> remain excluded from the AAM input.

At each 50 ms ROM tick, 26-D observable features are recorded per run:
applied action 8, cylinder extension/velocity 8, joint position/velocity 10.
Excluded: physical factors, true `z` (target only), ROM hidden state, future
observations, dP and the desired task reference.

Each causal history window (lags 5/10/20/40 ticks = 0.25/0.5/1.0/2.0 s)
carries the stopped-gradient target `z = mu(e)` from the frozen Stage 23
encoder. Valid rows are keyed by `window_end_tick` (the latest observation in
the causal window, never by a future row). The derived dataset lives under
`datasets/master_v3_stage22_wide_frozen/aam/dataset/` (118 runs, 444,624
ticks). Builder: `tools/build_aam_dataset.py`.
