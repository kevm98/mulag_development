# Three-pipeline tracking comparison

This comparison uses the same difficult `double_s_curve`, held-out mesh `909`,
stage 6, seed `17`, 2.2x path amplitude, and 2808 control steps for:

- PPO-ROM
- PPO-ID-FD
- PPO-ID(AAM)-FD, with the online AAM residual scale fixed at `0.25`

Run the complete evaluation and plotting workflow:

```bash
bash experiments/run_three_pipeline_comparison.sh
```

## Ordered truck-speed robustness campaign

`run_ordered_training_forever.sh` trains ROM, then ID-FD, then ID(AAM)-FD with
one constant truck speed sampled per episode from 2--7 km/h. After all three
policies pass their held-out gates, it evaluates the promoted checkpoint at
`unimog_speed` 2, 3, 4, 5, 6, and 7 km/h. The six-mesh reports and checkpoint identities are summarized
in `experiments/speed_range_2_7_pathfix_v2/post_training_eval/summary.json`.

The run writes per-step traces to
`experiments/three_pipeline_tracking/traces/` and plots to
`experiments/three_pipeline_tracking/plots/`:

- `ee_tracking_errors.png`: EE norm and signed x/y/z errors
- `joint_tracking_errors.png`: signed error for each of the five joints and RMS bars
- `ee_paths.png`: actual versus reference EE paths
- `summary.csv`: MAE, RMS, p95, and maximum errors

To replot existing traces only:

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  experiments/plot_three_pipeline_errors.py --no-show
```
