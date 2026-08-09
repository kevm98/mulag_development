# End-effector tracking experiments

This folder contains comparison utilities for PPO1-A and PPO2-A tracking results.

Run the default comparison:

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python experiments/plot_end_effector_tracking_errors.py
```

The script reads the existing JSON reports in `outputs/`. By default it compares
PPO1-A `outputs/id_fd_policy_v3/model_575_evaluation.json` against PPO2-A
`outputs/ppo2_rom_residual/plant_reachable_v1_blend25_five_shape_strict.json`.

It saves:

- `experiments/end_effector_tracking_summary.csv`
- `experiments/end_effector_tracking_comparison.png`

It also opens the plot window by default. Use `--no-show` for headless runs.
