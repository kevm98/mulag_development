# Stage 8 — Master raw dataset

Status: **complete**.

Each deployment run is stored as:

```text
datasets/master_v3_stage22_wide/
  manifest.csv
  runs/<run_id>/metadata.json
  runs/<run_id>/trajectory.npz
```

The schema in `aam_data/master_schema.py` records state, full command chain,
quality flags, timing and the applied factor vector. `validate_run` checks every
run before it can be resumed, frozen or trained on.
