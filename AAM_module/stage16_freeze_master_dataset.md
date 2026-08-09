# Stage 16 — Freeze and split the dataset

Status: **complete**.

The immutable dataset is `datasets/master_v3_stage22_wide_frozen/`. Raw runs
and the manifest are protected by SHA-256 checksums; derived SD/AAM data is
written outside `raw/`.

Run-level split:

- train: 82 runs;
- validation: 20 runs;
- reserved test: 16 combined-domain runs.

There is no train/validation/test factor-key leakage, and no learning window
crosses a run boundary.
