# Stage 27 — Validate the complete adaptation chain

Status: **PASS after causal temporal-consistency retraining** on 2026-08-09.

The corrected chain was evaluated on the held-out runs (36: 20 validation +
16 reserved test). The selected temporal-trained L5 model is smoothed with a
strictly past-only rolling-20 mean. For each rollout start, `z_hat` is the AAM prediction from
the causal window ending at that start tick and is held fixed for that
20-tick SD rollout. The three members compared at H20 with hold-start q:

```text
SD_0(s, u)               no-latent control model
SD_z(s, u, z_true)       frozen privileged encoder latent
SD_z(s, u, z_hat)        frozen SD_z with the trained AAM latent (L5, 0.25 s)
```

## Results (paired bootstrap 95% CIs, per run)

Pooled (n=36):

| comparison | extension | velocity |
|---|---|---|
| `z_true` vs none | -0.000175 [-0.000236, -0.000120] | -0.000449 [-0.000711, -0.000188] |
| `z_hat` vs none | -0.000144 [-0.000193, -0.000098] | -0.000495 [-0.000717, -0.000281] |
| `z_hat` vs `z_true` | +0.000031 [-0.000002, +0.000071] | -0.000046 [-0.000234, +0.000139] |

Retention of the true-latent advantage recovered by `z_hat`: extension 0.825,
velocity 1.101.

## Gate

Required checks:

- `z_true` beats none pooled (ext + vel);
- `z_hat` beats none pooled (ext + vel);
- `z_hat` is not significantly worse than `z_true` (paired CI contains 0) — **PASS**;
- retention is positive on both channels.

The pre-retraining causal verdict failed the extension equivalence check. The
temporal-trained L5 model closes it while preserving the no-latent advantage.
Latest verdict:
`AAM_module/results/stage27_chain_verify_temporal/verdict.json`.
The prior `stage27_chain_verify/verdict.json` used noncausal full-run z_hat
means and is historical only.

Historical verdict: `AAM_module/results/stage27_chain_verify/verdict.json`.
Tool: `AAM_module/tools/validate_stage27_chain.py`.
