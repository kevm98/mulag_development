# Stage 7 — Parameter draws

Status: **complete**.

`tools/domain_sampler.py` uses deterministic scrambled Sobol draws with fixed
per-domain seeds. Test factor keys are rejected if they overlap train/validation
keys. Boundary anchors are inserted explicitly.

The final 118-run design has 116 unique keys (98.3%), full active-factor rank,
all quartiles populated and no unexpected duplicates. The three nominal-key
runs are intentional baseline/repeatability measurements.
