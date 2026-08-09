# Stage 6 — Deployment domains

Status: **complete**.

The experiment uses five groups: nominal, hydraulic-only, mechanical-only,
combined-train and unseen combined-test. The factor vector is constant within
each run. Domain labels and factors are privileged metadata and are never AAM
inputs at deployment.

Evaluation is by complete run and unseen factor configuration, not randomly
split windows.
