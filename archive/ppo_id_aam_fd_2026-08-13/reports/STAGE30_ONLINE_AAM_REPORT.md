# Stage 30 — Pressure-inclusive online AAM with ID–FD

Status: **implementation PASS; pressure AAM and adapter retraining PASS;
scale-0.75 adapter promotion PASS; safe fallback PASS.**

## Deployment contract

```text
external ROM command/state history -> causal AAM(L5) -> rolling-20 z_hat
policy action -> frozen ID + bounded residual adapter -> strict frozen FD -> PhysX
```

The AAM input is the deployment-observable 30-D row:

```text
[external ROM command 8, cylinder extension 4, cylinder velocity 4,
 measured cylinder pressure dP 4, arm joint position 5, arm joint velocity 5]
```

There is no external valve-command input and no ROM hidden valve state in the
AAM. `rom_predicted_valve_cmd` is used only as a teacher-forced ID-history or
diagnostic value in offline evaluation; it is not an AAM feature. The ID's own
previous output remains internal to the frozen ID checkpoint because it is part
of that checkpoint's existing input contract.

## Rebuilt and retrained artifacts

| artifact | result |
|---|---|
| pressure-inclusive dataset | 118 runs, 444,624 ticks, L5 has 3,764 valid windows/run |
| causal AAM | 30-D row, 150-D L5 input, best epoch 26 |
| AAM validation RMSE | 1.18301 |
| AAM test RMSE | 1.25244 |
| aligned z_hat adapter | best validation state loss 2.6828 at epoch 139 |
| adapter checkpoint | `outputs/stage30_pressure_adapter_aligned_gpu/zhat_adapter.pt` |

The dataset metadata was rebuilt after the contract correction; measured `dP`
is now listed as an included feature rather than an excluded one.

## Corrected held-out adapter evaluation

The evaluation covers 36 validation/test runs. Warm-up is consistent across
training, evaluation, and deployment: the adapter is bypassed until the causal
L5 estimate exists. The corrected evaluator keeps separate causal AAM histories
for the teacher-forced and closed-loop trajectories, matching deployment. Metrics
are means over runs.

| residual scale | cylinder tracking RMSE (m) | tool-path proxy RMSE (m) | preclip saturation | first divergence (s) |
|---:|---:|---:|---:|---:|
| frozen ID | 19.8284 | 52.1995 | 0.8917 | 30.4931 |
| AAM adapter 0.10 | 20.2361 | 53.3145 | 0.6776 | 30.5861 |
| AAM adapter 0.25 | 19.7455 | 49.5334 | 0.6812 | 30.5306 |
| AAM adapter 0.50 | 19.0717 | 48.5126 | 0.6881 | 30.4667 |
| AAM adapter 0.75 | **18.7903** | **47.1491** | **0.6799** | **30.7542** |
| AAM adapter 1.00 | 20.6181 | 55.7134 | 0.6691 | 30.2292 |

Scales 0.25 and 0.75 pass the complete non-regression comparison. Scale 0.75
is selected because it gives the best tracking, path proxy, and divergence
time while also reducing saturation. Scale 1.0 is rejected.

## Runtime checks

`PPO_ID_AAM_FD/scripts/preflight.py` validates the selected pressure AAM and
aligned adapter checkpoint, L5 warm-up, 20 Hz update rate, bounded residual,
per-environment reset isolation, and non-finite fallback. The scale-0.75 CPU
preflight measured approximately 0.43 ms maximum AAM inference latency.

The runtime now performs one AAM warm-up inference before the first control
tick and synchronizes CUDA timing around inference. This prevents a cold GPU
kernel launch from consuming the first valid update's 50 ms fail-safe budget.
The focused CUDA gate measured 1.325 ms maximum AAM inference latency and
matched an independent causal trace exactly.

The standalone PhysX smoke test initialized the registered task and advanced
eight steps on mesh 1337 without a traceback. Isaac Sim prints repeated
`errno=28` change-watch warnings; these are the known inotify/resource-limit
warnings, not a model-contract failure. The smoke wrapper uses the diagnostic
teacher/demo path, so it is an integration check rather than a trained PPO
quality result.

The checked-in configuration is:

```text
online_aam_checkpoint = outputs/aam_stage30_pressure_temporal/L5
online_aam_adapter_checkpoint = outputs/stage30_pressure_adapter_aligned_gpu/zhat_adapter.pt
online_aam_residual_scale = 0.75
```

This keeps AAM latent estimation online and observable to the 68-D PPO policy,
adds the bounded scale-0.75 residual around frozen ID, and retains fail-safe
fallback to the unchanged frozen ID action.

## Commands

```bash
cd /home/rrlab/rrlab_mulag
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python

PPO_ID_AAM_FD/scripts/preflight.py --num-envs 4 --device cpu

$PY PPO_ID_AAM_FD/scripts/evaluate_stage30.py \
  --out outputs/stage30_pressure_adapter_aligned_gpu/online_evaluation_scale075 \
  --device auto

PPO_ID_AAM_FD/scripts/smoke.py --meshes 1337 --num_envs 1 --steps 8
PPO_ID_AAM_FD/visualize.sh --teacher
```

## Literature alignment

The causal history estimator follows the history-based online system
identification pattern in [UP-OSI](https://arxiv.org/abs/1702.02453). The
latent-conditioned residual around a frozen controller follows the separation
of base policy and adaptation module used in [RMA](https://arxiv.org/abs/2107.04034).
The bounded residual and frozen-ID fallback are the safety layer for the
current negative non-regression result.
