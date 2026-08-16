# Minimum online-adaptation demonstration protocol

This is the reduced gate set for showing that the AAM is actually running
online around the frozen ID–FD controller. It is intentionally smaller than a
full Stage 31 thesis comparison. The reduction removes repeated domain and
mesh sweeps; it does not remove causal-integrity or safety checks.

## Required gates

| Gate | What it proves | Result |
|---|---|---|
| M1 | CUDA streaming AAM matches an independent causal L5/20 Hz reference | **PASS** |
| M2 | Changing future rows cannot change an earlier `z_hat` | **PASS** |
| M3 | Five-row warm-up, per-environment reset, bounded residual, finite output, and frozen-ID fallback work | **PASS** |
| M4 | AAM updates are active online and the residual is nonzero but bounded | **PASS** |
| M5 | One short GPU PhysX task run initializes the standalone pipeline and advances | **PASS: execution only** |
| M6 | A paired frozen-ID versus online-AAM result is reported with cylinder, path, saturation, and divergence metrics | **PASS: result recorded; no universal-superiority claim** |

The reproducible CUDA runtime result is in
`outputs/stage30_online_aam/minimum_online_adaptation_check_cuda.json`:

- exact causal-trace maximum error: `0.0`;
- future-perturbed prefix maximum error: `0.0`;
- first ready sample: zero-based tick `5` (the sixth 60 Hz control row);
- maximum measured CUDA AAM inference latency: `1.3253 ms`, below the `50 ms`
  fail-safe limit;
- maximum residual: `0.1125000045`, equal to the configured scale-0.75 bound;
- 14 online AAM updates were executed.

## Minimum plant evidence

The existing corrected Stage 30 evaluation at residual scale `0.75` is the
paired plant-model demonstration over 36 validation/test runs:

| variant | cylinder tracking RMSE | tool-path proxy RMSE | preclip saturation | first divergence |
|---|---:|---:|---:|---:|
| frozen ID | 19.8284 m | 52.1995 m | 0.8917 | 30.4931 s |
| online AAM adapter | 18.7903 m | 47.1491 m | 0.6799 | 30.7542 s |

This demonstrates a measurable online residual effect and lower pooled error
and saturation. It is not evidence of universal improvement: on the reserved
16-run combined test split, tracking/path error regressed at this setting.
Those 16 configurations remain untouched. The claim supported here is
“causal, bounded online adaptation is functioning,” not “AAM always beats
frozen ID.”

## What is deliberately deferred

The following are not required for this focused demonstration:

- the full hydraulic factor sweep across both ROM and ID–FD arms;
- the separate smoothness/monotonicity factor gate;
- all four arms across all five domains and six or more meshes;
- bootstrap confidence intervals and the complete Stage 31 comparison;
- a new PPO training run.

These remain required before making the full thesis-level generalization or
universal ranking claim. A one-mesh smoke run cannot pass quality gates; it is
only the minimum integration check.

## Commands

```bash
cd /home/rrlab/rrlab_mulag
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python

# Minimum causal/safety/adaptation gate on the NVIDIA GPU.
$PY PPO_ID_AAM_FD/scripts/minimum_online_adaptation_check.py \
  --device cuda --num-envs 2 --ticks 72 \
  --out outputs/stage30_online_aam/minimum_online_adaptation_check_cuda.json

# Artifact/runtime preflight on CUDA.
$PY PPO_ID_AAM_FD/scripts/preflight.py \
  --num-envs 4 --device cuda \
  --out outputs/stage30_online_aam/preflight_cuda.json

# Minimum PhysX integration smoke; this is execution-only with one mesh.
$PY PPO_ID_AAM_FD/scripts/smoke.py \
  --meshes 1337 --num_envs 1 --steps 8

# Existing paired held-out evaluation; no training is started.
$PY PPO_ID_AAM_FD/scripts/evaluate_stage30.py \
  --device cuda --residual-scale 0.75 \
  --out outputs/stage30_online_aam/evaluation_scale075
```

## Protocol note

The current implementation and these results use the pressure-inclusive 30-D
AAM row. The attached deployment brief also contains a stricter alternative
that excludes measured `dP` from the deployable AAM. That contract discrepancy
must be resolved before freezing the final thesis protocol; the numbers above
are explicitly the current pressure-inclusive online path.
