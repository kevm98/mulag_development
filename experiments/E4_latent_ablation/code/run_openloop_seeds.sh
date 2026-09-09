#!/usr/bin/env bash
set -uo pipefail

# Claim 2, closed with n=5: does the jointly fine-tuned AAM+SD beat a no-latent
# SD trained on the SAME open-loop data, repeatably?
#
# The pilot (sd_none seed 20260820 vs e2e seed 20260816) gave 0.02438 vs 0.01914
# (-21.5%) but the seeds were not matched and n=1. The prior `zhat<frozen`
# result was 3/5 with seed 0 carrying the mean, so a single large margin is not
# sufficient here. This runs a matched pair per seed and reports the paired
# distribution.
#
# Scheduling: train_sd_z.py takes the GPU (device auto), train_sd_aam_e2e.py is
# CPU-only. The two families therefore run concurrently on different hardware;
# within the CPU family, seeds run JOBS-at-a-time with a thread cap so they do
# not thrash the 12 cores.

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
py="$HOME/miniconda3/envs/env_isaaclab/bin/python"
aam="$repo/AAM_module"
frozen="$aam/datasets/rom_openloop_v1_frozen"
out="$repo/outputs/stage31_openloop/seeds"
log="$out/logs"
mkdir -p "$log"

SEEDS=(20260820 20260821 20260822 20260823 20260824)
JOBS=3
THREADS=4

[[ -d "$frozen" ]] || { echo "missing $frozen" >&2; exit 2; }

# ---- baselines, GPU, sequential -------------------------------------------
(
  for s in "${SEEDS[@]}"; do
    d="$out/sd_none_s$s"
    [[ -f "$d/metrics.json" ]] && { echo "skip sd_none s$s"; continue; }
    echo "[sd_none s$s] start $(date +%T)"
    "$py" "$repo/sd_model/training/train_sd_z.py" \
      --root "$frozen" --history-lag 6 --belief-dim 16 \
      --unroll 5 --l1-weight 1.0 --rollout-weight 1.0 \
      --state-loss-weight 1.0 --rollout-q-mode hold-start \
      --rollout-ramp-epochs 40 --val-unroll 5 --epochs 200 --patience 25 \
      --seed "$s" --latent-source none --sample-mask \
      --out "$d" > "$log/sd_none_s$s.log" 2>&1 \
      || echo "[sd_none s$s] FAILED (see $log/sd_none_s$s.log)"
    echo "[sd_none s$s] done $(date +%T)"
  done
) &
gpu_pid=$!

# ---- e2e, CPU, JOBS at a time ---------------------------------------------
(
  export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
  running=0
  for s in "${SEEDS[@]}"; do
    d="$out/e2e_s$s"
    [[ -f "$d/metrics.json" ]] && { echo "skip e2e s$s"; continue; }
    (
      echo "[e2e s$s] start $(date +%T)"
      "$py" "$repo/sd_model/training/train_sd_aam_e2e.py" \
        --root "$frozen" --lag 20 --latent-dim 2 --epochs 60 --device cpu \
        --unroll 5 --rollout-weight 1.0 --l1-weight 1.0 --tf-floor 0.2 \
        --rollout-ramp-epochs 20 --seed "$s" \
        --out "$d" > "$log/e2e_s$s.log" 2>&1 \
        || echo "[e2e s$s] FAILED (see $log/e2e_s$s.log)"
      echo "[e2e s$s] done $(date +%T)"
    ) &
    running=$((running + 1))
    if (( running >= JOBS )); then wait -n; running=$((running - 1)); fi
  done
  wait
) &
cpu_pid=$!

wait $gpu_pid $cpu_pid
echo
echo "== paired comparison, n=${#SEEDS[@]} =="
"$py" "$repo/experiments/E4_latent_ablation/code/summarise_openloop_seeds.py" --root "$out"
