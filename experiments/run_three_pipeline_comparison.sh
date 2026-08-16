#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="/home/rrlab/miniconda3/envs/env_isaaclab/bin/python"
OUT="$ROOT/experiments/three_pipeline_tracking"
TRACE="$OUT/traces"
REPORT="$OUT/reports"

mkdir -p "$TRACE" "$REPORT"

COMMON=(
  --meshes 909
  --num-envs 1
  --steps 2808
  --stage 6
  --seed 17
  --trajectory-shape double_s_curve
  --tracking-amplitude 2.2
  --headless
)

"$PY" "$ROOT/PPO_ROM/scripts/evaluate_gates.py" \
  "$ROOT/logs/rsl_rl/ppo3_1_sole/rom_wide_reference_kl_block2_v1/model_selected.pt" \
  "${COMMON[@]}" \
  --trace-out "$TRACE/rom.npz" \
  --report "$REPORT/rom.json"

"$PY" "$ROOT/PPO_ROM/scripts/evaluate_gates.py" \
  "$ROOT/logs/rsl_rl/ppo4_1_sole/idfd_promotable_v1/model_525.pt" \
  "${COMMON[@]}" \
  --task RRLAB-Mulag-Verge-IDFD-v0 \
  --trace-out "$TRACE/idfd.npz" \
  --report "$REPORT/idfd.json"

"$PY" "$ROOT/PPO_ROM/scripts/evaluate_gates.py" \
  "$ROOT/logs/rsl_rl/ppo_id_aam_fd/stage31_idfd_aam_adapt_v1/model_24.pt" \
  "${COMMON[@]}" \
  --task RRLAB-Mulag-Verge-ID-AAM-FD-v0 \
  --online-aam-residual-scale 0.25 \
  --trace-out "$TRACE/aam.npz" \
  --report "$REPORT/aam.json"

"$PY" "$ROOT/experiments/plot_three_pipeline_errors.py" \
  --trace-dir "$TRACE" \
  --output-dir "$OUT/plots" \
  --shape double_s_curve \
  --mesh 909 \
  --no-show

