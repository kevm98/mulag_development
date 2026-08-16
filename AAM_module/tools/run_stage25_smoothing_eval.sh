#!/usr/bin/env bash
set -euo pipefail

# Inference-only Stage 25 follow-up. The AAM checkpoints and raw causal
# predictions must already exist under outputs/aam_stage25_causal/.
repo_root=/home/rrlab/rrlab_mulag
aam_root="$repo_root/AAM_module"
python_bin=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
frozen="$aam_root/datasets/master_v3_stage22_wide_frozen"
frozen_model="$repo_root/sd_model/stage23_frozen/dz2_nodp_holdq"
out_root="$repo_root/outputs/aam_stage25_rolling20"

mkdir -p "$out_root"
for lag in 5 10 20 40; do
    "$python_bin" "$aam_root/tools/smooth_aam_predictions.py" \
        --input "$repo_root/outputs/aam_stage25_causal/L$lag/zhat_by_tick.json" \
        --method rolling --window 20 \
        --out "$out_root/zhat_L${lag}_rolling20.json"
    "$python_bin" "$aam_root/tools/evaluate_sd_z_rollout.py" \
        --root "$frozen" --model-dir "$frozen_model" \
        --latent-mode zhat --zhat-json "$out_root/zhat_L${lag}_rolling20.json" \
        --horizons 1 5 10 20 --q-rollout hold-start \
        --out "$out_root/rollout_L$lag.json"
done

"$python_bin" "$aam_root/tools/select_aam_history.py" \
    --results "$out_root" --lags 5 10 20 40 \
    --out "$out_root/selection.json"

date -Is > "$out_root/COMPLETE"
echo "Stage 25 rolling20 evaluation complete -> $out_root/selection.json"
