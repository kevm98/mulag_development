#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/rrlab/rrlab_mulag
aam_root="$repo_root/AAM_module"
python_bin=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
frozen="$aam_root/datasets/master_v3_stage22_wide_frozen"
frozen_model="$repo_root/sd_model/stage23_frozen/dz2_nodp_holdq"
out_root="$repo_root/outputs/aam_stage25"

mkdir -p "$out_root"
echo "$$" > "$out_root/queue.pid"

lags=(5 10 20 40)
for lag in "${lags[@]}"; do
    lag_dir="$out_root/L$lag"
    if [[ -e "$lag_dir/metrics.json" && -e "$lag_dir/aam.pt" ]]; then
        echo "refusing existing AAM model: $lag_dir" >&2
        exit 2
    fi
    "$python_bin" "$aam_root/tools/train_aam.py" \
        --root "$frozen" --lag "$lag" --out "$lag_dir"
    "$python_bin" "$aam_root/tools/predict_aam.py" \
        --root "$frozen" --model-dir "$lag_dir" \
        --out "$lag_dir/zhat_by_tick.json"
    "$python_bin" "$repo_root/sd_model/evaluation/evaluate_sd_z_rollout.py" \
        --root "$frozen" --model-dir "$frozen_model" \
        --latent-mode zhat --zhat-json "$lag_dir/zhat_by_tick.json" \
        --horizons 1 5 10 20 --q-rollout hold-start \
        --out "$out_root/rollout_L$lag.json"
done

"$python_bin" "$aam_root/tools/select_aam_history.py" \
    --results "$out_root" --lags "${lags[@]}" \
    --out "$out_root/selection.json"

date -Is > "$out_root/COMPLETE"
echo "Stage 25 sweep complete -> $out_root/selection.json"
