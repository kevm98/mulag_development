#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/rrlab/rrlab_mulag
aam_root="$repo_root/AAM_module"
python_bin=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
frozen="$aam_root/datasets/master_v3_stage22_wide_frozen"
frozen_sdz="$repo_root/sd_model/stage23_frozen/dz2_nodp_holdq"
no_latent="$repo_root/outputs/sd_z_stage22_v6_wide/none_nodp_holdq"
out_root="$repo_root/outputs/aam_stage26"
lag=40  # Stage 25 selection: 2.0 s

mkdir -p "$out_root"
echo "$$" > "$out_root/queue.pid"

aam_dir="$out_root/L${lag}"
if [[ -e "$aam_dir/metrics.json" && -e "$aam_dir/aam.pt" ]]; then
    echo "refusing existing AAM model: $aam_dir" >&2
    exit 2
fi

"$python_bin" "$aam_root/tools/train_aam.py" \
    --root "$frozen" --lag "$lag" --out "$aam_dir"
"$python_bin" "$aam_root/tools/predict_aam.py" \
    --root "$frozen" --model-dir "$aam_dir" \
    --out "$aam_dir/zhat_by_tick.json"

"$python_bin" "$repo_root/sd_model/evaluation/evaluate_sd_z_rollout.py" \
    --root "$frozen" --model-dir "$frozen_sdz" \
    --latent-mode true zhat zero shuffled mean \
    --zhat-json "$aam_dir/zhat_by_tick.json" \
    --horizons 1 5 10 20 --q-rollout hold-start \
    --out "$out_root/rollout_gate.json"

"$python_bin" "$repo_root/sd_model/evaluation/evaluate_sd_z_rollout.py" \
    --root "$frozen" --model-dir "$no_latent" \
    --latent-mode true \
    --horizons 1 5 10 20 --q-rollout hold-start \
    --out "$out_root/rollout_none.json"

"$python_bin" "$repo_root/sd_model/evaluation/compare_sd_z_rollouts.py" \
    --candidate "$out_root/rollout_gate.json" --candidate-mode zhat \
    --control "$out_root/rollout_none.json" --control-mode true \
    --horizon 20 --out "$out_root/zhat_vs_none_h20.json"

"$python_bin" "$aam_root/tools/report_aam_stage26.py" \
    --aam-metrics "$aam_dir/metrics.json" \
    --zhat-stats "$aam_dir/zhat_stats.json" \
    --rollout "$out_root/rollout_gate.json" \
    --vs-none "$out_root/zhat_vs_none_h20.json" \
    --out "$out_root/report.json"

date -Is > "$out_root/COMPLETE"
echo "Stage 26 complete -> $out_root/report.json"
