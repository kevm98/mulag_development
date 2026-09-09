#!/usr/bin/env bash
set -euo pipefail

repo_root=$HOME/rrlab_mulag
python_bin=$HOME/miniconda3/envs/env_isaaclab/bin/python
dataset="$repo_root/AAM_module/datasets/master_v2_frozen"
v5_root="$repo_root/outputs/sd_z_stage22_v5"
none_dir="$v5_root/none_nodp_holdq"

mkdir -p "$v5_root"
echo "$$" > "$v5_root/queue.pid"

train_common=(
    --root "$dataset"
    --obs-drop 8,9,10,11
    --unroll 10
    --l1-weight 1.0
    --rollout-weight 1.0
    --state-loss-weight 1.0
    --rollout-q-mode hold-start
    --rollout-ramp-epochs 60
    --epochs 300
    --patience 30
    --seed 20260810
)

if [[ -e "$none_dir/metrics.json" || -e "$none_dir/sd.pt" ]]; then
    echo "refusing to overwrite existing no-latent arm: $none_dir" >&2
    exit 2
fi
"$python_bin" "$repo_root/learned_models/sd_model/training/train_sd_z.py" \
    "${train_common[@]}" --latent-source none --out "$none_dir"
"$python_bin" "$repo_root/learned_models/sd_model/evaluation/evaluate_sd_z_rollout.py" \
    --root "$dataset" --model-dir "$none_dir" --latent-mode true \
    --horizons 1 5 10 20 --q-rollout hold-start \
    --out "$none_dir/rollout_hold-start.json"

for rank_weight in 0.25 1.0; do
    rank_tag=${rank_weight/./p}
    candidate_dir="$v5_root/dz2_nodp_holdq_rank${rank_tag}"
    if [[ -e "$candidate_dir/metrics.json" || -e "$candidate_dir/sd.pt" ]]; then
        echo "refusing to overwrite existing candidate arm: $candidate_dir" >&2
        exit 2
    fi
    "$python_bin" "$repo_root/learned_models/sd_model/training/train_sd_z.py" \
        "${train_common[@]}" --latent-source encoder --latent-dim 2 \
        --latent-rank-weight "$rank_weight" --latent-rank-margin 0.01 \
        --latent-rank-control both --out "$candidate_dir"
    "$python_bin" "$repo_root/learned_models/sd_model/evaluation/evaluate_sd_z_rollout.py" \
        --root "$dataset" --model-dir "$candidate_dir" \
        --latent-mode true zero shuffled random mean nominal \
        --horizons 1 5 10 20 --q-rollout hold-start \
        --out "$candidate_dir/rollout_gate.json"
    "$python_bin" "$repo_root/learned_models/sd_model/evaluation/compare_sd_z_rollouts.py" \
        --candidate "$candidate_dir/rollout_gate.json" \
        --control "$none_dir/rollout_hold-start.json" --horizon 20 \
        --out "$candidate_dir/true_vs_none_h20.json"
    "$python_bin" "$repo_root/AAM_module/tools/check_stage22_gate.py" \
        --rollout "$candidate_dir/rollout_gate.json" \
        --vs-none "$candidate_dir/true_vs_none_h20.json" \
        --out "$candidate_dir/gate_verdict.json"
done

date -Is > "$v5_root/COMPLETE"
