#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/rrlab/rrlab_mulag
python_bin=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
dataset="$repo_root/AAM_module/datasets/master_v2_frozen"
v3_root="$repo_root/outputs/sd_z_stage22_v3"
v4_root="$repo_root/outputs/sd_z_stage22_v4"
encoder_dir="$v3_root/dz2_nodp_ramp60"
none_dir="$v3_root/none_nodp_ramp60"
direct_dir="$v4_root/direct6_nodp_ramp60_seed20260809"

mkdir -p "$v4_root" "$direct_dir"
echo "$$" > "$v4_root/queue.pid"
if [[ -e "$direct_dir/metrics.json" || -e "$direct_dir/sd.pt" ]]; then
    echo "refusing to overwrite an existing direct-oracle run: $direct_dir" >&2
    exit 2
fi

# Re-score the completed v3 arms with controls constructed exclusively from
# training runs and with both explicit future-q contracts.
for q_protocol in recorded hold-start; do
    "$python_bin" "$repo_root/AAM_module/tools/evaluate_sd_z_rollout.py" \
        --root "$dataset" \
        --model-dir "$encoder_dir" \
        --latent-mode true zero shuffled random mean nominal \
        --horizons 1 5 10 20 \
        --q-rollout "$q_protocol" \
        --out "$encoder_dir/rollout_train_controls_${q_protocol}.json"

    "$python_bin" "$repo_root/AAM_module/tools/evaluate_sd_z_rollout.py" \
        --root "$dataset" \
        --model-dir "$none_dir" \
        --latent-mode true \
        --horizons 1 5 10 20 \
        --q-rollout "$q_protocol" \
        --out "$none_dir/rollout_${q_protocol}.json"
done

# Oracle-ceiling arm: identical v3 observation, belief, q, state-loss, and
# curriculum protocol, but the normalized six physical factors condition the
# SD directly instead of passing through a learned two-dimensional encoder.
"$python_bin" "$repo_root/AAM_module/tools/train_sd_z.py" \
    --root "$dataset" \
    --latent-source direct --latent-dim 6 --obs-drop 8,9,10,11 \
    --unroll 10 --l1-weight 1.0 --rollout-weight 1.0 --state-loss-weight 1.0 \
    --rollout-ramp-epochs 60 --epochs 300 --patience 30 --seed 20260809 \
    --out "$direct_dir"

for q_protocol in recorded hold-start; do
    "$python_bin" "$repo_root/AAM_module/tools/evaluate_sd_z_rollout.py" \
        --root "$dataset" \
        --model-dir "$direct_dir" \
        --latent-mode true zero shuffled random mean nominal \
        --horizons 1 5 10 20 \
        --q-rollout "$q_protocol" \
        --out "$direct_dir/rollout_train_controls_${q_protocol}.json"

    "$python_bin" "$repo_root/AAM_module/tools/compare_sd_z_rollouts.py" \
        --candidate "$direct_dir/rollout_train_controls_${q_protocol}.json" \
        --control "$none_dir/rollout_${q_protocol}.json" \
        --horizon 20 \
        --out "$v4_root/direct_true_vs_none_h20_${q_protocol}.json"

    "$python_bin" "$repo_root/AAM_module/tools/compare_sd_z_rollouts.py" \
        --candidate "$encoder_dir/rollout_train_controls_${q_protocol}.json" \
        --control "$none_dir/rollout_${q_protocol}.json" \
        --horizon 20 \
        --out "$v4_root/encoder_true_vs_none_h20_${q_protocol}.json"
done

date -Is > "$v4_root/COMPLETE"
