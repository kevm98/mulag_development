#!/usr/bin/env bash
set -euo pipefail

repo_root=$HOME/rrlab_mulag
aam_root="$repo_root/AAM_module"
python_bin=$HOME/miniconda3/envs/env_isaaclab/bin/python
manifest="$aam_root/results/stage22_wide_manifest.csv"
collected="$aam_root/datasets/master_v3_stage22_wide"
frozen="$aam_root/datasets/master_v3_stage22_wide_frozen"
model_root="$repo_root/outputs/sd_z_stage22_v6_wide"
none_dir="$model_root/none_nodp_holdq"

mkdir -p "$model_root"
echo "$$" > "$model_root/queue.pid"

"$python_bin" "$aam_root/tools/validate_stage22_collection.py" \
    --manifest "$manifest" --require-full \
    --out "$aam_root/results/stage22_wide_design_gate.json"

if [[ -e "$frozen" ]]; then
    echo "refusing existing frozen target: $frozen" >&2
    exit 2
fi

# Resume is validation-backed: existing runs are skipped only when their
# schema and manifest factor row match exactly.
"$python_bin" "$aam_root/tools/collect_run_structure.py" \
    --manifest "$manifest" --out "$collected" --resume \
    --num-envs 4 \
    --matched-reset-seed --factor-profile stage22_wide \
    --amplitude 0.02 --probe-seed 20260812 --probe-hold-steps 10 \
    --headless

"$python_bin" "$aam_root/tools/validate_stage22_collection.py" \
    --manifest "$manifest" --dataset "$collected" --require-full \
    --out "$aam_root/results/stage22_wide_collection_gate.json"
date -Is > "$collected/COLLECTION_COMPLETE"

"$python_bin" "$aam_root/tools/freeze_dataset.py" \
    --src "$collected" --out "$frozen" --readonly
"$python_bin" "$aam_root/tools/make_splits.py" --root "$frozen"
"$python_bin" "$aam_root/tools/build_timebase_samples.py" \
    --root "$frozen" --force
"$python_bin" "$repo_root/sd_model/dataset_tools/build_sd_dataset.py" \
    --root "$frozen" --force
date -Is > "$frozen/DERIVED_COMPLETE"

train_common=(
    --root "$frozen"
    --obs-drop 8,9,10,11
    --unroll 10
    --l1-weight 1.0
    --rollout-weight 1.0
    --state-loss-weight 1.0
    --rollout-q-mode hold-start
    --rollout-ramp-epochs 60
    --epochs 300
    --patience 30
    --seed 20260813
)

if [[ -e "$none_dir/metrics.json" || -e "$none_dir/sd.pt" ]]; then
    echo "refusing existing no-latent model: $none_dir" >&2
    exit 2
fi
"$python_bin" "$repo_root/sd_model/training/train_sd_z.py" \
    "${train_common[@]}" --latent-source none --out "$none_dir"
"$python_bin" "$repo_root/sd_model/evaluation/evaluate_sd_z_rollout.py" \
    --root "$frozen" --model-dir "$none_dir" --latent-mode true \
    --horizons 1 5 10 20 --q-rollout hold-start \
    --out "$none_dir/rollout_hold-start.json"

for latent_dim in 2 4; do
    candidate_dir="$model_root/dz${latent_dim}_nodp_holdq"
    if [[ -e "$candidate_dir/metrics.json" || -e "$candidate_dir/sd.pt" ]]; then
        echo "refusing existing candidate model: $candidate_dir" >&2
        exit 2
    fi
    "$python_bin" "$repo_root/sd_model/training/train_sd_z.py" \
        "${train_common[@]}" --latent-source encoder \
        --latent-dim "$latent_dim" --out "$candidate_dir"
    "$python_bin" "$repo_root/sd_model/evaluation/evaluate_sd_z_rollout.py" \
        --root "$frozen" --model-dir "$candidate_dir" \
        --latent-mode true zero shuffled random mean nominal \
        --horizons 1 5 10 20 --q-rollout hold-start \
        --out "$candidate_dir/rollout_gate.json"
    "$python_bin" "$repo_root/sd_model/evaluation/compare_sd_z_rollouts.py" \
        --candidate "$candidate_dir/rollout_gate.json" \
        --control "$none_dir/rollout_hold-start.json" --horizon 20 \
        --out "$candidate_dir/true_vs_none_h20.json"
    "$python_bin" "$aam_root/tools/check_stage22_gate.py" \
        --rollout "$candidate_dir/rollout_gate.json" \
        --vs-none "$candidate_dir/true_vs_none_h20.json" \
        --out "$candidate_dir/gate_verdict.json"
    if "$python_bin" - "$candidate_dir/gate_verdict.json" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["strict_pass"] else 1)
PY
    then
        cp "$candidate_dir/gate_verdict.json" "$model_root/PASSED_GATE.json"
        date -Is > "$model_root/COMPLETE"
        exit 0
    fi
done

date -Is > "$model_root/GATE_NOT_PASSED"
date -Is > "$model_root/COMPLETE"
