#!/usr/bin/env bash
set -euo pipefail

# Stage 31: one fresh ROM-plant collection -> SD/AAM -> ID(AAM)-FD gates.
# ID-FD checkpoints are consumed only for deployment/evaluation; this script
# never launches an ID-FD data collector.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
aam_root="$repo_root/AAM_module"
python_bin="/home/rrlab/miniconda3/envs/env_isaaclab/bin/python"
manifest="$aam_root/results/stage22_wide_manifest.csv"

raw="$aam_root/datasets/rom_deployment_stage31_v1"
frozen="$aam_root/datasets/rom_deployment_stage31_v1_frozen"
out_root="$repo_root/outputs/stage31_rom"
none_dir="$out_root/sd_none"
candidate_dir="$out_root/sd_z2"
frozen_sd="$repo_root/sd_model/stage31_frozen/z2_rom"
aam_rows="$frozen/aam/rom_rows"
aam_data="$frozen/aam/dataset"
aam_dir="$out_root/aam_cnn/L20_d64"
zhat="$out_root/aam_cnn/L20_d64/zhat_all.json"
adapter_dir="$out_root/id_adapter_cnn_l20_s5"
adapter="$adapter_dir/zhat_adapter.pt"
eval_dir="$out_root/deployment_eval"

id_checkpoint="$repo_root/id_fd_model/best_id_cylinder_control.pt"
fd_checkpoint="$repo_root/id_fd_model/best_fd_cylinder_strict.pt"
models_root="$repo_root/id_fd_model/vendor_models"

resume=0
force=0
if [[ "${2:-}" == "--resume" ]]; then resume=1; fi
if [[ "${2:-}" == "--force" || "${3:-}" == "--force" ]]; then force=1; fi

die() { echo "stage31: $*" >&2; exit 2; }
require_file() { [[ -f "$1" ]] || die "missing file: $1"; }
require_dir() { [[ -d "$1" ]] || die "missing directory: $1"; }
refuse_existing() {
    if [[ -e "$1" && "$force" -ne 1 ]]; then
        die "target exists: $1 (use --force only when intentionally replacing it)";
    fi
}

collect() {
    require_file "$manifest"
    if [[ "$resume" -eq 1 ]]; then
        resume_flag=(--resume)
    else
        resume_flag=()
        refuse_existing "$raw"
    fi
    "$python_bin" "$aam_root/tools/collect_verge_classical.py" \
        --headless --actuator rom --manifest "$manifest" --out "$raw" \
        --num-envs 6 --progress-steps 300 --probe prbs \
        --probe-amplitude 0.01 --probe-hold-steps 10 \
        "${resume_flag[@]}"
    "$python_bin" "$aam_root/tools/validate_verge_classical.py" \
        --dataset "$raw" --out "$raw/collection_gate.json"
}

freeze_and_derive() {
    require_dir "$raw"
    require_file "$raw/collection_gate.json"
    if [[ "$force" -eq 1 ]]; then freeze_force=(--force); else freeze_force=(); fi
    "$python_bin" "$aam_root/tools/freeze_dataset.py" \
        --src "$raw" --out "$frozen" --readonly "${freeze_force[@]}"
    "$python_bin" "$aam_root/tools/make_splits.py" --root "$frozen"
    "$python_bin" "$aam_root/tools/build_stage31_rom_datasets.py" \
        --root "$frozen" --sd-out "$frozen/sd/dataset" --aam-out "$aam_rows" \
        --lags 5 10 20 40 --force
}

train_sd() {
    require_dir "$frozen"
    mkdir -p "$out_root"
    train_common=(
        --root "$frozen" --history-lag 6 --belief-dim 16
        --unroll 5 --l1-weight 1.0 --rollout-weight 1.0
        --state-loss-weight 1.0 --rollout-q-mode hold-start
        --rollout-ramp-epochs 40 --val-unroll 5 --epochs 200 --patience 25
        --seed 20260820
    )
    if [[ ! -e "$none_dir" || "$force" -eq 1 ]]; then
        "$python_bin" "$repo_root/sd_model/training/train_sd_z.py" \
            "${train_common[@]}" --latent-source none --out "$none_dir"
    fi
    "$python_bin" "$repo_root/sd_model/evaluation/evaluate_sd_z_rollout.py" \
        --root "$frozen" --model-dir "$none_dir" --latent-mode zero \
        --horizons 1 5 10 20 --q-rollout hold-start \
        --out "$none_dir/rollout_hold-start.json"

    if [[ ! -e "$candidate_dir" || "$force" -eq 1 ]]; then
        "$python_bin" "$repo_root/sd_model/training/train_sd_z.py" \
            "${train_common[@]}" --latent-source encoder --latent-dim 2 \
            --out "$candidate_dir"
    fi
    "$python_bin" "$repo_root/sd_model/evaluation/evaluate_sd_z_rollout.py" \
        --root "$frozen" --model-dir "$candidate_dir" \
        --latent-mode true zero shuffled random mean nominal \
        --horizons 1 5 10 20 --q-rollout hold-start \
        --out "$candidate_dir/rollout_gate.json"
    "$python_bin" "$repo_root/sd_model/evaluation/gate_stage31_sd.py" \
        --rollout "$candidate_dir/rollout_gate.json" \
        --metrics "$candidate_dir/metrics.json" \
        --horizon 20 --out "$candidate_dir/gate_verdict.json"
}

freeze_sd() {
    require_file "$candidate_dir/gate_verdict.json"
    require_file "$none_dir/rollout_hold-start.json"
    if [[ "$force" -eq 1 ]]; then freeze_sd_force=(--force); else freeze_sd_force=(); fi
    "$python_bin" "$repo_root/sd_model/freezing/freeze_stage31_sd.py" \
        --src "$candidate_dir" --out "$frozen_sd" \
        --dataset-root "$frozen" \
        "${freeze_sd_force[@]}"
}

train_aam() {
    require_dir "$frozen_sd"
    "$python_bin" "$aam_root/tools/build_stage31_aam_dataset.py" \
        --root "$frozen" --sd-model "$frozen_sd" --rows "$aam_rows" \
        --out "$aam_data" --lags 5 10 20 40 --force
    refuse_existing "$aam_dir"
    "$python_bin" "$aam_root/tools/train_stage31_aam.py" \
        --root "$frozen" --aam-root "$aam_data" --lag 20 \
        --architecture cnn --hidden 64 --depth 4 \
        --temporal-consistency-weight 1.0 --epochs 150 --patience 20 \
        --seed 20260820 --out "$aam_dir"
    "$python_bin" "$aam_root/tools/gate_stage31_aam.py" \
        --root "$frozen" --aam-root "$aam_data" --model-dir "$aam_dir" \
        --lag 20 --out "$aam_dir/gate_verdict.json"
    "$python_bin" "$aam_root/tools/predict_stage31_aam.py" \
        --root "$frozen" --aam-root "$aam_data" --model-dir "$aam_dir" \
        --splits train val test --out "$zhat"
}

train_adapter() {
    require_file "$zhat"
    require_file "$id_checkpoint"
    refuse_existing "$adapter_dir"
    "$python_bin" "$aam_root/tools/train_stage29_latent_adapter.py" \
        --root "$frozen" --aam-root "$aam_data" --zhat "$zhat" --lag 20 \
        --smoothing-window 5 \
        --models-root "$models_root" --id-checkpoint "$id_checkpoint" \
        --fd-checkpoint "$fd_checkpoint" --variants zhat \
        --epochs 120 --patience 18 --seed 20260820 --out "$adapter_dir"
    require_file "$adapter"
}

runtime_gates() {
    require_file "$adapter"
    mkdir -p "$eval_dir"
    "$python_bin" "$repo_root/PPO_AAM_FD/scripts/preflight.py" \
        --aam-model "$aam_dir" --adapter-model "$adapter" \
        --id-checkpoint "$id_checkpoint" --fd-checkpoint "$fd_checkpoint" \
        --out "$eval_dir/preflight.json"
    "$python_bin" "$repo_root/PPO_AAM_FD/scripts/minimum_online_adaptation_check.py" \
        --aam-model "$aam_dir" --adapter-model "$adapter" \
        --out "$eval_dir/minimum_online_adaptation_check.json"
    "$python_bin" "$aam_root/tools/evaluate_stage31_id_aam.py" \
        --root "$frozen" --zhat "$zhat" --adapter "$adapter" \
        --lag 20 --smoothing-window 5 \
        --id-checkpoint "$id_checkpoint" --models-root "$models_root" \
        --out "$eval_dir/heldout_valve_gate.json"
}

case "${1:-help}" in
    collect) collect ;;
    freeze) freeze_and_derive ;;
    sd) train_sd ;;
    freeze-sd) freeze_sd ;;
    aam) train_aam ;;
    adapter) train_adapter ;;
    gates) runtime_gates ;;
    all)
        collect
        freeze_and_derive
        train_sd
        freeze_sd
        train_aam
        train_adapter
        runtime_gates
        ;;
    paths)
        printf 'raw=%s\nfrozen=%s\nsd=%s\naam=%s\nadapter=%s\n' \
            "$raw" "$frozen" "$frozen_sd" "$aam_dir" "$adapter"
        ;;
    *)
        cat <<'EOF'
Stage 31 ROM-source pipeline

  stage31_rom_pipeline.sh collect [--resume]
  stage31_rom_pipeline.sh freeze
  stage31_rom_pipeline.sh sd
  stage31_rom_pipeline.sh freeze-sd
  stage31_rom_pipeline.sh aam
  stage31_rom_pipeline.sh adapter
  stage31_rom_pipeline.sh gates
  stage31_rom_pipeline.sh all

The physical collector is always RRLAB-Mulag-Verge-ROM-v0 with DLS/PRBS.
No ID-FD dataset is collected. Use --force only to intentionally replace a
stage output; use --resume only for an interrupted ROM collection.
EOF
        ;;
esac
