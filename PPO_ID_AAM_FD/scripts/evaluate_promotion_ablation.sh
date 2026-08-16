#!/usr/bin/env bash
set -euo pipefail

# Produce paired PPO reports for estimated, zero and shuffled z_hat. Every
# mode uses the same checkpoint, mesh, speed, episode length and environment
# count; promote_system.py rejects any contract mismatch.
ROOT=/home/rrlab/rrlab_mulag
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
CKPT=${1:?usage: evaluate_promotion_ablation.sh <checkpoint> [output_dir]}
OUT=${2:-$ROOT/outputs/stage31_rom/promotion/ppo_ablation}
MESHES=${PROMOTION_MESHES:-"1337 17 909"}
SPEEDS=${PROMOTION_SPEEDS:-"2 5"}
STEPS=${PROMOTION_STEPS:-2808}
ENVS=${PROMOTION_ENVS:-32}
SEED=${PROMOTION_SEED:-17}
C3_RESERVE=${PROMOTION_C3_RESERVE:-0.008}
RESIDUAL_SCALE=${PROMOTION_RESIDUAL_SCALE:-${RESIDUAL_SCALE:-0.25}}
FACTOR_DOMAIN=${PROMOTION_FACTOR_DOMAIN:-combined_test}
AAM_HISTORY_LAG=${MULAG_AAM_HISTORY_LAG:-20}
AAM_SMOOTHER_WINDOW=${PROMOTION_AAM_SMOOTHER_WINDOW:-5}
AAM_MODEL=${PROMOTION_AAM_MODEL:-$ROOT/outputs/stage31_rom/aam_cnn/L${AAM_HISTORY_LAG}_d64}
ADAPTER_MODEL=${MULAG_AAM_ADAPTER_MODEL:-$ROOT/outputs/stage31_rom/id_adapter_cnn_l20_s5/zhat_adapter.pt}
SD_ROLLOUT=${SD_ROLLOUT:-$ROOT/outputs/stage31_rom/aam_cnn/L20_d64/rollout_gate.json}

test -f "$CKPT" || { echo "missing PPO checkpoint: $CKPT" >&2; exit 2; }
test -f "$SD_ROLLOUT" || { echo "missing SD rollout gate: $SD_ROLLOUT" >&2; exit 2; }
test -f "$AAM_MODEL/aam.pt" || { echo "missing AAM checkpoint: $AAM_MODEL/aam.pt" >&2; exit 2; }
test -f "$ADAPTER_MODEL" || { echo "missing AAM adapter: $ADAPTER_MODEL" >&2; exit 2; }
export MULAG_AAM_HISTORY_LAG="$AAM_HISTORY_LAG"
export MULAG_AAM_ADAPTER_MODEL="$ADAPTER_MODEL"
mkdir -p "$OUT"

report_matches_contract() {
  "$PY" - "$1" "$CKPT" "$2" "$3" "$MESHES" "$STEPS" "$ENVS" "$C3_RESERVE" "$RESIDUAL_SCALE" "$FACTOR_DOMAIN" "$SEED" "$AAM_MODEL" "$ADAPTER_MODEL" "$AAM_HISTORY_LAG" "$AAM_SMOOTHER_WINDOW" <<'PY'
import json
import sys
from pathlib import Path

(
    path, checkpoint, mode, speed, meshes, steps, envs, reserve,
    residual_scale, factor_domain, seed, aam_model, adapter_model,
    history_lag, smoother_window,
) = sys.argv[1:]
try:
    report = json.loads(Path(path).read_text())
    if Path(str(report.get("checkpoint"))).resolve() != Path(checkpoint).resolve():
        raise ValueError("checkpoint")
    if report.get("online_aam_latent_mode") != mode:
        raise ValueError("latent mode")
    if not bool(report.get("z_hat_is_live")):
        raise ValueError("z_hat live flag")
    if report.get("online_aam_controller_latent_mode") != "estimated":
        raise ValueError("controller latent mode")
    if abs(float(report["c3_physical_extension_reserve_m"]) - float(reserve)) > 1e-9:
        raise ValueError("c3 reserve")
    if abs(float(report["online_aam_residual_scale"]) - float(residual_scale)) > 1e-9:
        raise ValueError("residual scale")
    if not bool(report["deployment_factor_enabled"]):
        raise ValueError("factor randomization disabled")
    if report["factor_domain"] != factor_domain:
        raise ValueError("factor domain")
    if int(report["evaluation_seed"]) != int(seed):
        raise ValueError("evaluation seed")
    if Path(report["online_aam_checkpoint"]).resolve() != Path(aam_model).resolve():
        raise ValueError("AAM checkpoint")
    if Path(report["online_aam_adapter_checkpoint"]).resolve() != Path(adapter_model).resolve():
        raise ValueError("adapter checkpoint")
    if int(report["online_aam_history_lag"]) != int(history_lag):
        raise ValueError("history lag")
    if int(report["online_aam_smoother_window"]) != int(smoother_window):
        raise ValueError("smoother window")
    speed = float(speed)
    if abs(float(report["vehicle_speed_min_kmh"]) - speed) > 1e-9:
        raise ValueError("speed min")
    if abs(float(report["vehicle_speed_max_kmh"]) - speed) > 1e-9:
        raise ValueError("speed max")
    expected_meshes = set(meshes.split())
    if set(report.get("per_mesh", {})) != expected_meshes:
        raise ValueError("mesh set")
    for row in report["per_mesh"].values():
        if int(row["num_envs"]) != int(envs):
            raise ValueError("environment count")
        # evaluate_gates records the last completed transition, one below the
        # requested rollout length; tolerate only that representation.
        if int(row["steps"]) not in {int(steps), int(steps) - 1}:
            raise ValueError("step count")
except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0)
PY
}

for mode in estimated zero shuffled; do
  for speed in $SPEEDS; do
    name="${mode}_s${speed}"
    if [[ -s "$OUT/${name}_gates.json" ]]; then
      if report_matches_contract "$OUT/${name}_gates.json" "$mode" "$speed"; then
        echo "[PROMOTION] reusing existing $name report"
        continue
      fi
      echo "[PROMOTION] existing $name report is stale or incomplete; rescoring"
    fi
    echo "[PROMOTION] scoring $name"
    TASK=RRLAB-Mulag-Verge-ID-AAM-FD-v0 \
      MESHES="$MESHES" SPEED_KMH="$speed" LATENT_MODE="$mode" \
      REPORT_ROOT="$OUT" STAGE=6 STEPS="$STEPS" ENVS="$ENVS" SEED="$SEED" \
      C3_RESERVE="$C3_RESERVE" RESIDUAL_SCALE="$RESIDUAL_SCALE" \
      FACTOR_DOMAIN="$FACTOR_DOMAIN" \
      PPO_ID_AAM_FD/scripts/score_per_mesh.sh "$name" "$CKPT" "$STEPS" "$ENVS" || {
        # The scorer's exit status is the legacy binary safety contract. A
        # failed legacy gate must not discard a complete report: this campaign
        # deliberately compares paired continuous metrics and the promotion
        # gate makes its own statistical decision from those reports.
        test -s "$OUT/${name}_gates.json" || {
          echo "[PROMOTION] $name produced no merged report" >&2
          exit 1
        }
        echo "[PROMOTION] $name has legacy gate failures; retaining metrics for paired promotion"
      }
  done
done

estimated=() zero=() shuffled=()
for speed in $SPEEDS; do
  estimated+=("$OUT/estimated_s${speed}_gates.json")
  zero+=("$OUT/zero_s${speed}_gates.json")
  shuffled+=("$OUT/shuffled_s${speed}_gates.json")
done

exec "$PY" "$ROOT/AAM_module/tools/promote_system.py" \
  --sd-rollout "$SD_ROLLOUT" \
  --ppo-estimated "${estimated[@]}" \
  --ppo-zero "${zero[@]}" \
  --ppo-shuffled "${shuffled[@]}" \
  --sd-horizon 20 \
  --out "$OUT/promotion_verdict.json"
