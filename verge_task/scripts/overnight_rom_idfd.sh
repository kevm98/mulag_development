#!/usr/bin/env bash
# Overnight: finish the two LEGACY pipelines end to end, unattended.
#
#   1. wait for the running ROM PPO to finish
#   2. score its final checkpoint on the 8-gate contract (6 held-out meshes)
#   3. train ID/FD PPO from its own distilled v3 checkpoint
#   4. score that on the same contract
#   5. write ONE summary comparing every arm measured so far
#
# Each stage is guarded: a missing checkpoint or a failed score is recorded and
# the run continues, because a stall at 02:00 that blocks the remaining stages
# wastes the whole night. Every number lands in a file; nothing depends on this
# script's stdout being read.
set -uo pipefail

ROOT=/home/rrlab/rrlab_mulag
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
SP=/tmp/rrlab_scratch
OUT=$ROOT/PPO_ROM/reports/frontier
SUMMARY=$ROOT/reports/OVERNIGHT_SUMMARY.md
cd "$ROOT"
mkdir -p "$ROOT/reports"

say() { echo "[OVERNIGHT $(date +%H:%M:%S)] $*"; }

# --- 1. wait for the in-flight ROM PPO -------------------------------------
say "waiting for ROM PPO to finish"
while pgrep -f "supervise_training.sh|[t]rain.py --resume" >/dev/null; do sleep 60; done
say "ROM PPO finished"

ROM_RUN=$ROOT/logs/rsl_rl/ppo3_1_sole/rom_v3_baseline_ppo
ROM_CKPT=$(find "$ROM_RUN" -name "model_*.pt" 2>/dev/null \
    | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
ROM_CKPT="$ROM_RUN/model_${ROM_CKPT}.pt"
say "ROM final checkpoint: $ROM_CKPT"

# --- 2. score it ------------------------------------------------------------
if [ -f "$ROM_CKPT" ]; then
  say "scoring ROM PPO"
  MESHES="1337 17 2024 4242 555 909" bash PPO_ROM/scripts/score_per_mesh.sh \
      romppo "$ROM_CKPT" 480 32 > "$SP/score_romppo.log" 2>&1
else
  say "ROM checkpoint missing, skipping score"
fi

# --- 3. ID/FD PPO -----------------------------------------------------------
say "training ID/FD PPO from distilled v3"
NUM_ENVS=256 STAGE_ITERATIONS=600 FREEZE_ITERATIONS=150 \
  STUDENT=$ROOT/logs/rsl_rl/ppo4_1_sole/student_distilled_v3.pt \
  LOG_DIR=$ROOT/logs/rsl_rl/ppo4_1_sole/idfd_v3_baseline_ppo \
  TASK=RRLAB-Mulag-Verge-IDFD-v0 \
  bash PPO_ROM/scripts/supervise_training.sh > "$SP/idfd_ppo.log" 2>&1
say "ID/FD PPO finished"

IDFD_RUN=$ROOT/logs/rsl_rl/ppo4_1_sole/idfd_v3_baseline_ppo
IDFD_CKPT=$(find "$IDFD_RUN" -name "model_*.pt" 2>/dev/null \
    | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
IDFD_CKPT="$IDFD_RUN/model_${IDFD_CKPT}.pt"

# --- 4. score it ------------------------------------------------------------
if [ -f "$IDFD_CKPT" ]; then
  say "scoring ID/FD PPO: $IDFD_CKPT"
  MESHES="1337 17 2024 4242 555 909" TASK=RRLAB-Mulag-Verge-IDFD-v0 \
    bash PPO_ROM/scripts/score_per_mesh.sh idfdppo "$IDFD_CKPT" 480 32 \
    > "$SP/score_idfdppo.log" 2>&1
else
  say "ID/FD checkpoint missing, skipping score"
fi

# --- 5. one summary ---------------------------------------------------------
say "writing summary"
"$PY" - "$SUMMARY" <<'PYEOF'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
OUT = Path("/home/rrlab/rrlab_mulag/PPO_ROM/reports/frontier")
ARMS = [
    ("teacher (privileged DLS)", "teacherv3_gates.json"),
    ("ROM   distilled v3",       "romv3_gates.json"),
    ("ROM   distilled + PPO",    "romppo_gates.json"),
    ("ID/FD distilled v3",       "idfdv3_gates.json"),
    ("ID/FD distilled + PPO",    "idfdppo_gates.json"),
]
KEYS = [
    ("worst_mesh_worst_family_mean_abs_lateral_error_m", "lateral",  "<=0.300"),
    ("worst_mesh_worst_family_ground_contact_fraction",  "contact",  ">=0.700"),
    ("worst_mesh_hard_impact_step_fraction",             "impacts",  "<=0.350"),
    ("worst_mesh_mean_abs_cut_height_error_m",           "cut",      "<=0.040"),
    ("worst_mesh_p01_head_alignment",                    "head",     ">=0.900"),
]
rows = []
for label, fname in ARMS:
    p = OUT / fname
    if not p.is_file():
        rows.append((label, None, None, {}))
        continue
    g = json.loads(p.read_text())["gates"]
    rows.append((label, g["passed"], g["failing"], g["metrics"]))

lines = ["# Overnight results: legacy ROM and ID/FD pipelines", ""]
lines.append("Widened (4.4x amplitude, 3.0 m strip), XY trajectories, per-env")
lines.append("randomised width ladder. 6 held-out meshes, 32 envs, 480 steps.")
lines.append("")
hdr = f"| {'arm':<26} | pass | " + " | ".join(f"{n:>8}" for _, n, _ in KEYS) + " |"
lines += [hdr, "|" + "-" * (len(hdr) - 2) + "|"]
for label, passed, failing, m in rows:
    if passed is None:
        lines.append(f"| {label:<26} | MISSING | " + " | ".join(f"{'-':>8}" for _ in KEYS) + " |")
        continue
    cells = " | ".join(f"{m.get(k, float('nan')):8.4f}" for k, _, _ in KEYS)
    lines.append(f"| {label:<26} | {'YES ' if passed else 'NO  '} | {cells} |")
lines += ["", "Gates: " + ", ".join(f"{n} {t}" for _, n, t in KEYS), ""]
for label, passed, failing, _ in rows:
    if passed is False:
        lines.append(f"- **{label} FAILS**: {failing}")
lines += [
    "",
    "## What is NOT in these numbers",
    "",
    "Every PPO arm here is REWARD-ONLY PPO -- the historical-regression baseline.",
    "The demonstration-regularized formulation (teacher queried on policy-visited",
    "states, advantage/state-filtered imitation, KL trust region, block",
    "accept/rollback) is NOT wired into training. Its primitives exist and are",
    "unit-tested in verge_task/demo_regularized.py; the integration points are",
    "the rollout teacher-label buffer and a wrapper around runner.alg.update.",
    "",
    "So 'distilled + PPO' rows answer 'does plain PPO still regress on the",
    "corrected plant', which is the control -- not the proposed design.",
]
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PYEOF

say "done -- summary at $SUMMARY"
