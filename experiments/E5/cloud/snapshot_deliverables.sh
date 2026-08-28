#!/usr/bin/env bash
# Snapshot the E5 work into a versioned local repo on the VM's persistent disk.
#
# This is the on-machine safety net. It is NOT a substitute for getting the
# work onto the Mac: the persistent disk survives `isaac-stop`, but it dies
# with the instance, and git on the Mac is still the only real backup.
#
# Deliberately a SEPARATE directory rather than `git init` in the working tree:
# the Mac copy already has its own .git, and a second one here would collide
# the moment anything is copied across.
#
#   bash experiments/E5/cloud/snapshot_deliverables.sh "message"
set -Eeuo pipefail

SRC=${SRC:-/home/kevinmurera/projects/rrlab_mulag}
DEST=${DEST:-/home/kevinmurera/e5_deliverables}
MSG=${1:-"E5 ROM robustness snapshot $(date -u +%Y-%m-%dT%H:%M:%SZ)"}

PATHS=(
  PPO_ROM/scripts/score_per_mesh.sh
  PPO_ROM/scripts/supervise_training.sh
  PPO_ROM/scripts/select_checkpoint.sh
  PPO_ROM/scripts/preflight.py
  PPO_ROM/scripts/training_status.py
  PPO_ROM/scripts/train.py
  PPO_ROM/scripts/evaluate_gates.py
  experiments/E5/code
  experiments/E5/cloud
  experiments/E5/results
  mulag_plant/runtime/physx_compat.py
  mulag_plant/runtime/mulag_upper_level_env_cfg.py
)

mkdir -p "$DEST"
if [ ! -d "$DEST/.git" ]; then
  git -C "$DEST" init -q
  git -C "$DEST" config user.email "kevin2585m@gmail.com"
  git -C "$DEST" config user.name "Kevin Murera"
fi

for p in "${PATHS[@]}"; do
  if [ -e "$SRC/$p" ]; then
    mkdir -p "$DEST/$(dirname "$p")"
    cp -r "$SRC/$p" "$DEST/$(dirname "$p")/"
  fi
done

# Run outputs can be large and are regenerable; keep the reports and the logs,
# skip the traces.
mkdir -p "$DEST/outputs"
for d in "$SRC"/outputs/e5_*; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  mkdir -p "$DEST/outputs/$name"
  find "$d" -maxdepth 1 -type f \( -name '*.json' -o -name '*.log' \) \
    -exec cp {} "$DEST/outputs/$name/" \; 2>/dev/null || true
done
mkdir -p "$DEST/logs"
cp -r "$SRC/logs/e5_run" "$DEST/logs/" 2>/dev/null || true

git -C "$DEST" add -A
if git -C "$DEST" diff --cached --quiet; then
  echo "[snapshot] nothing changed"
else
  git -C "$DEST" commit -q -m "$MSG"
  echo "[snapshot] committed: $MSG"
fi
git -C "$DEST" --no-pager log --oneline -5
echo "[snapshot] repo at $DEST"
echo "[snapshot] REMINDER: this dies with the instance -- still pull to the Mac."
