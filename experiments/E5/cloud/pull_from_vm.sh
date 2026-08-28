#!/usr/bin/env bash
# Pull the VM's E5 work back to the Mac. RUN THIS ON THE MAC, not on the VM --
# the VM has no outbound path to the Mac by design, so the transfer has to be
# initiated from this side.
#
#   bash pull_from_vm.sh            # copy into /Volumes/DATA/rrlab_mulag
#   DEST=/some/other/path bash pull_from_vm.sh
#
# Safe to re-run: it overwrites the same paths and touches nothing else.
set -Eeuo pipefail

DEST=${DEST:-/Volumes/DATA/rrlab_mulag}
CFG=${CFG:-"/Users/kevinmurera/Documents/New project/.gcloud-config/isaaclab-vscode-ssh-config"}
HOST=${HOST:-isaaclab-gcp}
SRC=${SRC:-/home/kevinmurera/projects/rrlab_mulag}

# Code fixes, the cloud harness + environment notes, and every result/log the
# runs have produced so far.
PATHS=(
  # --- code fixes that made the ROM path run off the Legion box -------------
  PPO_ROM/scripts/score_per_mesh.sh
  PPO_ROM/scripts/supervise_training.sh
  PPO_ROM/scripts/select_checkpoint.sh
  PPO_ROM/scripts/preflight.py
  PPO_ROM/scripts/training_status.py
  PPO_ROM/scripts/train.py
  PPO_ROM/scripts/evaluate_gates.py
  mulag_plant/runtime/physx_compat.py
  mulag_plant/runtime/mulag_upper_level_env_cfg.py
  # --- the experiment itself: runner, summariser, plotters ------------------
  experiments/E5/code
  experiments/E5/cloud
  # --- the deliverables. These are the point of the exercise. ---------------
  experiments/E5/results
  experiments/E5/figures
  # --- raw evidence, so every number in the figure can be re-derived --------
  outputs/e5_rom_authority_ladder
  outputs/e5_authority_stage0
  PPO_ROM/reports/final
  logs/e5_run
)

echo "[pull] $HOST:$SRC -> $DEST"
mkdir -p "$DEST"
ssh -F "$CFG" "$HOST" "tar -czf - -C '$SRC' $(printf '%q ' "${PATHS[@]}")" \
  | tar -xzf - -C "$DEST"

echo "[pull] done. Review before committing:"
echo "    cd $DEST && git status"
