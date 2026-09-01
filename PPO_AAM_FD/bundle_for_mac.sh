#!/usr/bin/env bash
# Collect everything needed to reproduce and write up the adaptation result.
#
# Excludes the ~653 MB of intermediate training checkpoints and keeps only the
# ones a result actually depends on: the distilled base, the zeroed Phase 1
# init, and the final model of every trained arm. Everything else in
# logs/rsl_rl is reproducible from those plus the commands in memory.md.
#
# Usage:  PPO_AAM_FD/bundle_for_mac.sh [output.tgz]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-/tmp/rrlab_adaptation_bundle.tgz}
LIST=$(mktemp)

# --- source needed to reproduce ------------------------------------------
for d in PPO_AAM_FD verge_task PPO_ROM PPO_ID_FD mulag_plant; do
  [ -d "$d" ] && find "$d" -type f \
    \( -name '*.py' -o -name '*.sh' -o -name '*.md' -o -name '*.json' -o -name '*.yaml' \) \
    -not -path '*/__pycache__/*' >> "$LIST"
done

# --- results, figures, write-ups -----------------------------------------
[ -d reports ] && find reports -type f >> "$LIST"
[ -d outputs/videos ] && find outputs/videos -name '*.mp4' >> "$LIST"
[ -d outputs/rif_delay ] && find outputs/rif_delay -maxdepth 1 -name '*.json' >> "$LIST"
[ -d experiments ] && find experiments -type f -name '*.md' >> "$LIST"

# --- only the checkpoints a result depends on ----------------------------
CK=logs/rsl_rl/ppo_id_aam_fd
for f in \
  "$CK/distilled_v1/model_distilled_70b.pt" \
  "$CK/distilled_v1/phase1_init_70.pt" ; do
  [ -f "$f" ] && echo "$f" >> "$LIST"
done
# final model of every trained arm (residual_*, seed*_*)
for d in "$CK"/residual_* "$CK"/seed*_* "$CK"/rebal*_* "$CK"/long*_* ; do
  [ -d "$d" ] || continue
  for f in "$d/model_399.pt" "$d/model_799.pt" ; do
    [ -f "$f" ] && echo "$f" >> "$LIST"
  done
done
# tensorboard event files are small and carry the training curves
find "$CK" -name 'events.out.tfevents.*' -size -20M 2>/dev/null >> "$LIST" || true

sort -u "$LIST" -o "$LIST"
COUNT=$(wc -l < "$LIST")
tar czf "$OUT" -T "$LIST"
rm -f "$LIST"
echo "bundled $COUNT files -> $OUT ($(du -h "$OUT" | cut -f1))"
echo
echo "Pull it from the Mac with:"
echo "  scp <thisbox>:$OUT ~/Downloads/"
echo "  tar xzf ~/Downloads/$(basename "$OUT") -C ~/path/to/local/rrlab_mulag"
