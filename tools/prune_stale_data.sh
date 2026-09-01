#!/usr/bin/env bash
# Reclaim ~10.5 GB of superseded training data.
#
# Everything this touches is superseded, unreferenced, or reconstructible.
# Every artifact the thesis pins in Appendix C, every scored report behind
# Chapter 7, and the Simscape training export are all committed and pushed, so
# nothing here is the last copy of anything the document depends on.
#
# Run from the repository root. Pass --dry-run to see the effect first.
set -euo pipefail

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

say() { printf '%s\n' "$*"; }
free_now() { df -h . | awk 'NR==2 {print $4}'; }

say "free before: $(free_now)"

# ---------------------------------------------------------------------------
# 1. Intermediate training checkpoints                              ~4.55 GB
#
# 151 runs hold 1958 .pt files; only 76 are referenced by anything. Keep the
# newest of each run so every run stays resumable, and keep everything the
# thesis names. TensorBoard event files and run manifests are untouched.
# ---------------------------------------------------------------------------
say "1/3  pruning intermediate checkpoints"
DRY=$DRY python3 - <<'PY'
import os, re, collections
dry = os.environ.get("DRY") == "1"

# Runs the thesis pins in Appendix C. Never touched, whatever the rule says.
PINNED = {
    "logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt",
    "logs/rsl_rl/ppo_id_aam_fd/rebal17_history/model_399.pt",
    "logs/rsl_rl/ppo_id_aam_fd/rebal23_history/model_399.pt",
    "logs/rsl_rl/ppo_id_aam_fd/rebal101_history/model_399.pt",
    "logs/rsl_rl/ppo_id_aam_fd/rebal42_history/model_399.pt",
    "logs/rsl_rl/ppo_id_aam_fd/rebal7_history/model_399.pt",
    "logs/rsl_rl/ppo_id_aam_fd/rebal17_nohistory/model_399.pt",
    "logs/rsl_rl/ppo_id_aam_fd/headroom_priced_v1/model_299.pt",
    "logs/rsl_rl/ppo3_1_sole/rom_s6_aligned/model_598.pt",
    "logs/rsl_rl/ppo4_1_sole/idfd_s6_aligned/model_598.pt",
}

runs = collections.defaultdict(list)
for dirpath, _, names in os.walk("logs"):
    for n in names:
        if n.endswith(".pt"):
            runs[dirpath].append(n)

def step(name):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else -1

removed = freed = 0
for dirpath, names in runs.items():
    newest = max(names, key=step)          # keep each run resumable
    for n in names:
        path = os.path.join(dirpath, n)
        if n == newest or path in PINNED:
            continue
        freed += os.path.getsize(path)
        removed += 1
        if not dry:
            os.remove(path)
verb = "would remove" if dry else "removed"
print(f"      {verb} {removed} checkpoints, {freed / 1e9:.2f} GB")
PY

# ---------------------------------------------------------------------------
# 2. Superseded checkpoint archive                                   ~4.8 GB
#
# 84 runs from July and early August, all superseded. No run the thesis reports
# is among them. MANIFEST.tsv stays: it records what was there.
# ---------------------------------------------------------------------------
say "2/3  clearing archive/checkpoints_2026-08-16"
if [ -d archive/checkpoints_2026-08-16 ]; then
  if [ "$DRY" = "1" ]; then
    say "      would remove $(du -sh archive/checkpoints_2026-08-16 | cut -f1), keeping MANIFEST.tsv"
  else
    find archive/checkpoints_2026-08-16 -mindepth 1 -maxdepth 1 \
         ! -name MANIFEST.tsv -exec rm -rf {} +
  fi
fi

# ---------------------------------------------------------------------------
# 3. Unreferenced datasets                                           ~1.1 GB
#
# rom_deployment_stage31_v1 is the pre-freeze working copy; every result uses
# the _frozen one. The two rom_phase2 sets are referenced nowhere at all.
#
# NOT touched, and must not be: archive/model_training_trees, which holds
# ID_cylinder_data.mat and FD_cylinder_data.mat, the export the inverse and
# forward models are fitted on.
# ---------------------------------------------------------------------------
say "3/3  removing unreferenced datasets"
for d in AAM_module/datasets/rom_deployment_stage31_v1 \
         AAM_module/datasets/rom_phase2_v1_frozen \
         AAM_module/datasets/rom_phase2_v2_frozen; do
  [ -d "$d" ] || continue
  if [ "$DRY" = "1" ]; then
    say "      would remove $d ($(du -sh "$d" | cut -f1))"
  else
    rm -rf "$d"
  fi
done

say "free after:  $(free_now)"
[ "$DRY" = "1" ] && say "(dry run: nothing was deleted)"
