#!/usr/bin/env bash
# Drive the rest of the ROM authority ladder unattended:
#   1. wait for the in-flight shards
#   2. fill in the levels that still lack a full 3-mesh measurement, plus two
#      extra levels inside the knee
#   3. summarise and plot
#
# Run from the VM host (not inside the container); it launches its own
# containers.
set -uo pipefail

ROOT=/home/kevinmurera/isaaclab-cloud/projects/rrlab_mulag
cd "$ROOT"
IMG=isaac-lab:2.3.2-rl052
OUTDIR=outputs/e5_rom_authority_ladder
TS=$(date +%Y%m%d_%H%M%S)
say() { echo "[$(TZ=Europe/Berlin date +%H:%M:%S)] $*"; }

say "waiting for in-flight ladder shards"
while pgrep -f "run_rom_authority_ladder.sh" >/dev/null 2>&1; do sleep 20; done
say "in-flight shards finished"
sudo chown -R kevinmurera:kevinmurera "$OUTDIR" 2>/dev/null

# Fill the plateau interior out to 3 meshes and add two levels inside the knee,
# so the curve has an error band everywhere rather than only at its ends.
launch() {
  local shard=$1 alphas=$2
  nohup env IMAGE=$IMG ~/bin/isaaclab-run bash -c "
    export ROOT=/workspace/projects/rrlab_mulag
    export PY=/isaac-sim/python.sh
    export ENVS=32 REPEATS=1 STALL=1800
    export MESHES='1337 17 909'
    export ALPHAS='$alphas'
    cd \$ROOT && bash experiments/E5_plant_robustness_envelope/code/run_rom_authority_ladder.sh
  " > "logs/e5_run/fill_${shard}_${TS}.log" 2>&1 &
  say "  shard $shard ($alphas) pid=$!"
}

say "launching fill-in shards"
launch A "0.90 0.60"; sleep 15
launch B "0.40 0.10"; sleep 15
launch C "0.03 0.005"; sleep 15

while pgrep -f "run_rom_authority_ladder.sh" >/dev/null 2>&1; do sleep 20; done
say "fill-in finished"
sudo chown -R kevinmurera:kevinmurera "$OUTDIR" 2>/dev/null

say "summarising"
env IMAGE=$IMG ~/bin/isaaclab-run bash -c "
  cd /workspace/projects/rrlab_mulag
  /isaac-sim/python.sh experiments/E5_plant_robustness_envelope/code/summarise_rom_ladder.py \
      --report-dir $OUTDIR --min-meshes 3 \
      --out experiments/E5_plant_robustness_envelope/results/rom_authority_ladder.json
  /isaac-sim/python.sh experiments/E5_plant_robustness_envelope/code/plot_rom_ladder.py \
      --summary experiments/E5_plant_robustness_envelope/results/rom_authority_ladder.json \
      --out experiments/E5_plant_robustness_envelope/figures/ch7_e5_rom_authority.pdf
" 2>&1 | tee "logs/e5_run/summary_${TS}.log"

sudo chown -R kevinmurera:kevinmurera experiments/E5_plant_robustness_envelope/results experiments/E5_plant_robustness_envelope/figures 2>/dev/null
say "snapshotting"
bash experiments/E5_plant_robustness_envelope/cloud/snapshot_deliverables.sh "E5 ROM authority robustness curve"
say "DONE"
