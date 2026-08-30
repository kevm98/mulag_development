#!/usr/bin/env bash
# Liveness probe for a long GPU job.
#
# "docker ps says Up" is NOT liveness -- the six-mesh gate sat "Up" for 82
# minutes at 28.0/31.3 GiB with GPU at 0% and no log output, thrashing rather
# than computing. The signals that actually distinguish alive from dead:
#
#   log mtime      advancing        -> producing output
#   GPU util       non-zero         -> computing (0% can be legitimate during a
#                                     CPU-side terrain rebuild, so read it with
#                                     the log)
#   container mem  well under limit -> not about to thrash
#   report files   appearing        -> completing units of work
set -u
LOG=${LOG:?set LOG to the run log}
# Point LOG at the STREAMING log (the tee target), never the outer wrapper log:
# the wrapper only receives its `tail` at process exit, so watching it reports a
# healthy run as STALLED. That false alarm fired once already.
PAT=${PAT:-outputs/rif_delay/gate_m*.json}
cd /home/kevinmurera/isaaclab-cloud/projects/rrlab_mulag

now=$(date '+%H:%M:%S')
age_s=$(( $(date +%s) - $(stat -c %Y "$LOG" 2>/dev/null || echo 0) ))
size=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
util=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)
cid=$(docker ps -q | head -1)
mem=$(docker stats --no-stream --format '{{.MemUsage}} ({{.MemPerc}}) cpu={{.CPUPerc}}' "$cid" 2>/dev/null)
done_n=$(ls -1 $PAT 2>/dev/null | wc -l)

printf '%s  log_age=%ss size=%s  gpu=[%s]  reports=%s\n' "$now" "$age_s" "$size" "$util" "$done_n"
printf '            container=%s\n' "${mem:-<none running>}"

if [ -z "$cid" ]; then
  echo "            STATUS: no container running"
elif [ "$age_s" -gt 900 ]; then
  echo "            *** STALLED: no log output for ${age_s}s -- investigate"
else
  echo "            STATUS: alive"
fi
