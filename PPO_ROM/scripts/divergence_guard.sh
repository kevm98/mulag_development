#!/usr/bin/env bash
set -uo pipefail

LOG=${1:?usage: divergence_guard.sh <train.log> <pid> [warmup] [factor] [strikes]}
PID=${2:?}
WARMUP=${3:-150}
FACTOR=${4:-2.5}
STRIKES=${5:-3}
POLL=${POLL:-60}
START_BYTES=${6:-0}
VERDICT="$(dirname "$LOG")/divergence_verdict.txt"

echo "[GUARD] watching $LOG pid=$PID warmup=$WARMUP factor=${FACTOR}x strikes=$STRIKES start_bytes=$START_BYTES"

strikes=0
last_seen_iter=-1
while kill -0 "$PID" 2>/dev/null; do
  sleep "$POLL"
  read -r BASE LAST BASE_CONTACT LAST_CONTACT BASE_CUT LAST_CUT ITER < <(python3 - "$LOG" "$WARMUP" "$START_BYTES" <<'PYEOF'
import re, sys, statistics
path = sys.argv[1]
start_bytes = max(int(sys.argv[3]), 0)
with open(path, "rb") as handle:
    full_text = handle.read().decode(errors="ignore")
with open(path, "rb") as handle:
    handle.seek(start_bytes)
    attempt_text = handle.read().decode(errors="ignore")
warmup = int(sys.argv[2])
def parse(text):
    blocks = re.split(r"Learning iteration (\d+)/", text)
    vals = {}
    for i in range(1, len(blocks), 2):
        block = blocks[i + 1]
        row = {}
        for name, label in (
            ("lateral", "Metrics/lateral_error_m"),
            ("contact", "Metrics/ground_contact_fraction"),
            ("cut", "Metrics/cut_height_error_m"),
        ):
            match = re.search(re.escape(label) + r":\s+([\d.eE+-]+)", block)
            if match:
                row[name] = float(match.group(1))
        if row:
            vals[int(blocks[i])] = row
    return vals
all_vals = parse(full_text)
attempt_vals = parse(attempt_text)
if not all_vals or not attempt_vals:
    print("0 0 0 0 0 0 0"); raise SystemExit
warm = [v for k, v in all_vals.items() if k <= warmup]
def med(name):
    values = [v[name] for v in warm if name in v]
    return statistics.median(values) if values else 0.0
last_iter = max(attempt_vals)
last = attempt_vals[last_iter]
print(
    f"{med('lateral'):.6f} {last.get('lateral', 0.0):.6f} "
    f"{med('contact'):.6f} {last.get('contact', 0.0):.6f} "
    f"{med('cut'):.6f} {last.get('cut', 0.0):.6f} {last_iter}"
)
PYEOF
)
  if [ "$ITER" -le "$last_seen_iter" ]; then
    continue
  fi
  last_seen_iter=$ITER
  TRIGGERS=$(awk \
      -v b="$BASE" -v l="$LAST" \
      -v bc="$BASE_CONTACT" -v lc="$LAST_CONTACT" \
      -v bk="$BASE_CUT" -v lk="$LAST_CUT" \
      -v it="$ITER" -v w="$WARMUP" -v f="$FACTOR" \
      'BEGIN {
        lateral_bad = b > 0 && l > b * f
        contact_bad = bc > 0 && lc < bc - 0.15
        cut_bad = bk > 0 && lk > bk + 0.05
        if (it <= w) exit
        sep = ""
        if (lateral_bad) { printf "%slateral", sep; sep="," }
        if (contact_bad) { printf "%scontact", sep; sep="," }
        if (cut_bad) { printf "%scut", sep }
      }')
  if [ -n "$TRIGGERS" ]; then
    strikes=$((strikes + 1))
  else
    strikes=0
  fi
  echo "[GUARD] iter=$ITER lateral=$LAST contact=$LAST_CONTACT cut=$LAST_CUT " \
       "baseline=($BASE,$BASE_CONTACT,$BASE_CUT) triggers=${TRIGGERS:-none} " \
       "strikes=$strikes/$STRIKES"
  if [ "$strikes" -ge "$STRIKES" ]; then
    {
      echo "DIVERGED"
      echo "iteration=$ITER"
      echo "lateral=$LAST"
      echo "warmup_baseline=$BASE"
      echo "contact=$LAST_CONTACT"
      echo "warmup_contact_baseline=$BASE_CONTACT"
      echo "cut_error=$LAST_CUT"
      echo "warmup_cut_baseline=$BASE_CUT"
      echo "triggers=$TRIGGERS"
      echo "factor=$FACTOR"
      echo "killed=$(date -Is)"
    } > "$VERDICT"
    echo "[GUARD] DIVERGED at iter $ITER: triggers=$TRIGGERS -- killing $PID"
    kill -9 "$PID" 2>/dev/null
    exit 2
  fi
done
echo "[GUARD] training exited without diverging"
exit 0
