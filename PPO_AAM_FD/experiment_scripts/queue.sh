#!/usr/bin/env bash
# Run the remaining work in THESIS PRIORITY order.
#
# Fixes an ordering bug: periodic.sh calls finish.sh inline, so anything waiting
# on periodic.sh to EXIT would have sat behind the 9-hour seed run. The trace
# figure would have landed last instead of second.
#
# Priority, highest first:
#   1 PERIODIC tracking test  -- decides whether "adaptation" is earned at all.
#                                Already running; everything else waits on it.
#   2 TRACE figure            -- the artefact the literature asks for and the
#                                one a reader understands at a glance. Useless
#                                if 1 is negative, so it goes second not first.
#   3 HELD-OUT mass step      -- removes "selection mesh only" from the
#                                time-varying claim.
#   4 SEEDS                   -- removes n=1 against the project's 3-of-5 bar.
#                                Last because it is 9 h and validates whatever
#                                the first three settle on.
set -u
cd /home/kevinmurera
exec >>/home/kevinmurera/queue.log 2>&1
echo "=== queue armed $(date -u +%FT%TZ) ==="

# Wait on the MARKER, not the process, since periodic.sh stays alive while it
# runs finish.sh.
while ! grep -q "PERIODIC_COMPLETE" /home/kevinmurera/periodic.log 2>/dev/null; do
  ps -eo args | grep -q "[p]eriodic.sh" || { echo "periodic died before completing"; break; }
  sleep 45
done
echo "periodic done $(date -u +%FT%TZ)"

# periodic.sh launches finish.sh (held-out step then seeds) at its end. Stop it
# so the order below is the one that actually runs.
pkill -f "[f]inish.sh" && echo "finish.sh stopped, will be re-run in priority order"
sleep 5
for c in $(docker ps -q); do docker stop -t 20 "$c" >/dev/null 2>&1; done
sleep 8

echo "=== [2/4] trace figure ==="; date -u +%FT%TZ
./trace.sh
echo "=== [3/4] held-out mass step + [4/4] seeds ==="; date -u +%FT%TZ
./finish.sh
echo "=== queue complete $(date -u +%FT%TZ) ==="
