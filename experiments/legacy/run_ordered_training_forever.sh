#!/usr/bin/env bash
set -u
while true; do
  CAMPAIGN=speed_range_2_7_pathfix_v2 bash /home/rrlab/rrlab_mulag/experiments/run_ordered_training_only.sh >>/tmp/rrlab-ordered-training.log 2>&1
  rc=$?
  printf '[%s] ordered training exited rc=%s; resuming from checkpoint\n' "$(date -Is)" "$rc" >>/tmp/rrlab-ordered-training.log
  sleep 5
done
