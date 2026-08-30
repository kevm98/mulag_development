#!/usr/bin/env bash
# Keep the ordered ROM -> ID-FD -> ID(AAM)-FD campaign as the only owner of
# AAM GPU work.  This guard intentionally leaves AAM descendants of the
# ordered training runner alone, while stopping legacy AAM-first schedulers,
# evaluators, and trainers that would contend for the same GPU.
set -u

ROOT=/home/rrlab/rrlab_mulag
LOG=${AAM_BLOCK_LOG:-/tmp/rrlab-aam-blocker.log}

args_for() {
  ps -p "$1" -o args= 2>/dev/null || true
}

ordered_ancestor() {
  local pid="$1" parent args seen=" "
  while [[ "$pid" =~ ^[0-9]+$ && "$pid" -gt 1 ]]; do
    [[ "$seen" == *" $pid "* ]] && return 1
    seen+="$pid "
    args=$(args_for "$pid")
    case "$args" in
      *"$ROOT/experiments/run_ordered_training_only.sh"*|*"$ROOT/experiments/run_ordered_training_forever.sh"*)
        return 0 ;;
    esac
    parent=$(ps -p "$pid" -o ppid= 2>/dev/null | tr -d ' ')
    [[ "$parent" =~ ^[0-9]+$ ]] || return 1
    pid="$parent"
  done
  return 1
}

stop_unordered() {
  local pid ppid args
  while read -r pid ppid args; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    [[ "$args" == *"$ROOT/experiments/block_unordered_aam_live.sh"* ]] && continue
    [[ "$args" == *"bash -c"* || "$args" == *"/bin/sh -c"* ]] && continue
    case "$args" in
      *"$ROOT/experiments/schedule_speed_curriculum_aam_first.sh"*|\
      *"$ROOT/experiments/schedule_speed_randomization.sh"*|\
      *"PPO_AAM_FD/scripts/train.py"*|\
      *"verge_task/scripts/train.py"*"--actuator id_aam_fd"*|\
      *"PPO_AAM_FD/scripts/score"*|\
      *"score_per_mesh.sh"*"id_aam"*)
        if ! ordered_ancestor "$pid"; then
          printf '[%s] stopping unordered AAM pid=%s ppid=%s args=%s\n' "$(date -Is)" "$pid" "$ppid" "$args" >>"$LOG"
          kill -TERM "$pid" 2>/dev/null || true
        fi
        ;;
    esac
  done < <(ps -eo pid=,ppid=,args=)
}

while :; do
  stop_unordered
  sleep 3
done
