#!/usr/bin/env bash
# EVAL-QUEUE MANAGER (orchestrator 09:12): keep <= EVAL_WORKERS long-lived eval worker apps (Modal task evalq, 1 H100 each) alive while /vol/q36/evalq has pending specs.
# Workers drain the queue in priority order and exit after 20 idle minutes. Each worker is ONE app for many evals (instead of one app per eval).
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; source /home/celeste/nlt-q36-logs/gpu_lib.sh
LOGD=/home/celeste/nlt-q36-logs; EVAL_WORKERS=${EVAL_WORKERS:-3}; IDLE_MIN=${IDLE_MIN:-20}
log(){ echo "[evalq] $(date -u +%H:%M) $*"; }
wid=0
for i in $(seq 1 2000); do
  ls_=$(timeout 120 modal volume ls nlt q36/evalq 2>/dev/null | grep -oE "[0-9]_[0-9]+_[0-9]+_[A-Za-z0-9_.-]+\.json" | sort)
  pending=$(echo "$ls_" | grep -vE "\.running\.|\.done" | grep -c "\.json$" || true); running=$(echo "$ls_" | grep -c "\.running\." || true)
  raw=$(app_list); [ -z "$raw" ] && { sleep 120; continue; }     # transient empty list: do not spawn extra workers
  live=$(echo "$raw" | grep nlt-q36 | grep -vE "stopped|stopping" | awk -F'│' '{gsub(/ /,"",$2); print $2}' | while read a; do grep -qE "^$a 1 evalq_w" $LOGD/gpu_ledger.txt && echo $a; done | wc -l)
  want=$(( pending + running )); [ $want -gt $EVAL_WORKERS ] && want=$EVAL_WORKERS
  [ $((i % 5)) -eq 0 ] && log "queue: $pending pending, $running running, $live workers live (target $want)"
  while [ "$live" -lt "$want" ]; do
    top=$(echo "$ls_" | grep -vE "\.running\.|\.done" | head -n 1 | cut -c1-1); [ -z "$top" ] && top=9
    PRIO=$top wait_gpu 1 || exit 1
    wid=$((wid + 1)); out=$(spawn_retry env NLT_Q36_GPU=H100 timeout 900 modal run --detach scripts/modal_nlt_q36.py --task evalq --args "$IDLE_MIN" --module "w$wid"); echo "$out" | sed "s/^/[evalq_w$wid] /" | tee -a $LOGD/apps.txt; ledger_add "$out" 1 evalq_w$wid
    echo "$out" | grep -q SPAWNED && live=$((live + 1)) || break
  done
  sleep 120
done
