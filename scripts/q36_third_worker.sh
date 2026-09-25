#!/usr/bin/env bash
# A THIRD eval worker while the prio-1 backlog (RL v4b exact-view evals = the RL stop decision; v4 / v3c gate evals) is deep: pause harvest engine 3 (resumable: written spec files are skipped),
# run eval_workers.sh with EVAL_WORKERS=3, and when no prio-1/2 spec is pending or running any more, go back to 2 workers (the extra one exits after 20 idle min) and unpause engine 3.
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; LOGD=/home/celeste/nlt-q36-logs; log(){ echo "[third] $(date -u +%H:%M) $*"; }
ENG=${ENG:-3}; A=$(cat $LOGD/harvest_app_$ENG.txt 2>/dev/null || true)
touch $LOGD/harvest_pause_$ENG; [ -n "$A" ] && timeout 120 modal app stop -y "$A" >/dev/null 2>&1 && log "paused harvest engine $ENG ($A stopped; resumable)"; sed -i -E "s|^($A 1 harvest_$ENG [0-9:]+)|# \1 (paused $(date -u +%H:%M) for a 3rd eval worker)|" $LOGD/gpu_ledger.txt
for p in $(pgrep -f "bash eval_workers.sh"); do kill $p 2>/dev/null; done; sleep 2
systemd-run --user --scope -q -p MemoryMax=1G --setenv=EVAL_WORKERS=3 bash $LOGD/eval_workers.sh >> $LOGD/eval_workers.out 2>&1 & log "eval_workers.sh restarted with EVAL_WORKERS=3"
sleep 600
for i in $(seq 1 200); do ls_=$(timeout 120 modal volume ls nlt q36/evalq 2>/dev/null); if [ -n "$ls_" ]; then hot=$(echo "$ls_" | grep -oE "q36/evalq/[12]_[0-9]+_[0-9]+_[A-Za-z0-9_.-]+\.json" | grep -vc "\.done" || true); [ "$hot" -eq 0 ] && break; [ $((i % 6)) -eq 0 ] && log "prio-1/2 specs pending or running: $hot"; fi; sleep 300; done
log "prio-1/2 backlog clear -> back to 2 workers, engine $ENG unpaused"
for p in $(pgrep -f "bash eval_workers.sh"); do kill $p 2>/dev/null; done; sleep 2
systemd-run --user --scope -q -p MemoryMax=1G --setenv=EVAL_WORKERS=2 bash $LOGD/eval_workers.sh >> $LOGD/eval_workers.out 2>&1 &
rm -f $LOGD/harvest_pause_$ENG; log "THIRD DONE"
