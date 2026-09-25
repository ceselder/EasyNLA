#!/usr/bin/env bash
# polls the RL run: pulls eval_*.json + the trainer log into the report data dir every 10 min, plots the curves, rebuilds the HTML; stops when eval for the last step exists or the run dies
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; RL_TAG=${RL_TAG:-rl_v1}; STEPS=${STEPS:-150}; D=/home/celeste/shared/reports/nlt-27b-olens/data/rl_$RL_TAG; mkdir -p $D
log(){ echo "[pullrl] $(date -u +%H:%M) $*"; }
for i in $(seq 1 400); do n=$(timeout 120 modal volume ls nlt q36/rl/$RL_TAG 2>/dev/null | grep -c "eval_"); [ "$n" -ge 1 ] && break; [ $((i % 5)) -eq 0 ] && log "waiting for the first RL eval"; sleep 120; done
last=0
for i in $(seq 1 200); do
  files=$(timeout 120 modal volume ls nlt q36/rl/$RL_TAG 2>/dev/null | grep -E "eval_[0-9]+\.json" | sort); n=$(echo "$files" | grep -c eval_)
  if [ "$n" -gt "$last" ]; then
    for f in $files; do b=$(basename $f); [ -f $D/$b ] || timeout 120 modal volume get nlt "$f" $D/$b --force >/dev/null 2>&1; done; last=$n
    A=$(grep "^\[rl\] https" /home/celeste/nlt-q36-logs/apps.txt | tail -1 | grep -oE "ap-[A-Za-z0-9]+"); timeout 120 modal app logs $A 2>&1 | grep -E "^step [0-9]+ \| reward" > $D/train_log.txt
    systemd-run --user --scope -q -p MemoryMax=2G python3 scripts/plot_nlt_q36_rl.py --tag $RL_TAG --log $D/train_log.txt 2>&1 | tail -3
    (cd /home/celeste/shared/reports/nlt-27b-olens && systemd-run --user --scope -q -p MemoryMax=1G python3 build_html.py >/dev/null 2>&1); log "RL evals pulled: $n (plotted)"
  fi
  echo "$files" | grep -q "eval_$(printf '%04d' $STEPS).json" && break
  A=$(grep "^\[rl\] https" /home/celeste/nlt-q36-logs/apps.txt | tail -1 | grep -oE "ap-[A-Za-z0-9]+"); timeout 60 modal app logs $A 2>&1 | grep -qE "\[run\] exit [1-9]|UNCAUGHT" && { log "RL run died"; break; }
  sleep 600
done
log "PULLRL DONE"
