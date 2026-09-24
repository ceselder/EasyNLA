#!/usr/bin/env bash
# Sequential benchmark queue: one B200 at a time per queue.  usage: queue.sh <queue-name> <engine:name> [<engine:name> ...]
# Each item runs `modal run` (blocking) with scripts/gemma_bench/cfgs/<name>.json; logs -> ~/nla-exp-logs/gemma_engine/<name>.log
set -u
Q=$1; shift
cd /home/celeste/easynla-qwen36 || exit 1
LOGD=~/nla-exp-logs/gemma_engine; mkdir -p "$LOGD"
for item in "$@"; do
  eng=${item%%:*}; name=${item#*:}
  echo "[$Q] $(date -u +%H:%M) start $name ($eng)" >> "$LOGD/queue_$Q.log"
  timeout 4h modal run scripts/gemma_bench/modal_bench.py --task run --engine "$eng" --name "$name" --cfg "@scripts/gemma_bench/cfgs/$name.json" > "$LOGD/$name.log" 2>&1 < /dev/null
  rc=$?
  echo "[$Q] $(date -u +%H:%M) done  $name rc=$rc $(grep -o '"prompts_per_s": [0-9.]*' "$LOGD/$name.log" | head -1)" >> "$LOGD/queue_$Q.log"
done
echo "[$Q] $(date -u +%H:%M) queue finished" >> "$LOGD/queue_$Q.log"
