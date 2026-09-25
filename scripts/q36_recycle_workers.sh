#!/usr/bin/env bash
# RECYCLE the long-lived eval workers so they pick up the current code (add_local_dir snapshots the repo at app launch: workers w1/w2 from 09:15 predate the
# load_text_pairs harvest-pair-id parse and eval_bits --fixed-from-texts, so every eval touching text/v3 rows on them scores 0 rows - the v4 step-500 train probe came back n 0).
# Sequence: wait for the v4 step-500 held-out gate (running on w1) to finish -> stop every live worker app -> restore its claimed spec to pending -> comment ledger lines ->
# delete the empty train-probe result -> enqueue the train probes (steps 500, 1000) with the stage-shard glob + --fixed-from-texts. eval_workers.sh respawns fresh workers.
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; source /home/celeste/nlt-q36-logs/gpu_lib.sh
LOGD=/home/celeste/nlt-q36-logs; D=/home/celeste/shared/reports/nlt-27b-olens/data; log(){ echo "[recycle] $(date -u +%H:%M) $*"; }
WAIT_LABEL=${WAIT_LABEL:-v4eval_step000500}
for i in $(seq 1 90); do ls_=$(timeout 120 modal volume ls nlt q36/evalq 2>/dev/null); [ -n "$ls_" ] && ! echo "$ls_" | grep -q "_${WAIT_LABEL}\.running\." && break; [ $((i % 5)) -eq 0 ] && log "waiting for $WAIT_LABEL to finish"; sleep 60; done
log "$WAIT_LABEL no longer running -> recycling workers"
L=$(app_list); [ -z "$L" ] && { sleep 60; L=$(app_list); }
for a in $(grep -E "^ap-[A-Za-z0-9]+ 1 evalq_w" $LOGD/gpu_ledger.txt | awk '{print $1}'); do
  echo "$L" | grep -vE "stopped|stopping" | grep -q "$a" || continue
  timeout 120 modal app stop -y "$a" >/dev/null 2>&1 && log "stopped worker $a" || log "stop of $a failed"
  sed -i -E "s|^($a 1 evalq_w[^ ]* [0-9:]+)|# \1 (recycled $(date -u +%H:%M): stale code snapshot)|" $LOGD/gpu_ledger.txt
done
sleep 20
for r in $(timeout 120 modal volume ls nlt q36/evalq 2>/dev/null | grep -oE "[0-9]_[0-9]+_[0-9]+_[A-Za-z0-9_.-]+\.running\.w[0-9]+\.json"); do
  orig=$(echo "$r" | sed -E 's/\.running\.w[0-9]+\.json$/.json/'); timeout 120 modal volume get nlt "q36/evalq/$r" /tmp/q36_restore.json --force >/dev/null 2>&1 || { log "get $r failed"; continue; }
  timeout 120 modal volume put nlt /tmp/q36_restore.json "q36/evalq/$orig" >/dev/null 2>&1 && timeout 120 modal volume rm nlt "q36/evalq/$r" >/dev/null 2>&1 && log "restored $orig to pending" || log "restore of $r failed"
done
timeout 120 modal volume rm nlt q36/results/bits_v4_step000500_train.json >/dev/null 2>&1 && log "removed the empty (n 0) bits_v4_step000500_train.json"; rm -f $D/bits_v4_step000500_train.json
TRAIN_GLOB=$(python3 -c "import json,glob; sh=sorted({s for f in glob.glob('$D/critic_v4_stage*.json') for s in json.load(open(f))['shards']}); print(';'.join(f'/vol/q36/text/v3/train/craft_full__{s}.parquet' for s in sh))")
for st in step000500 step001000; do
  [ "$(timeout 120 modal volume ls nlt q36/critic/v4 2>/dev/null | grep -c ckpt_$st.pt)" -ge 1 ] || { log "no ckpt_$st.pt yet"; continue; }
  enqueue_eval "--data-dir /vol/q36/data --ckpt /vol/q36/critic/v4/ckpt_$st.pt --split train --out /vol/q36/results/bits_v4_${st}_train.json --sets 'craft_full:$TRAIN_GLOB' --n 128 --n-fixed 512 --ode-steps 64 --skip-samples --skip-sw --fixed-from-texts" v4eval_${st}_train 1 | tee -a $LOGD/watch_critic_v4.out
done
log "RECYCLE DONE (eval_workers.sh respawns fresh workers; queue: $(timeout 120 modal volume ls nlt q36/evalq 2>/dev/null | grep -cE '\.json$' ) spec files)"
