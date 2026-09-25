#!/usr/bin/env bash
# HARVEST v5 = POSITIONS SCALING (orchestrator 2026-09-25 11:05 rule (3); plan + cost in notes/LOG.md 11:24): 30 NEW shards of harvest_v5 contexts (shard33..62 part0000,
# 120,960 new positions; shard00 = the olens test file is never touched) -> (A) 16-layer activation store /vol/q36/data/acts_v5 (extract_layers, 2 apps x 15 files, ~10 min),
# (B) merged store dir /vol/q36/data_v5 (finalize_v5.py, CPU: old train list + new shards appended so every existing pos_idx keeps its value; layer_stats COPIED; pairs +
# x4 delta pairs for the new positions), (C) olens 16-layer reads + K=4 delta reads per new position (harvest_v5.sh = harvest.sh with DATA_DIR=data_v5, tokens train:26..55),
# (D) crafting into /vol/q36/text/v5 (craft_v5.sh). GATES: the current harvest must be finished (.harvest_done) AND the go-file $LOGD/.go_harvest_v5 must exist
# (orchestrator veto window: I touch it when the harvest finishes unless told otherwise). Budget: 6 engines <= 12 GPUs with the eval workers; ~37 GPU-h total.
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; source /home/celeste/nlt-q36-logs/gpu_lib.sh
LOGD=/home/celeste/nlt-q36-logs; log(){ echo "[hv5] $(date -u +%H:%M) $*"; }
SRC=/vol_ol1/data/harvest_v5; ACTS=/vol/q36/data/acts_v5; DATA=/vol/q36/data_v5; NEW_SHARDS=${NEW_SHARDS:-$(seq -s ' ' 33 62)}
run(){ out=$(spawn_retry env NLT_Q36_GPU=${GPU:-H100} timeout 900 modal run --detach scripts/modal_nlt_q36.py --task "$1" --gpus 1 --script "$2" --args "$3"); echo "$out" | sed "s/^/[$4] /" | tee -a $LOGD/apps.txt; ledger_add "$out" "${5:-1}" "$4"; }
for i in $(seq 1 2000); do [ -f $LOGD/.harvest_done ] && [ -f $LOGD/.go_harvest_v5 ] && break; [ $((i % 20)) -eq 0 ] && log "waiting: harvest_done=$([ -f $LOGD/.harvest_done ] && echo yes || echo no) go=$([ -f $LOGD/.go_harvest_v5 ] && echo yes || echo no)"; sleep 60; done
[ -f $LOGD/.harvest_done ] && [ -f $LOGD/.go_harvest_v5 ] || { log "gates never opened; exit"; exit 1; }
# (A) extraction: two apps, 15 files each (skip files already present)
have=$(timeout 120 modal volume ls nlt ${ACTS#/vol/} 2>/dev/null | grep -oE "shard[0-9]+_part0000\.parquet" | sort -u)
todo=$(for s in $NEW_SHARDS; do f=$(printf "shard%02d_part0000.parquet" $s); echo "$have" | grep -q "^$f$" || echo "$f"; done)
if [ -n "$todo" ]; then
  n=$(echo "$todo" | wc -l); half=$(( (n + 1) / 2 )); G1=$(echo "$todo" | head -n $half | sed "s#^#$SRC/#" | paste -sd,); G2=$(echo "$todo" | tail -n +$((half + 1)) | sed "s#^#$SRC/#" | paste -sd,)
  for k in 1 2; do G=$([ $k -eq 1 ] && echo "$G1" || echo "$G2"); [ -z "$G" ] && continue; PRIO=1 wait_gpu 1 || exit 1; run hf extract_layers.py "--data '$G' --out-dir $ACTS --gaps ''" hv5_extract_$k; done
  log "extraction launched for $n files; waiting for all $(echo "$NEW_SHARDS" | wc -w) parquets"
  for i in $(seq 1 120); do c=$(timeout 120 modal volume ls nlt ${ACTS#/vol/} 2>/dev/null | grep -cE "shard[0-9]+_part0000\.parquet$"); [ "$c" -ge $(echo "$NEW_SHARDS" | wc -w) ] && break; [ $((i % 5)) -eq 0 ] && log "extraction: $c parquets so far"; sleep 60; done
fi
log "acts_v5 complete: $(timeout 120 modal volume ls nlt ${ACTS#/vol/} 2>/dev/null | grep -cE 'parquet$') files"
# (B) merged store dir (CPU)
if [ "$(timeout 120 modal volume ls nlt ${DATA#/vol/} 2>/dev/null | grep -c pairs_all_x4.parquet)" -lt 1 ]; then
  out=$(spawn_retry timeout 900 modal run --detach scripts/modal_nlt_q36.py --task cpu --script finalize_v5.py --args "--old-dir /vol/q36/data --new-acts '$ACTS/*.parquet' --out-dir $DATA"); echo "$out" | sed "s/^/[hv5_finalize] /" | tee -a $LOGD/apps.txt
  for i in $(seq 1 60); do [ "$(timeout 120 modal volume ls nlt ${DATA#/vol/} 2>/dev/null | grep -c pairs_all_x4.parquet)" -ge 1 ] && break; sleep 60; done
fi
timeout 60 modal volume get nlt ${DATA#/vol/}/splits.json /tmp/q36_splits_v5.json --force >/dev/null 2>&1 && log "data_v5: $(python3 -c "import json; s=json.load(open('/tmp/q36_splits_v5.json')); print(len(s['train']), 'train /', len(s['val']), 'val shards')")"
# (C) + (D): the olens read harvest over the new shards and the crafting into text/v5 (both resumable, both ledger-aware)
systemd-run --user --scope -q -p MemoryMax=1G --setenv=DATA_DIR=$DATA --setenv=PFX=harvest5 bash $LOGD/harvest_v5.sh >> $LOGD/harvest_v5.out 2>&1 &
sleep 5; systemd-run --user --scope -q -p MemoryMax=1G --setenv=DATA_DIR=$DATA --setenv=PFX=craft5 --setenv=TEXT_OUT=/vol/q36/text/v5 bash $LOGD/craft_v5.sh >> $LOGD/craft_v5.out 2>&1 &
log "harvest_v5.sh + craft_v5.sh launched (logs harvest_v5.out / craft_v5.out); HV5 CHAIN DONE"
