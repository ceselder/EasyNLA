#!/usr/bin/env bash
# waits for the two local Sonnet describer drivers (val A, train A), uploads their outputs to the nlt volume (text/v1), then runs variant B (300 val pairs) and uploads it.
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; DL=/home/celeste/nlt-q36-data/describer; TAG=v1
log(){ echo "[descup] $(date -u +%H:%M) $*"; }
VPART=$(ls $DL/in/val__describer_inputs__*.parquet | head -1 | sed 's/.*describer_inputs__//; s/\.parquet//')
up(){ timeout 600 modal volume put nlt "$1" "$2" --force >/dev/null 2>&1 && log "uploaded $2" || log "UPLOAD FAILED $2"; }
for i in $(seq 1 200); do [ -f $DL/out/val/describer_sonnet5_A__val_stats.json ] && break; sleep 60; done; log "val A done: $(tr -d '\n ' < $DL/out/val/describer_sonnet5_A__val_stats.json | cut -c1-300)"
up $DL/out/val/describer_sonnet5_A__val.parquet q36/text/$TAG/val/describer_sonnet5_A__$VPART.parquet; up $DL/out/val/describer_sonnet5_A__val_stats.json q36/text/$TAG/val/describer_sonnet5_A__${VPART}_stats.json
# variant B (passage tail) on 300 val pairs, after the val driver has freed its slot
systemd-run --user --scope -q -p MemoryMax=2G with-local-keys python3 nlt/q36/describe.py sonnet --inputs $DL/in/val__describer_inputs__$VPART.parquet --out $DL/out/val/describer_sonnet5_B__val.parquet --variant B --limit 300 --concurrency 32 > $DL/desc_val_B.log 2>&1
log "val B done: $(tr -d '\n ' < $DL/out/val/describer_sonnet5_B__val_stats.json 2>/dev/null | cut -c1-300)"
up $DL/out/val/describer_sonnet5_B__val.parquet q36/text/$TAG/val/describer_sonnet5_B__$VPART.parquet
for i in $(seq 1 400); do [ -f $DL/out/train/describer_sonnet5_A__train_stats.json ] && break; sleep 60; done; log "train A done: $(tr -d '\n ' < $DL/out/train/describer_sonnet5_A__train_stats.json | cut -c1-300)"
up $DL/out/train/describer_sonnet5_A__train.parquet q36/text/$TAG/train/describer_sonnet5_A__train4.parquet; up $DL/out/train/describer_sonnet5_A__train_stats.json q36/text/$TAG/train/describer_sonnet5_A__train4_stats.json
# open-model side-by-side (300 val pairs) on Modal
NLT_Q36_GPU="B200,H200,H100" timeout 900 modal run --detach scripts/modal_nlt_q36_describe.py --inputs "/vol/q36/text/$TAG/val/describer_inputs__*.parquet" --out-dir /vol/q36/text/$TAG/val --variant A --limit 300 --containers 1 2>&1 | grep -E "SPAWNED|modal.com/apps|rror" | sed "s/^/[desc_qwen] /" | tee -a /home/celeste/nlt-q36-logs/apps.txt
log "DESCUP DONE"
