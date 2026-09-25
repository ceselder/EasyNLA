#!/usr/bin/env bash
# checkpoint sensitivity of the judge: score the main sets with critic v1 ckpt_best (step 3500, highest spot content) next to ckpt_final; launched after bits v1 land (GPU budget)
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; MAXG=${MAXG:-8}; TX=/vol/q36/text/v1
log(){ echo "[v1best] $(date -u +%H:%M) $*"; }
nfiles(){ timeout 120 modal volume ls nlt "$1" 2>/dev/null | grep -cE "$2" || true; }
gpus_in_use(){ timeout 90 modal app list 2>/dev/null | grep "nlt-q36" | grep -vE "stopped" | awk -F'│' '{gsub(/ /,"",$5); s+=$5} END {print s+0}'; }
for i in $(seq 1 600); do n=$(nfiles q36/results "bits_v1_(main|components|verbalizer|describers)\.json"); [ "$n" -ge 4 ] && break; sleep 180; done
for i in $(seq 1 600); do g=$(gpus_in_use); [ $((g + 1)) -le $MAXG ] && break; sleep 120; done; log "launching ($g in use)"
NLT_Q36_GPU=H100 timeout 900 modal run --detach scripts/modal_nlt_q36.py --task hf --gpus 1 --script eval_bits.py --args "--data-dir /vol/q36/data --ckpt /vol/q36/critic/v1/ckpt_best.pt --out /vol/q36/results/bits_v1best_main.json --sets 'craft_full:$TX/val/craft_full__*.parquet,describer:$TX/val/describer_sonnet5_A__*.parquet,raw_all:$TX/val/raw_all__*.parquet,verbalizer:/vol/q36/dumps/verbalizer_v1.parquet' --twins 'craft_twins:$TX/val/twins__*.parquet' --neighbors /vol/q36/data/neigh --neighbor-n 128 --n 384 --ode-steps 64 --skip-samples" 2>&1 | grep -E "SPAWNED|modal.com/apps|rror" | sed "s/^/[bits_v1best] /" | tee -a /home/celeste/nlt-q36-logs/apps.txt
for i in $(seq 1 300); do [ "$(nfiles q36/results 'bits_v1best_main.json')" -ge 1 ] && break; sleep 180; done
timeout 300 modal volume get nlt q36/results/bits_v1best_main.json /home/celeste/shared/reports/nlt-27b-olens/data/bits_v1best_main.json --force >/dev/null 2>&1; log "V1BEST DONE"
