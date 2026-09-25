#!/usr/bin/env bash
# ABLATION critic v1b: same pools as v1 but NO unconditional pretraining and uncond-frac 0.10 (does the text-path exact-likelihood deficit come from the strong unconditional path?)
# starts after the v1 bits jobs have landed (GPU budget), 1 H100; then a small Heun-64 bits job on the same sets (n 256).
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; MAXG=${MAXG:-8}; BAND=12,16,20,24,28,30,32,36,40,42,44,48,52,54,56,60; TX=/vol/q36/text/v1
log(){ echo "[v1b] $(date -u +%H:%M) $*"; }
nfiles(){ timeout 120 modal volume ls nlt "$1" 2>/dev/null | grep -cE "$2" || true; }
source /home/celeste/nlt-q36-logs/gpu_lib.sh
wait_gpu(){ need=${1:-1}; for i in $(seq 1 600); do g=$(gpus_in_use); [ $((g + need)) -le $MAXG ] && return 0; sleep 120; done; return 1; }
run(){ out=$(NLT_Q36_GPU=H100 timeout 900 modal run --detach scripts/modal_nlt_q36.py --task hf --gpus 1 --script "$1" --args "$2" 2>&1 | grep -E "SPAWNED|modal.com/apps|rror"); echo "$out" | sed "s/^/[$3] /" | tee -a /home/celeste/nlt-q36-logs/apps.txt; ledger_add "$out" 1 "$3"; }
wait_bits bits_v1_main bits_v1_components bits_v1_verbalizer bits_v1_describers || exit 1; log "v1 bits landed; launching the v1b ablation"
POOLS="craft_full=0.30:$TX/train/craft_full__*.parquet,describer=0.20:$TX/train/describer_sonnet5_A__*.parquet,craft_nojl=0.08:$TX/train/craft_nojl__*.parquet,craft_delta=0.12:$TX/train/craft_delta__*.parquet,craft_newfaded=0.08:$TX/train/craft_newfaded__*.parquet,jlens=0.12:$TX/train/jlens__*.parquet,olens_j=0.10:$TX/train/olens_j__*.parquet,raw_all=0.10:$TX/train/raw_all__*.parquet"
VALS="craft_full:$TX/val/craft_full__*.parquet,describer:$TX/val/describer_sonnet5_A__*.parquet,raw_all:$TX/val/raw_all__*.parquet"
if [ "$(nfiles q36/critic/v1b 'ckpt_final.pt')" -lt 1 ]; then
  wait_gpu 1 || exit 1; run train_critic.py "--data-dir /vol/q36/data --out /vol/q36/critic/v1b --tag critic_v1b --pools '$POOLS' --val-sets '$VALS' --band $BAND --width 1536 --depth 16 --heads 16 --param v --uncond-steps 0 --uncond-frac 0.10 --steps 3000 --batch 1024 --micro-batch 128 --eval-every 500 --eval-n 256 --spot-exact-n 64 --spot-ode-steps 16 --max-hours 3.0" critic_v1b
  for i in $(seq 1 600); do [ "$(nfiles q36/critic/v1b 'ckpt_final.pt')" -ge 1 ] && break; sleep 300; done
fi
wait_gpu 1 || exit 1; run eval_bits.py "--data-dir /vol/q36/data --ckpt /vol/q36/critic/v1b/ckpt_final.pt --out /vol/q36/results/bits_v1b_main.json --sets 'craft_full:$TX/val/craft_full__*.parquet,describer_A:$TX/val/describer_sonnet5_A__*.parquet,raw_all:$TX/val/raw_all__*.parquet' --neighbors /vol/q36/data/neigh --neighbor-n 128 --n 256 --ode-steps 64 --skip-samples" bits_v1b
wait_bits bits_v1b_main || exit 1
timeout 300 modal volume get nlt q36/results/bits_v1b_main.json /home/celeste/shared/reports/nlt-27b-olens/data/bits_v1b_main.json --force >/dev/null 2>&1; log "V1B DONE"
