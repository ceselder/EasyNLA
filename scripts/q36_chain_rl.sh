#!/usr/bin/env bash
# RL stage for nlt-27b-olens: waits for the SFT verbalizer + critic v1 (chain_phase1.sh) and the mechanics smoke, then runs the co-training RL
# (rl_verbalizer.py) on RL_GPUS GPUs with the GPU-headroom guard. Idempotent.
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt
TAG=${TAG:-v1}; RL_TAG=${RL_TAG:-rl_v1}; RL_GPUS=${RL_GPUS:-4}; STEPS=${STEPS:-150}; BATCH=${BATCH:-16}; GROUP=${GROUP:-8}; MAXG=${MAXG:-8}; BAND=${BAND:?set BAND}; KL=${KL:-0.02}; LAM=${LAM:--1}
GPUS_HF=${GPUS_HF:-"H100"}; GPUS_VLLM=${GPUS_VLLM:-"H100"}; GPUS_BIG=${GPUS_BIG:-"H200"}   # Modal 1.5.4 takes ONE gpu type per function (no fallback lists): route around the B200 queue explicitly
log(){ echo "[chainrl] $(date -u +%H:%M) $*"; }
nfiles(){ timeout 120 modal volume ls nlt "$1" 2>/dev/null | grep -cE "$2" || true; }
waitn(){ for i in $(seq 1 600); do n=$(nfiles "$1" "$2"); [ "$n" -ge "$3" ] && { log "ready: $1 ($n >= $3)"; return 0; }; [ $((i % 5)) -eq 0 ] && log "waiting $1: $n/$3"; sleep 120; done; log "TIMEOUT waiting $1"; return 1; }
gpus_in_use(){ timeout 90 modal app list 2>/dev/null | grep "nlt-q36" | grep -vE "stopped" | awk -F'│' '{gsub(/ /,"",$5); s+=$5} END {print s+0}'; }
wait_gpu(){ need=${1:-1}; for i in $(seq 1 600); do g=$(gpus_in_use); [ $((g + need)) -le $MAXG ] && { log "GPU headroom: $g in use, launching $need (types ${GT:-$GPUS_HF})"; return 0; }; [ $((i % 5)) -eq 0 ] && log "waiting for GPU headroom ($g in use, need $need)"; sleep 120; done; return 1; }
run(){ NLT_Q36_GPU="${GT:-$GPUS_HF}" timeout 900 modal run --detach scripts/modal_nlt_q36.py --task "$1" --gpus "$2" ${6:+--nproc $6} --script "$3" --args "$4" 2>&1 | grep -E "SPAWNED|modal.com/apps|Error|rror:" | sed "s/^/[$5] /" | tee -a /home/celeste/nlt-q36-logs/apps.txt; }
TX=/vol/q36/text/$TAG
waitn q36/verbalizer/$TAG/final "adapter_model" 1 || exit 1
waitn q36/critic/$TAG "ckpt_final.pt" 1 || exit 1
# the mechanics smoke must have produced its eval json (otherwise the trainer is broken; do not burn 4 GPUs)
waitn q36/rl/_smoke "eval_0003.json" 1 || { log "smoke did not finish; check ap logs"; exit 1; }
if [ "$(nfiles q36/rl/$RL_TAG 'eval_0000.json')" -lt 1 ]; then
  wait_gpu $RL_GPUS || exit 1
  GT="$GPUS_BIG" run hf $RL_GPUS rl_verbalizer.py "--data-dir /vol/q36/data --policy /vol/q36/verbalizer/$TAG/final --critic /vol/q36/critic/$TAG/ckpt_final.pt --replay-text '$TX/train/craft_full__*.parquet,$TX/train/describer_sonnet5_A__*.parquet' --twins '$TX/val/twins__*.parquet' --out /vol/q36/rl/$RL_TAG --band $BAND --max-train-pos 60000 --steps $STEPS --batch $BATCH --group $GROUP --n-tok ${NTOK:-176} --lr 1e-5 --critic-lr 3e-5 --kl $KL --lam $LAM --eval-every 10 --save-every 25 --heldout 128 --gen-chunk 32 --bwd-chunk ${BWD:-2} --no-grad-ckpt --wandb-name $RL_TAG" rl $RL_GPUS
fi
waitn q36/rl/$RL_TAG "eval_00(1|2|3|4|5|6|7|8|9)0.json" 1 || exit 1; log "RL running (first eval landed)"
for i in $(seq 1 600); do n=$(nfiles q36/rl/$RL_TAG "eval_.*json"); log "RL evals: $n"; [ "$n" -ge $((STEPS / 10 + 1)) ] && break; sleep 300; done
log "RL CHAIN DONE"; notify-discord "nlt-27b-olens: RL co-training $RL_TAG finished ($STEPS steps)" || true
