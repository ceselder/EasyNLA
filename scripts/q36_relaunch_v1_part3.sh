#!/usr/bin/env bash
# second half of relaunch_v1.sh (ledger-aware, robust to transient `modal app list` failures): wait for the 176-token re-dump -> bits_v1_verbalizer (+ truncated-teacher
# diagnostics) -> SFT v1b (full-length targets) -> dumps -> bits_v1b_verbalizer.
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; source /home/celeste/nlt-q36-logs/gpu_lib.sh
BAND=12,16,20,24,28,30,32,36,40,42,44,48,52,54,56,60; TAG=v1; TX=/vol/q36/text/$TAG; CK=/vol/q36/critic/$TAG/ckpt_final.pt; LOGD=/home/celeste/nlt-q36-logs
log(){ echo "[relaunch3] $(date -u +%H:%M) $*"; }
nfiles(){ timeout 120 modal volume ls nlt "$1" 2>/dev/null | grep -cE "$2" || true; }
waitn(){ for i in $(seq 1 400); do n=$(nfiles "$1" "$2"); [ "$n" -ge "$3" ] && { log "ready: $1 ($n >= $3)"; return 0; }; [ $((i % 5)) -eq 0 ] && log "waiting $1: $n/$3"; sleep 120; done; log "TIMEOUT waiting $1"; return 1; }
run(){ # task gpus script args label total_gpus [nproc]
  out=$(NLT_Q36_GPU="${GT:-H100}" timeout 900 modal run --detach scripts/modal_nlt_q36.py --task "$1" --gpus "$2" ${7:+--nproc $7} --script "$3" --args "$4" 2>&1 | grep -E "SPAWNED|modal.com/apps|Error|rror:")
  echo "$out" | sed "s/^/[$5] /" | tee -a $LOGD/apps.txt; ledger_add "$out" "$6" "$5"; }
# 2. SFT v1b: full-length targets (max-len 256), otherwise identical to v1; then dumps + bits (verbalizer v1b vs v1 vs teacher on the same rows)
GT=H200 wait_gpu 3 || exit 1
GT=H200 run hf 3 sft_verbalizer.py "--data-dir /vol/q36/data --text '$TX/train/craft_full__*.parquet' --val-text '$TX/val/craft_full__*.parquet' --out /vol/q36/verbalizer/v1b --band $BAND --steps 600 --batch 4 --grad-accum 8 --lr 3e-5 --max-len 256 --no-grad-ckpt --eval-every 100 --save-every 200 --wandb-name sft_v1b" sft_v1b 3 3
waitn q36/verbalizer/v1b "final" 1 || exit 1
GT=H100 wait_gpu 1 || exit 1
GT=H100 run hf 1 dump_verbalizer.py "--data-dir /vol/q36/data --adapter /vol/q36/verbalizer/v1b/final --pairs-text '$TX/val/craft_full__*.parquet' --n 1024 --band $BAND --max-new 256 --out /vol/q36/dumps/verbalizer_v1b.parquet" dump_v1b 1
waitn q36/dumps "verbalizer_v1b.parquet" 1 || exit 1
GT=H100 wait_gpu 1 || exit 1
GT=H100 run hf 1 eval_bits.py "--data-dir /vol/q36/data --ckpt $CK --out /vol/q36/results/bits_v1b_verbalizer.json --sets 'teacher:$TX/val/craft_full__*.parquet,verbalizer_v1b:/vol/q36/dumps/verbalizer_v1b.parquet,verbalizer:/vol/q36/dumps/verbalizer_${TAG}.parquet,base_control:/vol/q36/dumps/base_control_${TAG}.parquet' --n 512 --ode-steps 64" bits_verb_v1b 1
wait_bits bits_v1b_verbalizer || exit 1
timeout 300 modal volume get nlt q36/results/bits_v1b_verbalizer.json /home/celeste/shared/reports/nlt-27b-olens/data/bits_v1b_verbalizer.json --force >/dev/null 2>&1
cd /home/celeste/nlt && systemd-run --user --scope -q -p MemoryMax=2G python3 scripts/plot_nlt_q36_phase1.py --tag v1 2>&1 | tail -3; cd /home/celeste/shared/reports/nlt-27b-olens && systemd-run --user --scope -q -p MemoryMax=1G python3 build_html.py >/dev/null 2>&1; log "RELAUNCH2 DONE"
