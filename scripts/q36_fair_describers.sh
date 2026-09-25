#!/usr/bin/env bash
# Orchestrator request (06:40): the describer-vs-crafted comparison must be judged by a critic that saw the describer register.
# Reruns the IDENTICAL set list of bits_v1_describers (same seeded fixed set + same 4 sets -> the same 271 rows) on critic v2 (describer A + W in its pool)
# and on critic v1b (describer A in its pool), plus a second group adding the W variant (rows common to A, W, crafted).
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; source /home/celeste/nlt-q36-logs/gpu_lib.sh
TX=/vol/q36/text/v1; TX2=/vol/q36/text/v2; LOGD=/home/celeste/nlt-q36-logs; D=/home/celeste/shared/reports/nlt-27b-olens/data
log(){ echo "[fair] $(date -u +%H:%M) $*"; }
nfiles(){ timeout 120 modal volume ls nlt "$1" 2>/dev/null | grep -cE "$2" || true; }
run(){ out=$(NLT_Q36_GPU=H100 timeout 900 modal run --detach scripts/modal_nlt_q36.py --task hf --gpus 1 --script eval_bits.py --args "$1" 2>&1 | grep -E "SPAWNED|modal.com/apps|rror"); echo "$out" | sed "s/^/[$2] /" | tee -a $LOGD/apps.txt; ledger_add "$out" 1 "$2"; }
SAME="describer_sonnet_A:$TX/val/describer_sonnet5_A__*.parquet,describer_sonnet_B:$TX/val/describer_sonnet5_B__*.parquet,describer_qwen32b:$TX/val/describer_qwen3-32b__*.parquet,craft_full_same:$TX/val/craft_full__*.parquet"
WITHW="describer_sonnet_A:$TX/val/describer_sonnet5_A__*.parquet,describer_sonnet_W:$TX2/val/describer_sonnet5_W__*.parquet,craft_full_same:$TX/val/craft_full__*.parquet,raw_all_same:$TX/val/raw_all__*.parquet"
# judge checkpoints (orchestrator 06:15): the best-CALIBRATED checkpoint, not the lowest-FM-loss final. v1: ckpt_best = step 3500 (no 2500/3000 saved). v2: ckpt_step3000.pt if captured, else ckpt_best. v1b: ckpt_best.
( GT=H100 wait_gpu 1 || exit 1; run "--data-dir /vol/q36/data --ckpt /vol/q36/critic/v1/ckpt_best.pt --out /vol/q36/results/bits_v1best_describers.json --sets '$SAME' --n 300 --ode-steps 64 --skip-samples" bits_desc_v1best
  wait_bits bits_v1best_describers || exit 1; cp /tmp/q36_chk_bits_v1best_describers.json $D/bits_v1best_describers.json; log "v1best describer comparison pulled" ) &
for V in v2 v1b; do
  ( for i in $(seq 1 400); do [ "$(nfiles q36/critic/$V 'ckpt_final.pt')" -ge 1 ] && break; [ $((i % 10)) -eq 0 ] && log "waiting for critic $V"; sleep 180; done; log "critic $V final exists"
    CKP=/vol/q36/critic/$V/ckpt_best.pt; [ "$V" = v2 ] && [ "$(nfiles q36/critic/v2 'ckpt_step3000.pt')" -ge 1 ] && CKP=/vol/q36/critic/v2/ckpt_step3000.pt; log "judge for $V: $CKP"
    GT=H100 wait_gpu 1 || exit 1; run "--data-dir /vol/q36/data --ckpt $CKP --out /vol/q36/results/bits_${V}_describers.json --sets '$SAME' --n 300 --ode-steps 64 --skip-samples" bits_desc_$V
    GT=H100 wait_gpu 1 || exit 1; run "--data-dir /vol/q36/data --ckpt $CKP --out /vol/q36/results/bits_${V}_describersW.json --sets '$WITHW' --n 512 --ode-steps 64 --skip-samples" bits_descW_$V
    wait_bits bits_${V}_describers bits_${V}_describersW || exit 1
    for n in describers describersW; do cp /tmp/q36_chk_bits_${V}_$n.json $D/bits_${V}_$n.json; done; log "$V describer comparisons pulled" ) &
done
wait; cd /home/celeste/shared/reports/nlt-27b-olens && systemd-run --user --scope -q -p MemoryMax=1G python3 build_html.py >/dev/null 2>&1; log "FAIR DONE"
