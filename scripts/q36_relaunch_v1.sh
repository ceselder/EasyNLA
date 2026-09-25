#!/usr/bin/env bash
# Relaunch after two bugs found at 05:00 UTC:
#   (1) eval_bits.py crashed at the end of every first set (torch Bool .mean()) -> all four bits_v1 jobs relaunched with the fix;
#   (2) rl_v1 collapsed to 1-token outputs in 7 steps: chain_rl.sh passed --lam 0.05 (stale default) instead of the auto-calibrated
#       lambda (-1 -> 20% of the group std at the median length), so the length penalty (0.05 x 96 = 4.8) dwarfed the FM reward std (~0.25).
#   Also: the crafted teacher text is ~200 tokens but SFT truncated targets at 160, dumps capped at 112 and RL at 96 -> the verbalizer never
#   emitted the Shift line (the most informative one). Fix: re-dump v1 with --max-new 176, RL v2 with --n-tok 176, SFT v1b with --max-len 256.
# GPU accounting: a ledger (app id -> GPUs) so multi-GPU apps are counted correctly; cap 8 minus the launches the older task-counting chains
# (chain2 critic v2 / bits2, v1b) have not made yet, so their undercount cannot push the total past 8.
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt
BAND=${BAND:-12,16,20,24,28,30,32,36,40,42,44,48,52,54,56,60}; TAG=v1; TX=/vol/q36/text/$TAG; CK=/vol/q36/critic/$TAG/ckpt_final.pt
LOGD=/home/celeste/nlt-q36-logs; LEDGER=$LOGD/gpu_ledger.txt; touch $LEDGER
log(){ echo "[relaunch] $(date -u +%H:%M) $*"; }
nfiles(){ timeout 120 modal volume ls nlt "$1" 2>/dev/null | grep -cE "$2" || true; }
waitn(){ for i in $(seq 1 400); do n=$(nfiles "$1" "$2"); [ "$n" -ge "$3" ] && { log "ready: $1 ($n >= $3)"; return 0; }; [ $((i % 5)) -eq 0 ] && log "waiting $1: $n/$3"; sleep 120; done; log "TIMEOUT waiting $1"; return 1; }
live_apps(){ timeout 90 modal app list 2>/dev/null | grep "nlt-q36" | grep -vE "stopped|stopping" | awk -F'│' '{gsub(/ /,"",$2); gsub(/ /,"",$5); print $2, $5}'; }
gpus_in_use(){ live_apps | while read -r id tasks; do g=$(awk -v a="$id" '$1==a {print $2; exit}' $LEDGER); echo "${g:-$tasks}"; done | awk '{s+=$1} END {print s+0}'; }
others_pending(){ p=0; grep -q "^\[critic2\].*modal.com/apps" $LOGD/apps.txt || p=$((p + 1)); grep -q "^\[critic_v1b\].*modal.com/apps" $LOGD/apps.txt || p=$((p + 1)); echo $p; }
wait_gpu(){ need=${1:-1}; for i in $(seq 1 900); do g=$(gpus_in_use); cap=$(( 8 - $(others_pending) )); [ $((g + need)) -le $cap ] && { log "GPU headroom: $g in use (cap $cap), launching $need x ${GT:-H100}"; return 0; }; [ $((i % 5)) -eq 0 ] && log "waiting for GPU headroom ($g in use, cap $cap, need $need)"; sleep 120; done; return 1; }
run(){ # task gpus script args label total_gpus [nproc]
  out=$(NLT_Q36_GPU="${GT:-H100}" timeout 900 modal run --detach scripts/modal_nlt_q36.py --task "$1" --gpus "$2" ${7:+--nproc $7} --script "$3" --args "$4" 2>&1 | grep -E "SPAWNED|modal.com/apps|Error|rror:")
  echo "$out" | sed "s/^/[$5] /" | tee -a $LOGD/apps.txt; id=$(echo "$out" | grep -oE "ap-[A-Za-z0-9]+" | head -1); [ -n "$id" ] && echo "$id $6 $5 $(date -u +%H:%M)" >> $LEDGER; echo "$id"; }

# 0. make sure the two broken apps are gone
for a in ap-XWBU8USGVTZPIGBmaYP54W ap-YSmHxGXZ4Chtz90pUtJWOO; do timeout 120 modal app stop -y $a >/dev/null 2>&1; done
for i in $(seq 1 20); do live_apps | grep -qE "ap-XWBU8USGVTZPIGBmaYP54W|ap-YSmHxGXZ4Chtz90pUtJWOO" || break; sleep 15; done; log "old apps stopped; GPUs in use now: $(gpus_in_use)"

# 1. RL v2: auto-calibrated lambda, 176-token budget, smaller backward chunk (memory), otherwise as rl_v1
GT=H200 wait_gpu 4 || exit 1
RLID=$(GT=H200 run hf 4 rl_verbalizer.py "--data-dir /vol/q36/data --policy /vol/q36/verbalizer/$TAG/final --critic $CK --replay-text '$TX/train/craft_full__*.parquet,$TX/train/describer_sonnet5_A__*.parquet' --twins '$TX/val/twins__*.parquet' --out /vol/q36/rl/rl_v2 --band $BAND --max-train-pos 60000 --steps 150 --batch 16 --group 8 --n-tok 176 --lr 1e-5 --critic-lr 3e-5 --kl 0.02 --lam -1 --eval-every 10 --save-every 25 --heldout 128 --gen-chunk 32 --bwd-chunk 2 --no-grad-ckpt --wandb-name rl_v2" rl_v2 4 4)
echo "$RLID" > $LOGD/rl_app.txt; log "RL v2 app $RLID"

# 2. bits v1 main + components (2 x H100, one app)
GT=H100 wait_gpu 2 || exit 1
BID=$(GT=H100 run hf-many 1 eval_bits.py "--data-dir /vol/q36/data --ckpt $CK --out /vol/q36/results/bits_${TAG}_main.json --sets 'craft_full:$TX/val/craft_full__*.parquet,describer:$TX/val/describer_sonnet5_A__*.parquet,jlens:$TX/val/jlens__*.parquet,raw_all:$TX/val/raw_all__*.parquet,skiplens_jd:$TX/val/skiplens_jd__*.parquet' --twins 'craft_twins:$TX/val/twins__*.parquet' --neighbors /vol/q36/data/neigh --neighbor-n 256 --n 512 --ode-steps 64 ;; --data-dir /vol/q36/data --ckpt $CK --out /vol/q36/results/bits_${TAG}_components.json --sets 'craft_nojl:$TX/val/craft_nojl__*.parquet,craft_delta:$TX/val/craft_delta__*.parquet,craft_newfaded:$TX/val/craft_newfaded__*.parquet,olens_j:$TX/val/olens_j__*.parquet,olens_i:$TX/val/olens_i__*.parquet' --n 512 --ode-steps 64 --skip-samples" bits 2)
echo "$BID" > $LOGD/bits_app.txt; log "bits main+components app $BID"

# 3. re-dump the v1 verbalizer with a 176-token budget (the SFT targets were cut at 160 + EOT), then bits verbalizer (+ truncated-teacher diagnostics) and bits describers
GT=H100 wait_gpu 1 || exit 1
run hf 1 dump_verbalizer.py "--data-dir /vol/q36/data --adapter /vol/q36/verbalizer/$TAG/final --pairs-text '$TX/val/craft_full__*.parquet' --n 1024 --band $BAND --max-new 176 --out /vol/q36/dumps/verbalizer_${TAG}.parquet" redump 1 >/dev/null
GT=H100 wait_gpu 1 || exit 1
run hf 1 eval_bits.py "--data-dir /vol/q36/data --ckpt $CK --out /vol/q36/results/bits_${TAG}_describers.json --sets 'describer_sonnet_A:$TX/val/describer_sonnet5_A__*.parquet,describer_sonnet_B:$TX/val/describer_sonnet5_B__*.parquet,describer_qwen32b:$TX/val/describer_qwen3-32b__*.parquet,craft_full_same:$TX/val/craft_full__*.parquet' --n 300 --ode-steps 64 --skip-samples" bits_desc 1 >/dev/null
# wait for the re-dump to overwrite the truncated dump (its mtime / a fresh log DONE line): key on the redump log on the volume
for i in $(seq 1 200); do timeout 120 modal volume get nlt q36/dumps/verbalizer_${TAG}.parquet /tmp/q36_vdump.parquet --force >/dev/null 2>&1; n=$(systemd-run --user --scope -q -p MemoryMax=2G python3 -c "import pandas as pd; d=pd.read_parquet('/tmp/q36_vdump.parquet'); print(int(d['text'].str.len().mean()))" 2>/dev/null); [ "${n:-0}" -gt 600 ] && { log "re-dump landed (mean $n chars; the capped dump had 497)"; break; }; [ $((i % 5)) -eq 0 ] && log "waiting for the re-dump (mean chars ${n:-?})"; sleep 90; done
GT=H100 wait_gpu 1 || exit 1
run hf 1 eval_bits.py "--data-dir /vol/q36/data --ckpt $CK --out /vol/q36/results/bits_${TAG}_verbalizer.json --sets 'teacher:$TX/val/craft_full__*.parquet,teacher_trunc176:$TX/val/teacher_trunc176__val.parquet,teacher_trunc96:$TX/val/teacher_trunc96__val.parquet,verbalizer:/vol/q36/dumps/verbalizer_${TAG}.parquet,base_control:/vol/q36/dumps/base_control_${TAG}.parquet' --n 512 --ode-steps 64" bits_verb 1 >/dev/null

# 4. SFT v1b: full-length targets (max-len 256), otherwise identical to v1; then dumps + bits v1b (verbalizer v1b vs v1 vs teacher on the same rows)
GT=H200 wait_gpu 3 || exit 1
run hf 3 sft_verbalizer.py "--data-dir /vol/q36/data --text '$TX/train/craft_full__*.parquet' --val-text '$TX/val/craft_full__*.parquet' --out /vol/q36/verbalizer/v1b --band $BAND --steps 600 --batch 4 --grad-accum 8 --lr 3e-5 --max-len 256 --no-grad-ckpt --eval-every 100 --save-every 200 --wandb-name sft_v1b" sft_v1b 3 3 >/dev/null
waitn q36/verbalizer/v1b "final" 1 || exit 1
GT=H100 wait_gpu 1 || exit 1
run hf 1 dump_verbalizer.py "--data-dir /vol/q36/data --adapter /vol/q36/verbalizer/v1b/final --pairs-text '$TX/val/craft_full__*.parquet' --n 1024 --band $BAND --max-new 256 --out /vol/q36/dumps/verbalizer_v1b.parquet" redump_v1b 1 >/dev/null
waitn q36/dumps "verbalizer_v1b.parquet" 1 || exit 1
GT=H100 wait_gpu 1 || exit 1
run hf 1 eval_bits.py "--data-dir /vol/q36/data --ckpt $CK --out /vol/q36/results/bits_v1b_verbalizer.json --sets 'teacher:$TX/val/craft_full__*.parquet,verbalizer_v1b:/vol/q36/dumps/verbalizer_v1b.parquet,verbalizer:/vol/q36/dumps/verbalizer_${TAG}.parquet,base_control:/vol/q36/dumps/base_control_${TAG}.parquet' --n 512 --ode-steps 64" bits_verb_v1b 1 >/dev/null
waitn q36/results "bits_v1b_verbalizer.json" 1 || exit 1
timeout 300 modal volume get nlt q36/results/bits_v1b_verbalizer.json /home/celeste/shared/reports/nlt-27b-olens/data/bits_v1b_verbalizer.json --force >/dev/null 2>&1
log "RELAUNCH DONE"
