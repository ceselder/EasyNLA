#!/usr/bin/env bash
# Phase-1 chain for nlt-27b-olens: finalize -> olens rollouts (pair mode) -> craft -> critic + verbalizer SFT -> dump -> exact bits.
# Usage: BAND=24,28,...  bash chain_phase1.sh   (stages are skipped when their outputs already exist; polls are >= 2 min apart: Modal rate limits)
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt
BAND=${BAND:?set BAND}; N_TRAIN=${N_TRAIN:-12}; N_VAL=${N_VAL:-3}; TAG=${TAG:-v1}; NSAMP=${NSAMP:-1}; MAXTOK=${MAXTOK:-80}
CRITIC_STEPS=${CRITIC_STEPS:-4500}; UNCOND_STEPS=${UNCOND_STEPS:-1500}; SFT_STEPS=${SFT_STEPS:-600}; SKIP_TO=${SKIP_TO:-}
GPUS_HF=${GPUS_HF:-"H100"}; GPUS_VLLM=${GPUS_VLLM:-"H100"}; GPUS_BIG=${GPUS_BIG:-"H200"}   # Modal 1.5.4 takes ONE gpu type per function (no fallback lists): route around the B200 queue explicitly
log(){ echo "[chain1] $(date -u +%H:%M) $*"; }
nfiles(){ timeout 120 modal volume ls nlt "$1" 2>/dev/null | grep -cE "$2" || true; }
waitn(){ for i in $(seq 1 400); do n=$(nfiles "$1" "$2"); [ "$n" -ge "$3" ] && { log "ready: $1 ($n >= $3)"; return 0; }; [ $((i % 5)) -eq 0 ] && log "waiting $1: $n/$3"; sleep 120; done; log "TIMEOUT waiting $1"; return 1; }
gpus_in_use(){ timeout 90 modal app list 2>/dev/null | grep "nlt-q36" | grep -vE "stopped" | awk -F'│' '{gsub(/ /,"",$5); s+=$5} END {print s+0}'; }
wait_gpu(){ need=${1:-1}; for i in $(seq 1 600); do g=$(gpus_in_use); [ $((g + need)) -le ${MAXG:-8} ] && { log "GPU headroom: $g in use, launching $need (types ${GT:-$GPUS_HF})"; return 0; }; [ $((i % 5)) -eq 0 ] && log "waiting for GPU headroom ($g in use, need $need)"; sleep 120; done; return 1; }
run(){ NLT_Q36_GPU="${GT:-$GPUS_HF}" timeout 900 modal run --detach scripts/modal_nlt_q36.py --task "$1" --gpus "$2" ${6:+--nproc $6} --script "$3" --args "$4" 2>&1 | grep -E "SPAWNED|modal.com/apps|Error|rror:" | sed "s/^/[$5] /" | tee -a /home/celeste/nlt-q36-logs/apps.txt; }

# 1. store complete (30 harvest parts)
waitn q36/data/acts "part0000.parquet$" 30 || exit 1
# 2. finalize (CPU): splits, per-layer stats, pairs
if [ "$(nfiles q36/data 'pairs_val.parquet')" -lt 1 ]; then
  run cpu 1 finalize_q36.py "--data-dir /vol/q36/data --acts-glob '/vol/q36/data/acts/*.parquet' --val-files 4 --band $BAND --pairs-per-pos 1 --val-pairs-per-pos 1 --stats-pos 40000" finalize
  waitn q36/data "pairs_val.parquet" 1 || exit 1
fi
# 3. rollouts, pair mode, 6 engines. shard-jobs: val 0..N_VAL-1 first, then train 0..N_TRAIN-1, round-robin over 6 processes
MARK=/home/celeste/nlt-q36-logs/.rollouts_launched_$TAG      # the rollout dir holds SUBDIRS (one per shard), so a file count cannot detect a running launch: use a local marker
if [ ! -f "$MARK" ] && [ "$SKIP_TO" = "" ]; then
  touch "$MARK"
  JOBS=(); for v in $(seq 0 $((N_VAL - 1))); do JOBS+=("val:$v"); done; for t in $(seq 0 $((N_TRAIN - 1))); do JOBS+=("train:$t"); done
  ARGS=""; for p in 0 1 2 3 4 5; do L=""; for k in "${!JOBS[@]}"; do [ $((k % 6)) -eq $p ] && L+="${JOBS[$k]},"; done; L=${L%,}; [ -n "$L" ] && ARGS+="--data-dir /vol/q36/data --pairs-shards '$L' --specs 'v_i;v_j;v_delta' --adapter /vol_go/ckpt/ar_ivrl/final --prompt bullets --n-samples $NSAMP --max-tokens $MAXTOK --grammar --out-dir /vol/q36/rollouts ;; "; done
  wait_gpu $(( 1 * $(( $(grep -o ';;' <<< "$ARGS" | wc -l) + 1 )) )) || exit 1; GT="$GPUS_VLLM" run vllm-many 1 rollout_vllm.py "${ARGS% ;; }" rollouts
fi
# wait: every shard dir has 3 spec files. modal volume ls is not recursive -> count per split dir
for i in $(seq 1 400); do
  nt=0; for d in $(timeout 120 modal volume ls nlt q36/rollouts/train 2>/dev/null); do c=$(nfiles "$d" "parquet$"); nt=$((nt + c)); sleep 2; done
  nv=0; for d in $(timeout 120 modal volume ls nlt q36/rollouts/val 2>/dev/null); do c=$(nfiles "$d" "parquet$"); nv=$((nv + c)); sleep 2; done
  log "rollout files: train $nt/$((3 * N_TRAIN)) val $nv/$((3 * N_VAL))"; [ "$nt" -ge $((3 * N_TRAIN)) ] && [ "$nv" -ge $((3 * N_VAL)) ] && break; sleep 150
done
# 4. craft (GPU: J-lens + embedder), 4 processes; twins on val
if [ "$(nfiles q36/text/$TAG/val 'stats__')" -lt "$N_VAL" ]; then
  VS=$(seq -s, 0 $((N_VAL - 1))); T1=$(seq -s, 0 $((N_TRAIN / 3 - 1))); T2=$(seq -s, $((N_TRAIN / 3)) $((2 * N_TRAIN / 3 - 1))); T3=$(seq -s, $((2 * N_TRAIN / 3)) $((N_TRAIN - 1)))
  wait_gpu $(( 1 * 4 )) || exit 1; run hf-many 1 craft_text.py "--data-dir /vol/q36/data --split val --shards $VS --rollouts-root /vol/q36/rollouts/val --out-dir /vol/q36/text/$TAG/val --twins ;; --data-dir /vol/q36/data --split train --shards $T1 --rollouts-root /vol/q36/rollouts/train --out-dir /vol/q36/text/$TAG/train ;; --data-dir /vol/q36/data --split train --shards $T2 --rollouts-root /vol/q36/rollouts/train --out-dir /vol/q36/text/$TAG/train ;; --data-dir /vol/q36/data --split train --shards $T3 --rollouts-root /vol/q36/rollouts/train --out-dir /vol/q36/text/$TAG/train" craft
  waitn q36/text/$TAG/train "stats__" "$N_TRAIN" || exit 1; waitn q36/text/$TAG/val "stats__" "$N_VAL" || exit 1
fi
TX=/vol/q36/text/$TAG
# 4b. DESCRIBER pool (user request): Sonnet 5 writes atomic change claims from the readouts. Local box only (with-local-keys), <= 2 capped drivers, conc 48.
#     val: all pairs of the first 2 val parts (variant A) + 300 pairs variant B; train: the first N_DESC_TRAIN parts (variant A). Open-model side-by-side: 300 val pairs on Modal.
N_DESC_TRAIN=${N_DESC_TRAIN:-4}; DL=/home/celeste/nlt-q36-data/describer; mkdir -p $DL/in $DL/out
if [ "$(nfiles q36/text/$TAG/val 'describer_sonnet5_A__')" -lt 1 ]; then
  for f in $(timeout 120 modal volume ls nlt q36/text/$TAG/val 2>/dev/null | grep describer_inputs__); do timeout 300 modal volume get nlt "$f" $DL/in/val__$(basename $f) --force >/dev/null 2>&1; done
  TR=$(timeout 120 modal volume ls nlt q36/text/$TAG/train 2>/dev/null | grep describer_inputs__ | sort | head -n $N_DESC_TRAIN)
  for f in $TR; do timeout 300 modal volume get nlt "$f" $DL/in/train__$(basename $f) --force >/dev/null 2>&1; done
  log "describer inputs pulled: $(ls $DL/in | wc -l) files"
  VIN=$(ls $DL/in/val__*.parquet | head -n 1 | tr '\n' ' '); TIN=$(ls $DL/in/train__*.parquet | tr '\n' ' ')
  systemd-run --user --scope -q -p MemoryMax=2G with-local-keys python3 nlt/q36/describe.py sonnet --inputs $VIN --out $DL/out/val/describer_sonnet5_A__val.parquet --variant A --concurrency 48 > $DL/desc_val_A.log 2>&1 &
  P1=$!
  systemd-run --user --scope -q -p MemoryMax=2G with-local-keys python3 nlt/q36/describe.py sonnet --inputs $TIN --out $DL/out/train/describer_sonnet5_A__train.parquet --variant A --concurrency 48 > $DL/desc_train_A.log 2>&1 &
  P2=$!; wait $P1; log "describer val A done: $(tail -c 400 $DL/desc_val_A.log | tr '\n' ' ')"
  systemd-run --user --scope -q -p MemoryMax=2G with-local-keys python3 nlt/q36/describe.py sonnet --inputs $VIN --out $DL/out/val/describer_sonnet5_B__val.parquet --variant B --limit 300 --concurrency 32 > $DL/desc_val_B.log 2>&1
  wait $P2; log "describer train A done: $(tail -c 400 $DL/desc_train_A.log | tr '\n' ' ')"
  for f in $DL/out/val/*.parquet $DL/out/val/*_stats.json; do timeout 300 modal volume put nlt "$f" q36/text/$TAG/val/$(basename $f) --force >/dev/null 2>&1; done
  for f in $DL/out/train/*.parquet $DL/out/train/*_stats.json; do timeout 300 modal volume put nlt "$f" q36/text/$TAG/train/$(basename $f) --force >/dev/null 2>&1; done
  log "describer uploaded"
  # open-model side-by-side (300 val pairs, 1 GPU), scored next to Sonnet by the bits job
  timeout 900 modal run --detach scripts/modal_nlt_q36_describe.py --inputs "$TX/val/describer_inputs__*.parquet" --out-dir $TX/val --variant A --limit 300 --containers 1 2>&1 | grep -E "SPAWNED|modal.com/apps|rror" | tee -a /home/celeste/nlt-q36-logs/apps.txt
fi
# 5. critic (1 GPU) + verbalizer SFT (4 GPUs) in parallel
POOLS="craft_full=0.30:$TX/train/craft_full__*.parquet,describer=0.20:$TX/train/describer_sonnet5_A__*.parquet,craft_nojl=0.08:$TX/train/craft_nojl__*.parquet,craft_delta=0.12:$TX/train/craft_delta__*.parquet,craft_newfaded=0.08:$TX/train/craft_newfaded__*.parquet,jlens=0.12:$TX/train/jlens__*.parquet,olens_j=0.10:$TX/train/olens_j__*.parquet"
VALS="craft_full:$TX/val/craft_full__*.parquet,describer:$TX/val/describer_sonnet5_A__*.parquet,jlens:$TX/val/jlens__*.parquet,craft_delta:$TX/val/craft_delta__*.parquet,olens_j:$TX/val/olens_j__*.parquet"
if [ "$(nfiles q36/critic/$TAG 'ckpt_final.pt')" -lt 1 ]; then
  wait_gpu $(( 1 * 1 )) || exit 1; run hf 1 train_critic.py "--data-dir /vol/q36/data --out /vol/q36/critic/$TAG --tag critic_$TAG --pools '$POOLS' --val-sets '$VALS' --band $BAND --width 1536 --depth 16 --heads 16 --param v --uncond-steps $UNCOND_STEPS --steps $CRITIC_STEPS --batch 1024 --micro-batch 128 --eval-every 500 --eval-n 256 --spot-exact-n 64 --spot-ode-steps 16 --max-hours 3.5" critic
fi
if [ "$(nfiles q36/verbalizer/$TAG/final 'adapter_model')" -lt 1 ]; then
  wait_gpu $(( 3 * 1 )) || exit 1; GT="$GPUS_BIG" run hf 3 sft_verbalizer.py "--data-dir /vol/q36/data --text '$TX/train/craft_full__*.parquet' --val-text '$TX/val/craft_full__*.parquet' --out /vol/q36/verbalizer/$TAG --band $BAND --steps $SFT_STEPS --batch 8 --grad-accum 4 --lr 3e-5 --eval-every 100 --save-every 200 --wandb-name sft_$TAG" sft 3
fi
waitn q36/verbalizer/$TAG/final "adapter_model" 1 || exit 1
# 6. dump the verbalizer on held-out pairs (greedy) + the base-model control
if [ "$(nfiles q36/dumps "verbalizer_$TAG.parquet")" -lt 1 ]; then
  wait_gpu $(( 1 * 2 )) || exit 1; run hf-many 1 dump_verbalizer.py "--data-dir /vol/q36/data --adapter /vol/q36/verbalizer/$TAG/final --pairs-text '$TX/val/craft_full__*.parquet' --n 1024 --band $BAND --out /vol/q36/dumps/verbalizer_$TAG.parquet ;; --data-dir /vol/q36/data --adapter none --base-only --pairs-text '$TX/val/craft_full__*.parquet' --n 512 --band $BAND --source base_control --out /vol/q36/dumps/base_control_$TAG.parquet" dump
fi
waitn q36/critic/$TAG "ckpt_final.pt" 1 || exit 1; waitn q36/dumps "verbalizer_$TAG.parquet" 1 || exit 1
# 7. exact bits: 3 parallel jobs (headline sets + twins | component ablations | verbalizer vs teacher on the same rows)
CK=/vol/q36/critic/$TAG/ckpt_final.pt
wait_gpu $(( 1 * 4 )) || exit 1; run hf-many 1 eval_bits.py "--data-dir /vol/q36/data --ckpt $CK --out /vol/q36/results/bits_${TAG}_main.json --sets 'craft_full:$TX/val/craft_full__*.parquet,describer:$TX/val/describer_sonnet5_A__*.parquet,jlens:$TX/val/jlens__*.parquet,raw_all:$TX/val/raw_all__*.parquet,skiplens_jd:$TX/val/skiplens_jd__*.parquet' --twins 'craft_twins:$TX/val/twins__*.parquet' --neighbors /vol/q36/data/neigh --neighbor-n 256 --n 512 --ode-steps 64 ;; --data-dir /vol/q36/data --ckpt $CK --out /vol/q36/results/bits_${TAG}_components.json --sets 'craft_nojl:$TX/val/craft_nojl__*.parquet,craft_delta:$TX/val/craft_delta__*.parquet,craft_newfaded:$TX/val/craft_newfaded__*.parquet,olens_j:$TX/val/olens_j__*.parquet,olens_i:$TX/val/olens_i__*.parquet' --n 512 --ode-steps 64 --skip-samples ;; --data-dir /vol/q36/data --ckpt $CK --out /vol/q36/results/bits_${TAG}_verbalizer.json --sets 'teacher:$TX/val/craft_full__*.parquet,verbalizer:/vol/q36/dumps/verbalizer_$TAG.parquet,base_control:/vol/q36/dumps/base_control_$TAG.parquet' --n 512 --ode-steps 64 ;; --data-dir /vol/q36/data --ckpt $CK --out /vol/q36/results/bits_${TAG}_describers.json --sets 'describer_sonnet_A:$TX/val/describer_sonnet5_A__*.parquet,describer_sonnet_B:$TX/val/describer_sonnet5_B__*.parquet,describer_qwen32b:$TX/val/describer_qwen3-32b__*.parquet,craft_full_same:$TX/val/craft_full__*.parquet' --n 300 --ode-steps 64 --skip-samples" bits
waitn q36/results "bits_${TAG}_(main|components|verbalizer|describers).json" 4 || exit 1
log "PHASE1 CHAIN DONE"; notify-discord "nlt-27b-olens: phase-1 chain done (critic $TAG, verbalizer, bits)" || true
