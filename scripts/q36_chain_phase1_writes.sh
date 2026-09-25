#!/usr/bin/env bash
# Phase-1 v2 for nlt-27b-olens: the attention / MLP write variants. Runs AFTER the writes store (extract_writes.py) and the v1 rollouts exist.
#   writes rollouts (v_attn, v_mlp; same 15 shards) -> craft v2 (text/v2 = v1 pools + raw_all_w / writes_only / LOO + describer inputs with writes)
#   -> Sonnet describer W (val shard 0 + N_DESC_TRAIN train shards) -> critic v2 on the FULL mix -> bits v2 (text-source search incl. writes variants)
# Usage: BAND=... bash chain_phase1_writes.sh   (idempotent; polls >= 2 min apart)
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt
BAND=${BAND:?set BAND}; N_TRAIN=${N_TRAIN:-12}; N_VAL=${N_VAL:-3}; TAG=${TAG:-v2}; V1=${V1:-v1}; NSAMP=${NSAMP:-1}; MAXTOK=${MAXTOK:-80}; N_DESC_TRAIN=${N_DESC_TRAIN:-4}
CRITIC_STEPS=${CRITIC_STEPS:-4500}; UNCOND_STEPS=${UNCOND_STEPS:-1500}; ENGINES=${ENGINES:-4}
GPUS_HF=${GPUS_HF:-"H100"}; GPUS_VLLM=${GPUS_VLLM:-"B200"}; GPUS_BIG=${GPUS_BIG:-"H200"}   # Modal 1.5.4 takes ONE gpu type per function (no fallback lists): route around the B200 queue explicitly
log(){ echo "[chain2] $(date -u +%H:%M) $*"; }
nfiles(){ timeout 120 modal volume ls nlt "$1" 2>/dev/null | grep -cE "$2" || true; }
waitn(){ for i in $(seq 1 400); do n=$(nfiles "$1" "$2"); [ "$n" -ge "$3" ] && { log "ready: $1 ($n >= $3)"; return 0; }; [ $((i % 5)) -eq 0 ] && log "waiting $1: $n/$3"; sleep 120; done; log "TIMEOUT waiting $1"; return 1; }
gpus_in_use(){ timeout 90 modal app list 2>/dev/null | grep "nlt-q36" | grep -vE "stopped" | awk -F'│' '{gsub(/ /,"",$5); s+=$5} END {print s+0}'; }
wait_gpu(){ need=${1:-1}; for i in $(seq 1 600); do g=$(gpus_in_use); [ $((g + need)) -le ${MAXG:-8} ] && { log "GPU headroom: $g in use, launching $need (types ${GT:-$GPUS_HF})"; return 0; }; [ $((i % 5)) -eq 0 ] && log "waiting for GPU headroom ($g in use, need $need)"; sleep 120; done; return 1; }
run(){ NLT_Q36_GPU="${GT:-$GPUS_HF}" timeout 900 modal run --detach scripts/modal_nlt_q36.py --task "$1" --gpus "$2" ${6:+--nproc $6} --script "$3" --args "$4" 2>&1 | grep -E "SPAWNED|modal.com/apps|Error|rror:" | sed "s/^/[$5] /" | tee -a /home/celeste/nlt-q36-logs/apps.txt; }
count_rollouts(){ n=0; for d in $(timeout 120 modal volume ls nlt q36/rollouts/$1 2>/dev/null); do c=$(nfiles "$d" "$2"); n=$((n + c)); sleep 2; done; echo $n; }

# 1. writes store complete (30 parts) + v1 rollouts complete (the v_i/v_j/v_delta files of all 15 shards)
waitn q36/data/writes "part0000.parquet$" 30 || exit 1
for i in $(seq 1 400); do nt=$(count_rollouts train "v_(i|j|delta).parquet$"); nv=$(count_rollouts val "v_(i|j|delta).parquet$"); log "v1 rollouts: train $nt/$((3 * N_TRAIN)) val $nv/$((3 * N_VAL))"; [ "$nt" -ge $((3 * N_TRAIN)) ] && [ "$nv" -ge $((3 * N_VAL)) ] && break; sleep 150; done
# 2. writes rollouts: v_attn, v_mlp over the same shards, ENGINES engines
MARK=/home/celeste/nlt-q36-logs/.rollouts_writes_launched_$TAG
if [ ! -f "$MARK" ]; then
  touch "$MARK"; JOBS=(); for v in $(seq 0 $((N_VAL - 1))); do JOBS+=("val:$v"); done; for t in $(seq 0 $((N_TRAIN - 1))); do JOBS+=("train:$t"); done
  ARGS=""; for p in $(seq 0 $((ENGINES - 1))); do L=""; for k in "${!JOBS[@]}"; do [ $((k % ENGINES)) -eq $p ] && L+="${JOBS[$k]},"; done; L=${L%,}; [ -n "$L" ] && ARGS+="--data-dir /vol/q36/data --pairs-shards '$L' --specs 'v_attn;v_mlp' --adapter /vol_go/ckpt/ar_ivrl/final --prompt bullets --n-samples $NSAMP --max-tokens $MAXTOK --grammar --out-dir /vol/q36/rollouts ;; "; done
  wait_gpu $(( 1 * $(( $(grep -o ';;' <<< "$ARGS" | wc -l) + 1 )) )) || exit 1; GT="$GPUS_VLLM" run vllm-many 1 rollout_vllm.py "${ARGS% ;; }" rollouts_w
fi
for i in $(seq 1 400); do nt=$(count_rollouts train "v_(attn|mlp).parquet$"); nv=$(count_rollouts val "v_(attn|mlp).parquet$"); log "writes rollouts: train $nt/$((2 * N_TRAIN)) val $nv/$((2 * N_VAL))"; [ "$nt" -ge $((2 * N_TRAIN)) ] && [ "$nv" -ge $((2 * N_VAL)) ] && break; sleep 150; done
# 3. craft v2 (with writes), twins on val
if [ "$(nfiles q36/text/$TAG/val 'stats__')" -lt "$N_VAL" ]; then
  VS=$(seq -s, 0 $((N_VAL - 1))); T1=$(seq -s, 0 $((N_TRAIN / 3 - 1))); T2=$(seq -s, $((N_TRAIN / 3)) $((2 * N_TRAIN / 3 - 1))); T3=$(seq -s, $((2 * N_TRAIN / 3)) $((N_TRAIN - 1)))
  W="--writes-dir /vol/q36/data/writes"
  wait_gpu $(( 1 * 4 )) || exit 1; run hf-many 1 craft_text.py "--data-dir /vol/q36/data --split val --shards $VS --rollouts-root /vol/q36/rollouts/val --out-dir /vol/q36/text/$TAG/val --twins $W ;; --data-dir /vol/q36/data --split train --shards $T1 --rollouts-root /vol/q36/rollouts/train --out-dir /vol/q36/text/$TAG/train $W ;; --data-dir /vol/q36/data --split train --shards $T2 --rollouts-root /vol/q36/rollouts/train --out-dir /vol/q36/text/$TAG/train $W ;; --data-dir /vol/q36/data --split train --shards $T3 --rollouts-root /vol/q36/rollouts/train --out-dir /vol/q36/text/$TAG/train $W" craft_w
  waitn q36/text/$TAG/train "stats__" "$N_TRAIN" || exit 1; waitn q36/text/$TAG/val "stats__" "$N_VAL" || exit 1
fi
TX=/vol/q36/text/$TAG; TX1=/vol/q36/text/$V1
# 4. Sonnet describer WITH writes (variant W): val shard 0 + N_DESC_TRAIN train shards; local box, <= 2 drivers
DL=/home/celeste/nlt-q36-data/describer_w; mkdir -p $DL/in $DL/out/train $DL/out/val
# <= 2 local Sonnet drivers at a time: wait for the v1 train-A describer to finish before starting the W drivers
for i in $(seq 1 200); do [ -f /home/celeste/nlt-q36-data/describer/out/train/describer_sonnet5_A__train_stats.json ] && break; [ $((i % 10)) -eq 0 ] && log "waiting for the train-A describer to finish"; sleep 60; done
if [ "$(nfiles q36/text/$TAG/val 'describer_sonnet5_W__')" -lt 1 ]; then
  for f in $(timeout 120 modal volume ls nlt q36/text/$TAG/val 2>/dev/null | grep describer_inputs__ | sort | head -n 1); do timeout 300 modal volume get nlt "$f" $DL/in/val__$(basename $f) --force >/dev/null 2>&1; done
  for f in $(timeout 120 modal volume ls nlt q36/text/$TAG/train 2>/dev/null | grep describer_inputs__ | sort | head -n $N_DESC_TRAIN); do timeout 300 modal volume get nlt "$f" $DL/in/train__$(basename $f) --force >/dev/null 2>&1; done
  log "describer-W inputs pulled: $(ls $DL/in | wc -l) files"
  VIN=$(ls $DL/in/val__*.parquet | tr '\n' ' '); TIN=$(ls $DL/in/train__*.parquet | tr '\n' ' ')
  systemd-run --user --scope -q -p MemoryMax=2G with-local-keys python3 nlt/q36/describe.py sonnet --inputs $VIN --out $DL/out/val/describer_sonnet5_W__val.parquet --variant W --concurrency 48 > $DL/desc_val_W.log 2>&1 &
  P1=$!
  systemd-run --user --scope -q -p MemoryMax=2G with-local-keys python3 nlt/q36/describe.py sonnet --inputs $TIN --out $DL/out/train/describer_sonnet5_W__train.parquet --variant W --concurrency 48 > $DL/desc_train_W.log 2>&1 &
  P2=$!; wait $P1; log "describer val W done: $(tail -c 300 $DL/desc_val_W.log | tr '\n' ' ')"; wait $P2; log "describer train W done: $(tail -c 300 $DL/desc_train_W.log | tr '\n' ' ')"
  for f in $DL/out/val/*.parquet $DL/out/val/*_stats.json; do timeout 300 modal volume put nlt "$f" q36/text/$TAG/val/$(basename $f) --force >/dev/null 2>&1; done
  for f in $DL/out/train/*.parquet $DL/out/train/*_stats.json; do timeout 300 modal volume put nlt "$f" q36/text/$TAG/train/$(basename $f) --force >/dev/null 2>&1; done
  log "describer W uploaded"
fi
# 5. critic v2 on the FULL mix (v1 pools incl. the Sonnet A describer + v2 write pools + describer W)
POOLS="craft_full=0.18:$TX/train/craft_full__*.parquet,describer_A=0.14:$TX1/train/describer_sonnet5_A__*.parquet,describer_W=0.14:$TX/train/describer_sonnet5_W__*.parquet,raw_all=0.08:$TX/train/raw_all__*.parquet,raw_all_w=0.10:$TX/train/raw_all_w__*.parquet,writes_only=0.06:$TX/train/writes_only__*.parquet,raw_loo=0.08:$TX/train/raw_no_*__*.parquet;$TX/train/raw_w_no_*__*.parquet,craft_delta=0.06:$TX/train/craft_delta__*.parquet,craft_newfaded=0.04:$TX/train/craft_newfaded__*.parquet,jlens=0.06:$TX/train/jlens__*.parquet,olens_j=0.06:$TX/train/olens_j__*.parquet"
VALS="craft_full:$TX/val/craft_full__*.parquet,describer_A:$TX1/val/describer_sonnet5_A__*.parquet,describer_W:$TX/val/describer_sonnet5_W__*.parquet,raw_all_w:$TX/val/raw_all_w__*.parquet,raw_all:$TX/val/raw_all__*.parquet"
if [ "$(nfiles q36/critic/$TAG 'ckpt_final.pt')" -lt 1 ]; then
  wait_gpu $(( 1 * 1 )) || exit 1; run hf 1 train_critic.py "--data-dir /vol/q36/data --out /vol/q36/critic/$TAG --tag critic_$TAG --pools '$POOLS' --val-sets '$VALS' --band $BAND --width 1536 --depth 16 --heads 16 --param v --uncond-steps $UNCOND_STEPS --steps $CRITIC_STEPS --batch 1024 --micro-batch 128 --eval-every 500 --eval-n 256 --spot-exact-n 64 --spot-ode-steps 16 --max-hours 3.5" critic2
  waitn q36/critic/$TAG "ckpt_final.pt" 1 || exit 1
fi
# 6. bits v2: the text-source search on ONE judge, identical rows (n 512 where all sets exist; describer sets limited by their pair coverage)
CK=/vol/q36/critic/$TAG/ckpt_final.pt
wait_gpu $(( 1 * 3 )) || exit 1; run hf-many 1 eval_bits.py "--data-dir /vol/q36/data --ckpt $CK --out /vol/q36/results/bits_${TAG}_sources.json --sets 'raw_all:$TX/val/raw_all__*.parquet,raw_all_w:$TX/val/raw_all_w__*.parquet,craft_full:$TX/val/craft_full__*.parquet,describer_A:$TX1/val/describer_sonnet5_A__*.parquet,describer_W:$TX/val/describer_sonnet5_W__*.parquet' --twins 'craft_twins:$TX/val/twins__*.parquet' --neighbors /vol/q36/data/neigh --neighbor-n 256 --n 512 --ode-steps 64 ;; --data-dir /vol/q36/data --ckpt $CK --out /vol/q36/results/bits_${TAG}_loo.json --sets 'raw_all:$TX/val/raw_all__*.parquet,raw_no_i:$TX/val/raw_no_i__*.parquet,raw_no_j:$TX/val/raw_no_j__*.parquet,raw_no_delta:$TX/val/raw_no_delta__*.parquet,raw_no_jl:$TX/val/raw_no_jl__*.parquet,raw_all_w:$TX/val/raw_all_w__*.parquet,raw_w_no_attn:$TX/val/raw_w_no_attn__*.parquet,raw_w_no_mlp:$TX/val/raw_w_no_mlp__*.parquet,writes_only:$TX/val/writes_only__*.parquet' --n 512 --ode-steps 64 --skip-samples ;; --data-dir /vol/q36/data --ckpt $CK --out /vol/q36/results/bits_${TAG}_singles.json --sets 'olens_i:$TX/val/olens_i__*.parquet,olens_j:$TX/val/olens_j__*.parquet,craft_delta:$TX/val/craft_delta__*.parquet,jlens:$TX/val/jlens__*.parquet,craft_newfaded:$TX/val/craft_newfaded__*.parquet,craft_nojl:$TX/val/craft_nojl__*.parquet' --n 512 --ode-steps 64 --skip-samples" bits2
waitn q36/results "bits_${TAG}_(sources|loo|singles).json" 3 || exit 1
log "PHASE1-WRITES CHAIN DONE"; notify-discord "nlt-27b-olens: writes variants + critic v2 + text-source search done" || true
