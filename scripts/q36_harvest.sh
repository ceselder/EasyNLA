#!/usr/bin/env bash
# HARVEST (orchestrator 07:52): multiply DISTINCT (pair, text) rows for the one-pass critics.
#  - per-position olens reads at ALL 16 stored layers (greedy + grammar, 72 tokens) -> every (i<j) pair of a position gets its h_i / h_j reads for free (120 pairs / position)
#  - v_delta reads for K = 4 extra random (i, j) pairs per position (pairs_<split>_x4.parquet)
#  - order: the 15 shards with no rollouts yet (train 15-28 in split index 12-25, val 3), then the 15 existing shards; one engine = one B200 vLLM process working through its shard list, resumable (existing spec files are skipped)
#  - budget: 6 engine slots; each slot launches when wait_gpu allows (cap 12 while RL v3 runs, 8 after); a slot whose engine died / was stopped is relaunched with the same shard list (work already written is skipped)
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; source /home/celeste/nlt-q36-logs/gpu_lib.sh
LOGD=/home/celeste/nlt-q36-logs; NENG=${NENG:-6}; MAXTOK=${MAXTOK:-72}
LAYERS="h_L12;h_L16;h_L20;h_L24;h_L28;h_L30;h_L32;h_L36;h_L40;h_L42;h_L44;h_L48;h_L52;h_L54;h_L56;h_L60"
log(){ echo "[harvest] $(date -u +%H:%M) $*"; }
# shard tokens: split index within splits.json (train idx 0..25 = shard03..28; val idx 0..3 = shard29..32). Existing rollouts: train 0-11, val 0-2.
NEW=(train:12 train:13 train:14 train:15 train:16 train:17 train:18 train:19 train:20 train:21 train:22 train:23 train:24 train:25 val:3)
OLD=(train:0 train:1 train:2 train:3 train:4 train:5 train:6 train:7 train:8 train:9 train:10 train:11 val:0 val:1 val:2)
ALL=("${NEW[@]}" "${OLD[@]}")
declare -A LIST; for k in "${!ALL[@]}"; do p=$((k % NENG)); LIST[$p]="${LIST[$p]:-}${ALL[$k]},"; done
for p in $(seq 0 $((NENG - 1))); do LIST[$p]=${LIST[$p]%,}; log "engine $p shards: ${LIST[$p]}"; done
args_for(){ echo "--data-dir /vol/q36/data --pairs-shards '$1' --layer-specs '$LAYERS' --delta-pairs /vol/q36/data/pairs_SPLIT_x4.parquet --adapter /vol_go/ckpt/ar_ivrl/final --prompt bullets --n-samples 0 --max-tokens $MAXTOK --grammar --out-dir /vol/q36/rollouts_layers --out-dir-delta /vol/q36/rollouts_delta"; }
declare -A APP
launch(){ p=$1; L=${LIST[$p]}
  # the delta pairs file depends on the split of each shard token; the loader picks by 'split' column, so pass BOTH files' union: use the train file for train tokens and val for val tokens -> two invocations would reload the model; instead give ONE combined parquet
  out=$(spawn_retry env NLT_Q36_GPU=B200 timeout 900 modal run --detach scripts/modal_nlt_q36.py --task vllm --gpus 1 --script rollout_vllm.py --args "$(args_for "$L" | sed 's#pairs_SPLIT_x4#pairs_all_x4#')"); echo "$out" | sed "s/^/[harvest_$p] /" | tee -a $LOGD/apps.txt; ledger_add "$out" 1 harvest_$p
  APP[$p]=$(echo "$out" | grep -oE "ap-[A-Za-z0-9]+" | head -1); echo "${APP[$p]}" > $LOGD/harvest_app_$p.txt; log "engine $p launched: ${APP[$p]}"; }
done_p(){ for tok in $(echo "${LIST[$1]}" | tr ',' ' '); do sp=${tok%%:*}; si=${tok##*:}; f=$(python3 -c "import json; print(json.load(open('/tmp/q36_splits.json'))['$sp'][$si].split('/')[-1].replace('.parquet',''))"); n=$(timeout 120 modal volume ls nlt q36/rollouts_layers/$sp/$f 2>/dev/null | grep -c "h_L"); m=$(timeout 120 modal volume ls nlt q36/rollouts_delta/$sp/$f 2>/dev/null | grep -c "v_delta.parquet"); [ "$n" -ge 16 ] && [ "$m" -ge 1 ] || return 1; done; return 0; }
for i in $(seq 1 400); do
  raw=$(app_list); [ -z "$raw" ] && { log "app list read failed (transient); skipping this round"; sleep 120; continue; }     # an empty list must never relaunch a live engine
  live=$(echo "$raw" | grep nlt-q36 | grep -vE "stopped|stopping" | awk -F'│' '{gsub(/ /,"",$2); print $2}')
  alldone=1
  for p in $(seq 0 $((NENG - 1))); do
    a=$(cat $LOGD/harvest_app_$p.txt 2>/dev/null || true)
    if [ -n "$a" ] && echo "$live" | grep -q "$a"; then alldone=0; continue; fi                       # running
    if done_p $p; then continue; fi                                                                    # finished
    alldone=0; [ -f $LOGD/harvest_pause_$p ] && { log "engine $p paused ($LOGD/harvest_pause_$p exists)"; continue; }   # orchestrator 07:57: one engine paused for the eval queue
    PRIO=0 wait_gpu 1 || exit 1; launch $p
  done
  [ $alldone -eq 1 ] && { log "HARVEST DONE"; touch $LOGD/.harvest_done; break; }
  sleep 600
done
