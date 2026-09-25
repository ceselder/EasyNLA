#!/usr/bin/env bash
# Craft the enlarged text pools (text/v3) from the harvest as shards complete: for each shard with 16 h_L files + v_delta, one H100 craft_text.py job in HARVEST MODE
# (K = 4 delta pairs + M = 12 extra pairs per position -> 16 pairs per position, 64.5k pairs per shard; pools craft_full / craft_nodelta / raw_all / raw_nodelta / jlens / olens_j / ...; twins on val idx 3 only).
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; source /home/celeste/nlt-q36-logs/gpu_lib.sh
LOGD=/home/celeste/nlt-q36-logs; M_EXTRA=${M_EXTRA:-12}; DATA_DIR=${DATA_DIR:-/vol/q36/data_v5}; LAYER_OUT=${LAYER_OUT:-/vol/q36/rollouts_layers}; DELTA_OUT=${DELTA_OUT:-/vol/q36/rollouts_delta}; TEXT_OUT=${TEXT_OUT:-/vol/q36/text/v5}; PFX=${PFX:-craft5}; SPLITS_TMP=/tmp/q36_splits_${PFX}.json
timeout 60 modal volume get nlt ${DATA_DIR#/vol/}/splits.json $SPLITS_TMP --force >/dev/null 2>&1 || { echo "cannot fetch $DATA_DIR/splits.json"; exit 1; }
log(){ echo "[$PFX] $(date -u +%H:%M) $*"; }
run(){ out=$(spawn_retry env NLT_Q36_GPU=H100 timeout 900 modal run --detach scripts/modal_nlt_q36.py --task hf --gpus 1 --script craft_text.py --args "$1"); echo "$out" | sed "s/^/[$2] /" | tee -a $LOGD/apps.txt; ledger_add "$out" 1 "$2"; }
shard_name(){ python3 -c "import json; print(json.load(open('$SPLITS_TMP'))['$1'][$2].split('/')[-1].replace('.parquet',''))"; }
ALL=(${TOKENS:-train:26 train:27 train:28 train:29 train:30 train:31 train:32 train:33 train:34 train:35 train:36 train:37 train:38 train:39 train:40 train:41 train:42 train:43 train:44 train:45 train:46 train:47 train:48 train:49 train:50 train:51 train:52 train:53 train:54 train:55})
declare -A DONE LAUNCHED
for i in $(seq 1 400); do
  todo=0
  for tok in "${ALL[@]}"; do
    sp=${tok%%:*}; si=${tok##*:}; [ -n "${DONE[$tok]:-}" ] && continue; f=$(shard_name $sp $si)
    if [ "$(timeout 120 modal volume ls nlt ${TEXT_OUT#/vol/}/$sp 2>/dev/null | grep -c "stats__$f.json")" -ge 1 ]; then DONE[$tok]=1; log "crafted $tok ($f)"; continue; fi
    todo=1; [ -n "${LAUNCHED[$tok]:-}" ] && continue
    # a craft launched by an earlier incarnation of this chain (restart) that is still live -> do not launch a second one (12:47: shard13 was crafted twice for 1 min)
    prev=$(grep -E "^\[${PFX}_${sp}_${si}\] https" $LOGD/apps.txt | tail -n 1 | grep -oE "ap-[A-Za-z0-9]+"); if [ -n "$prev" ] && app_live "$prev"; then LAUNCHED[$tok]=1; log "craft of $tok already running ($prev) from an earlier chain"; continue; fi
    n=$(timeout 120 modal volume ls nlt ${LAYER_OUT#/vol/}/$sp/$f 2>/dev/null | grep -c "h_L.*\.parquet$"); m=$(timeout 120 modal volume ls nlt ${DELTA_OUT#/vol/}/$sp/$f 2>/dev/null | grep -c "v_delta.parquet")
    [ "$n" -ge 16 ] && [ "$m" -ge 1 ] || continue
    PRIO=${CRAFT_PRIO:-3} wait_gpu 1 || exit 1                                                   # crafting feeds critic v4 (the main next judge): same priority as its gate evals
    TW=""
    run "--data-dir $DATA_DIR --split $sp --shards $si --rollouts-root /vol/q36/rollouts/$sp --layer-root $LAYER_OUT --delta-root $DELTA_OUT --extra-pairs $DATA_DIR/pairs_all_x4.parquet --m-extra $M_EXTRA --out-dir /vol/${TEXT_OUT#/vol/}/$sp --greedy-only $TW" craft3_${sp}_$si
    if grep -q "${PFX}_${sp}_${si}\] SPAWNED" $LOGD/apps.txt; then LAUNCHED[$tok]=1; log "craft launched for $tok ($(shard_name $sp $si): $n layer files, delta ok)"; else log "craft launch for $tok FAILED (will retry next round)"; fi
  done
  [ $todo -eq 0 ] && { log "CRAFT3 DONE (all 30 shards)"; break; }
  sleep 600
done
