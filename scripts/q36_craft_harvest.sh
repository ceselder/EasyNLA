#!/usr/bin/env bash
# Craft the enlarged text pools (text/v3) from the harvest as shards complete: for each shard with 16 h_L files + v_delta, one H100 craft_text.py job in HARVEST MODE
# (K = 4 delta pairs + M = 12 extra pairs per position -> 16 pairs per position, 64.5k pairs per shard; pools craft_full / craft_nodelta / raw_all / raw_nodelta / jlens / olens_j / ...; twins on val idx 3 only).
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; source /home/celeste/nlt-q36-logs/gpu_lib.sh
LOGD=/home/celeste/nlt-q36-logs; M_EXTRA=${M_EXTRA:-12}
log(){ echo "[craft3] $(date -u +%H:%M) $*"; }
run(){ out=$(spawn_retry env NLT_Q36_GPU=H100 timeout 900 modal run --detach scripts/modal_nlt_q36.py --task hf --gpus 1 --script craft_text.py --args "$1"); echo "$out" | sed "s/^/[$2] /" | tee -a $LOGD/apps.txt; ledger_add "$out" 1 "$2"; }
shard_name(){ python3 -c "import json; print(json.load(open('/tmp/q36_splits.json'))['$1'][$2].split('/')[-1].replace('.parquet',''))"; }
ALL=(train:12 train:13 train:14 train:15 train:16 train:17 train:18 train:19 train:20 train:21 train:22 train:23 train:24 train:25 val:3 train:0 train:1 train:2 train:3 train:4 train:5 train:6 train:7 train:8 train:9 train:10 train:11 val:0 val:1 val:2)
declare -A DONE LAUNCHED
for i in $(seq 1 400); do
  todo=0
  for tok in "${ALL[@]}"; do
    sp=${tok%%:*}; si=${tok##*:}; [ -n "${DONE[$tok]:-}" ] && continue; f=$(shard_name $sp $si)
    if [ "$(timeout 120 modal volume ls nlt q36/text/v3/$sp 2>/dev/null | grep -c "stats__$f.json")" -ge 1 ]; then DONE[$tok]=1; log "crafted $tok ($f)"; continue; fi
    todo=1; [ -n "${LAUNCHED[$tok]:-}" ] && continue
    n=$(timeout 120 modal volume ls nlt q36/rollouts_layers/$sp/$f 2>/dev/null | grep -c "h_L.*\.parquet$"); m=$(timeout 120 modal volume ls nlt q36/rollouts_delta/$sp/$f 2>/dev/null | grep -c "v_delta.parquet")
    [ "$n" -ge 16 ] && [ "$m" -ge 1 ] || continue
    PRIO=1 wait_gpu 1 || exit 1                                                   # crafting feeds critic v4 (the main next judge): same priority as its gate evals
    TW=""; [ "$tok" = "val:3" ] && TW="--twins"
    run "--data-dir /vol/q36/data --split $sp --shards $si --rollouts-root /vol/q36/rollouts/$sp --layer-root /vol/q36/rollouts_layers --delta-root /vol/q36/rollouts_delta --extra-pairs /vol/q36/data/pairs_all_x4.parquet --m-extra $M_EXTRA --out-dir /vol/q36/text/v3/$sp --greedy-only $TW" craft3_${sp}_$si
    if grep -q "craft3_${sp}_$si\] SPAWNED" $LOGD/apps.txt; then LAUNCHED[$tok]=1; log "craft launched for $tok ($(shard_name $sp $si): $n layer files, delta ok)"; else log "craft launch for $tok FAILED (will retry next round)"; fi
  done
  [ $todo -eq 0 ] && { log "CRAFT3 DONE (all 30 shards)"; break; }
  sleep 600
done
