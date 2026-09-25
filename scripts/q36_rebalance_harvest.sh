#!/usr/bin/env bash
# Rebalance the harvest tail (13:40): engine 1 is done, engines 0 + 2 finish their last shard in ~25 min, engine 5 (relaunched late) still has train:2, train:8, val:2 after shard26 (~3 h alone).
# Helper engines take val:2 and train:8 (rollout_vllm skips spec files that already exist, so an overlap only re-reads one in-flight layer). .harvest_done is then reached ~1 h earlier.
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; source /home/celeste/nlt-q36-logs/gpu_lib.sh
LOGD=/home/celeste/nlt-q36-logs; LAYERS="h_L12;h_L16;h_L20;h_L24;h_L28;h_L30;h_L32;h_L36;h_L40;h_L42;h_L44;h_L48;h_L52;h_L54;h_L56;h_L60"; MAXTOK=72; log(){ echo "[rebal] $(date -u +%H:%M) $*"; }
for tok in val:2 train:8; do
  PRIO=0 wait_gpu 1 || exit 1
  out=$(spawn_retry env NLT_Q36_GPU=B200 timeout 900 modal run --detach scripts/modal_nlt_q36.py --task vllm --gpus 1 --script rollout_vllm.py --args "--data-dir /vol/q36/data --pairs-shards '$tok' --layer-specs '$LAYERS' --delta-pairs /vol/q36/data/pairs_all_x4.parquet --adapter /vol_go/ckpt/ar_ivrl/final --prompt bullets --n-samples 0 --max-tokens $MAXTOK --grammar --out-dir /vol/q36/rollouts_layers --out-dir-delta /vol/q36/rollouts_delta")
  echo "$out" | sed "s/^/[harvest_helper_${tok/:/_}] /" | tee -a $LOGD/apps.txt; ledger_add "$out" 1 harvest_helper_${tok/:/_}; log "helper for $tok launched: $(echo "$out" | grep -oE 'ap-[A-Za-z0-9]+' | head -1)"
done
log "REBALANCE DONE"
