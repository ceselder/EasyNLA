#!/usr/bin/env bash
# Claim twins under critic v1 ckpt_final AND ckpt_best with BOTH views (exact Heun-64 likelihood + the RL reward's FM-loss view, common noise), n 512 pairs; 1 H100 each, sequential, ledger-aware.
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; source /home/celeste/nlt-q36-logs/gpu_lib.sh
TX=/vol/q36/text/v1; LOGD=/home/celeste/nlt-q36-logs; D=/home/celeste/shared/reports/nlt-27b-olens/data
log(){ echo "[twins] $(date -u +%H:%M) $*"; }
run(){ if [ "${QUEUE_MODE:-0}" = 1 ]; then enqueue_eval "$1" "$2" "${PRIO:-9}"; return; fi; out=$(spawn_retry env NLT_Q36_GPU=H100 timeout 900 modal run --detach scripts/modal_nlt_q36.py --task hf --gpus 1 --script eval_bits.py --args "$1"); echo "$out" | sed "s/^/[$2] /" | tee -a $LOGD/apps.txt; ledger_add "$out" 1 "$2"; }
for i in $(seq 1 400); do grep -q "bits_verb_v1b\] SPAWNED" $LOGD/apps.txt && grep -q "rl_v3\] SPAWNED" $LOGD/apps.txt && break; sleep 120; done; log "RL v3 and bits_v1b_verbalizer have their GPUs; queueing behind them"
for C in final best; do
  bits_complete bits_v1${C}_twins && { log "bits_v1${C}_twins already complete"; continue; }; grep -q "twins_v1$C\] SPAWNED" $LOGD/apps.txt && { log "twins_v1$C already launched as an app"; wait_bits bits_v1${C}_twins; continue; }
  PRIO=5 wait_gpu 1 || exit 1
  run "--data-dir /vol/q36/data --ckpt /vol/q36/critic/v1/ckpt_$C.pt --out /vol/q36/results/bits_v1${C}_twins.json --sets 'craft_full:$TX/val/craft_full__*.parquet' --twins 'craft_twins:$TX/val/twins__*.parquet' --n 512 --ode-steps 64 --skip-samples" twins_v1$C
  wait_bits bits_v1${C}_twins || exit 1; cp /tmp/q36_chk_bits_v1${C}_twins.json $D/bits_v1${C}_twins.json; log "twins (both views) under v1 ckpt_$C pulled"
  python3 -c "
import json; d=json.load(open('$D/bits_v1${C}_twins.json'))['twins']['craft_twins']['variants']
for k,v in d.items(): print(f\"  {k:12s} exact P {v['p_true_gt_twin']:.3f} ({v['mean_bits_true_minus_twin']:+.2f} bits) | FM-loss view P {v.get('proxy_p_true_gt_twin',float('nan')):.3f} (twin-true {v.get('proxy_fm_twin_minus_true',float('nan')):+.4f})\")"
done
(cd /home/celeste/shared/reports/nlt-27b-olens && systemd-run --user --scope -q -p MemoryMax=1G python3 build_html.py >/dev/null 2>&1); log "TWINS PROXY DONE"
