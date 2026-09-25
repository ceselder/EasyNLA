#!/usr/bin/env bash
# Over-epoching diagnostic (c), orchestrator 07:45: exact P(z > no text), PMI and content on TRAIN rows vs HELD-OUT rows for critic v1b at steps 500 and 1500 (crafted text, n 256, Heun 64).
# Train staying high while held-out collapses = memorisation of the text pools. 4 x 1 H100, ledger/lock-aware.
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; source /home/celeste/nlt-q36-logs/gpu_lib.sh
TX=/vol/q36/text/v1; LOGD=/home/celeste/nlt-q36-logs; D=/home/celeste/shared/reports/nlt-27b-olens/data
log(){ echo "[trainval] $(date -u +%H:%M) $*"; }
run(){ out=$(NLT_Q36_GPU=H100 timeout 900 modal run --detach scripts/modal_nlt_q36.py --task hf --gpus 1 --script eval_bits.py --args "$1" 2>&1 | grep -E "SPAWNED|modal.com/apps|rror"); echo "$out" | sed "s/^/[$2] /" | tee -a $LOGD/apps.txt; ledger_add "$out" 1 "$2"; }
for ST in 000500 001500; do for SP in train val; do
  PRIO=3 wait_gpu 1 || exit 1
  run "--data-dir /vol/q36/data --ckpt /vol/q36/critic/v1b/ckpt_step$ST.pt --split $SP --out /vol/q36/results/bits_v1b_${SP}_$ST.json --sets 'craft_full:$TX/$SP/craft_full__*.parquet' --n 256 --n-fixed 1024 --ode-steps 64 --skip-samples --skip-sw" v1b_${SP}_$ST
done; done
wait_bits bits_v1b_train_000500 bits_v1b_val_000500 bits_v1b_train_001500 bits_v1b_val_001500 || exit 1
for f in bits_v1b_train_000500 bits_v1b_val_000500 bits_v1b_train_001500 bits_v1b_val_001500; do cp /tmp/q36_chk_$f.json $D/$f.json; python3 -c "
import json; d=json.load(open('$D/$f.json')); v=d['sets']['craft_full']; print(f\"$f: n {v['n']} P_null {v['p_z_gt_null']:.3f} PMI {v['pmi_bits']['mean'] if isinstance(v['pmi_bits'],dict) else v['pmi_bits']:+.1f} content {v['content_bits']['mean']:.1f} P_dm {v['p_z_gt_dm']:.3f} P_rp {v['p_z_gt_rp']:.3f}\")"; done
(cd /home/celeste/shared/reports/nlt-27b-olens && systemd-run --user --scope -q -p MemoryMax=1G python3 build_html.py >/dev/null 2>&1); log "TRAINVAL DONE"
