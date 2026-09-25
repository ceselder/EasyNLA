#!/usr/bin/env bash
# critic v1b (no unconditional pretraining) judged on its held-out sets WITH claim twins (exact + FM view) and same-document neighbours, on ckpt_final AND ckpt_best (replaces the last stage of chain_critic_v1b2.sh, which had no --twins).
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; source /home/celeste/nlt-q36-logs/gpu_lib.sh
TX=/vol/q36/text/v1; LOGD=/home/celeste/nlt-q36-logs; D=/home/celeste/shared/reports/nlt-27b-olens/data
log(){ echo "[v1bmain] $(date -u +%H:%M) $*"; }
nfiles(){ timeout 120 modal volume ls nlt "$1" 2>/dev/null | grep -cE "$2" || true; }
run(){ out=$(NLT_Q36_GPU=H100 timeout 900 modal run --detach scripts/modal_nlt_q36.py --task hf --gpus 1 --script eval_bits.py --args "$1" 2>&1 | grep -E "SPAWNED|modal.com/apps|rror"); echo "$out" | sed "s/^/[$2] /" | tee -a $LOGD/apps.txt; ledger_add "$out" 1 "$2"; }
# priority 2 (orchestrator 07:57): the RL v4 judge (step 500) FIRST, without waiting for v1b's final; then best / final once they exist
for C in step000500 final best; do
  if [ $C != step000500 ]; then for i in $(seq 1 600); do [ "$(nfiles q36/critic/v1b 'ckpt_final.pt')" -ge 1 ] && break; sleep 180; done; fi
  OUT=bits_v1b_main; [ $C = best ] && OUT=bits_v1bbest_main; [ $C = step000500 ] && OUT=bits_v1bs500_main
  PRIO=2 wait_gpu 1 || exit 1
  run "--data-dir /vol/q36/data --ckpt /vol/q36/critic/v1b/ckpt_$C.pt --out /vol/q36/results/$OUT.json --sets 'craft_full:$TX/val/craft_full__*.parquet,describer_A:$TX/val/describer_sonnet5_A__*.parquet,raw_all:$TX/val/raw_all__*.parquet' --twins 'craft_twins:$TX/val/twins__*.parquet' --neighbors /vol/q36/data/neigh --neighbor-n 128 --n 256 --ode-steps 64 --skip-samples" v1bmain_$C
done
wait_bits bits_v1bs500_main bits_v1b_main bits_v1bbest_main || exit 1; for f in bits_v1bs500_main bits_v1b_main bits_v1bbest_main; do cp /tmp/q36_chk_$f.json $D/$f.json; done
timeout 120 modal volume get nlt q36/critic/v1b/eval_latest.json $D/critic_v1b_eval_latest.json --force >/dev/null 2>&1
for f in bits_v1bs500_main bits_v1b_main bits_v1bbest_main; do python3 -c "
import json; d=json.load(open('$D/$f.json')); print('$f ckpt', d['ckpt'].split('/')[-1], 'step', d['step'])
for k,v in d['sets'].items(): print(f\"  {k:12s} content {v['content_bits']['mean']:5.1f} ± {v['content_bits']['sem']:.1f} P_dm {v['p_z_gt_dm']:.3f} P_null {v.get('p_z_gt_null'):.3f} PMI {v['pmi_bits']['mean'] if isinstance(v['pmi_bits'],dict) else v['pmi_bits']:.1f}\")
for var,x in d.get('twins',{}).get('craft_twins',{}).get('variants',{}).items(): print(f\"  twin {var:12s} exact P {x['p_true_gt_twin']:.3f} ({x['mean_bits_true_minus_twin']:+.2f} bits) | FM view P {x.get('proxy_p_true_gt_twin',float('nan')):.3f}\")"; done
(cd /home/celeste/shared/reports/nlt-27b-olens && systemd-run --user --scope -q -p MemoryMax=1G python3 build_html.py >/dev/null 2>&1); log "V1B MAIN DONE"
