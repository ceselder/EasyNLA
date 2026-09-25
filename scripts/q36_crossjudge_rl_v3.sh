#!/usr/bin/env bash
# Cross-judge table for RL v3 (orchestrator 07:33): the frozen v1 judge shares its init with the co-trained reward critic, so score RL v3's step-0 policy (= SFT v1b) and its LAST SAVED policy,
# plus the crafted teacher, on held-out val pairs under critics of OTHER lineages: critic v2 step 3000 and critic v1b's picked judge (+ v1 ckpt_best for reference). content, P(z>z_dm), claim twins per judge.
# Dumps are regenerated greedily (same prompt, 208-token budget as the RL eval) from the saved adapters; eval_bits takes the common rows (n 256).
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; source /home/celeste/nlt-q36-logs/gpu_lib.sh
TX=/vol/q36/text/v1; LOGD=/home/celeste/nlt-q36-logs; D=/home/celeste/shared/reports/nlt-27b-olens/data; BAND=12,16,20,24,28,30,32,36,40,42,44,48,52,54,56,60
JUDGE_V1B=${JUDGE_V1B:-/vol/q36/critic/v1b/ckpt_best.pt}
log(){ echo "[crossjudge] $(date -u +%H:%M) $*"; }
nfiles(){ timeout 120 modal volume ls nlt "$1" 2>/dev/null | grep -cE "$2" || true; }
run(){ if [ "${QUEUE_MODE:-0}" = 1 ]; then enqueue_eval "$2" "$3" "${PRIO:-4}" "$1"; return; fi; out=$(spawn_retry env NLT_Q36_GPU=H100 timeout 900 modal run --detach scripts/modal_nlt_q36.py --task hf --gpus 1 --script "$1" --args "$2"); echo "$out" | sed "s/^/[$3] /" | tee -a $LOGD/apps.txt; ledger_add "$out" 1 "$3"; }
LAST=""; for i in 1 2 3 4 5; do LAST=$(timeout 120 modal volume ls nlt q36/rl/rl_v3 2>/dev/null | grep -oE "step_[0-9]+" | sort -u | tail -n 1); [ -n "$LAST" ] && break; log "volume listing empty (attempt $i), retrying"; sleep 60; done; [ -z "$LAST" ] && { log "no saved RL v3 policy step after 5 attempts -> nothing to score"; exit 0; }
log "last saved RL v3 policy: $LAST"
DUMP=/vol/q36/dumps/rl_v3_$LAST.parquet
if [ "$(nfiles q36/dumps "rl_v3_$LAST.parquet")" -lt 1 ]; then
  PRIO=4 wait_gpu 1 || exit 1
  run dump_verbalizer.py "--data-dir /vol/q36/data --adapter /vol/q36/rl/rl_v3/$LAST --pairs-text '$TX/val/craft_full__*.parquet' --n 512 --batch 64 --band $BAND --max-new 208 --out $DUMP" dump_rlv3_$LAST
  for i in $(seq 1 60); do [ "$(nfiles q36/dumps "rl_v3_$LAST.parquet")" -ge 1 ] && break; sleep 120; done
fi
SETS="teacher:$TX/val/craft_full__*.parquet,rl_step0:/vol/q36/dumps/verbalizer_v1b.parquet,rl_$LAST:$DUMP"
declare -A JUDGES=([v2s3000]=/vol/q36/critic/v2/ckpt_step3000.pt [v1bjudge]=$JUDGE_V1B [v1best]=/vol/q36/critic/v1/ckpt_best.pt)
for J in v2s3000 v1bjudge v1best; do
  PRIO=4 wait_gpu 1 || exit 1
  run eval_bits.py "--data-dir /vol/q36/data --ckpt ${JUDGES[$J]} --out /vol/q36/results/bits_crossjudge_rlv3_$J.json --sets '$SETS' --twins 'craft_twins:$TX/val/twins__*.parquet' --n 256 --ode-steps 64 --skip-samples" cross_$J
done
wait_bits bits_crossjudge_rlv3_v2s3000 bits_crossjudge_rlv3_v1bjudge bits_crossjudge_rlv3_v1best || exit 1
for J in v2s3000 v1bjudge v1best; do cp /tmp/q36_chk_bits_crossjudge_rlv3_$J.json $D/bits_crossjudge_rlv3_$J.json; python3 -c "
import json; d=json.load(open('$D/bits_crossjudge_rlv3_$J.json')); print('$J', d['ckpt'].split('/')[-2:], 'step', d['step'], 'n_common', d['n_common'])
for k,v in d['sets'].items(): print(f\"  {k:14s} content {v['content_bits']['mean']:5.1f} ± {v['content_bits']['sem']:.1f} P_dm {v['p_z_gt_dm']:.3f} P_rp {v['p_z_gt_rp']:.3f} tok {v['n_tokens_mean']:.0f}\")
for var,x in d.get('twins',{}).get('craft_twins',{}).get('variants',{}).items(): print(f\"  twin {var:12s} exact P {x['p_true_gt_twin']:.3f} | FM view P {x.get('proxy_p_true_gt_twin',float('nan')):.3f}\")"; done
(cd /home/celeste/shared/reports/nlt-27b-olens && systemd-run --user --scope -q -p MemoryMax=1G python3 build_html.py >/dev/null 2>&1); log "CROSSJUDGE DONE"
