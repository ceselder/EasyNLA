#!/usr/bin/env bash
# Critic v3 = v2's full text mix + the ACTIVATION-ANCHORED contrast (train_critic.py --anchor): text fixed, activation varied.
# GATE (orchestrator 06:28): launch only if the claim twins are still inverted under the best-calibrated v1 checkpoint (bits_v1best_main, twin_shift P < 0.5).
# Never text-edit negatives. Saves every 500 steps so the judge can be chosen on P(z>z_dm) + content, not FM loss.
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; source /home/celeste/nlt-q36-logs/gpu_lib.sh
BAND=12,16,20,24,28,30,32,36,40,42,44,48,52,54,56,60; TX=/vol/q36/text/v2; TX1=/vol/q36/text/v1; LOGD=/home/celeste/nlt-q36-logs; D=/home/celeste/shared/reports/nlt-27b-olens/data
ANCHOR=${ANCHOR:-0.5}; ATAU=${ATAU:-0.05}; AFRAC=${AFRAC:-0.5}; CRITIC_STEPS=${CRITIC_STEPS:-4500}; UNCOND_STEPS=${UNCOND_STEPS:-1500}
log(){ echo "[v3] $(date -u +%H:%M) $*"; }
run(){ out=$(NLT_Q36_GPU="${GT:-H100}" timeout 900 modal run --detach scripts/modal_nlt_q36.py --task hf --gpus 1 --script "$1" --args "$2" 2>&1 | grep -E "SPAWNED|modal.com/apps|Error|rror:"); echo "$out" | sed "s/^/[$3] /" | tee -a $LOGD/apps.txt; ledger_add "$out" 1 "$3"; }
POOLS="craft_full=0.18:$TX/train/craft_full__*.parquet,describer_A=0.14:$TX1/train/describer_sonnet5_A__*.parquet,describer_W=0.14:$TX/train/describer_sonnet5_W__*.parquet,raw_all=0.08:$TX/train/raw_all__*.parquet,raw_all_w=0.10:$TX/train/raw_all_w__*.parquet,writes_only=0.06:$TX/train/writes_only__*.parquet,raw_loo=0.08:$TX/train/raw_no_*__*.parquet;$TX/train/raw_w_no_*__*.parquet,craft_delta=0.06:$TX/train/craft_delta__*.parquet,craft_newfaded=0.04:$TX/train/craft_newfaded__*.parquet,jlens=0.06:$TX/train/jlens__*.parquet,olens_j=0.06:$TX/train/olens_j__*.parquet"
VALS="craft_full:$TX/val/craft_full__*.parquet,describer_A:$TX1/val/describer_sonnet5_A__*.parquet,describer_W:$TX/val/describer_sonnet5_W__*.parquet,raw_all_w:$TX/val/raw_all_w__*.parquet,raw_all:$TX/val/raw_all__*.parquet"
if [ "${FORCE:-0}" != 1 ]; then
  wait_bits bits_v1best_main || exit 1; cp /tmp/q36_chk_bits_v1best_main.json $D/bits_v1best_main.json
  P=$(python3 -c "import json; d=json.load(open('$D/bits_v1best_main.json')); v=d.get('twins',{}).get('craft_twins',{}).get('variants',{}); print(v.get('twin_shift',{}).get('p_true_gt_twin','nan'), v.get('twin_new',{}).get('p_true_gt_twin','nan'))")
  log "twins under v1 ckpt_best: P(true > twin_shift, twin_new) = $P"
  python3 -c "import sys; a,b=[float(x) for x in '$P'.split()]; sys.exit(0 if max(a,b) < 0.5 else 1)" || { log "twins NOT inverted under the calibrated checkpoint -> critic v3 not launched (the inversion was the drift)"; exit 0; }
  log "still inverted -> launching critic v3 (anchor $ANCHOR tau $ATAU frac $AFRAC)"
fi
GT=H100 wait_gpu 1 || exit 1
run train_critic.py "--data-dir /vol/q36/data --out /vol/q36/critic/v3 --tag critic_v3 --pools '$POOLS' --val-sets '$VALS' --band $BAND --width 1536 --depth 16 --heads 16 --param v --uncond-steps $UNCOND_STEPS --steps $CRITIC_STEPS --keep-every 500 --batch 1024 --micro-batch 128 --eval-every 500 --eval-n 256 --spot-exact-n 64 --spot-ode-steps 16 --max-hours 8.0 --anchor $ANCHOR --anchor-tau $ATAU --anchor-frac $AFRAC" critic_v3
log "critic v3 launched"
