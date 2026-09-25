#!/usr/bin/env bash
# waits for the 22 phase-0 rollout files, then launches phase0_score.py (1 B200)
unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt
for i in $(seq 1 60); do
  n=$(timeout 90 modal volume ls nlt q36/phase0/rollouts 2>/dev/null | grep -c "parquet$")
  echo "$(date -u +%H:%M) rollout files=$n"
  [ "$n" -ge 22 ] && break
  sleep 120
done
echo "$(date -u +%H:%M) launching phase0_score"
timeout 900 modal run --detach scripts/modal_nlt_q36.py --task hf --gpus 1 --script phase0_score.py --args "--acts /vol/q36/phase0/acts_4k.parquet --rollouts-dir /vol/q36/phase0/rollouts --ar /vol_go/ckpt/h2hpfx_ar_r512/final --out /vol/q36/phase0/phase0_metrics.json --examples-out /vol/q36/phase0/examples.json" 2>&1 | grep -E "SPAWNED|modal.com/apps|Error|error" | tee -a /home/celeste/nlt-q36-logs/apps.txt
echo CHAIN_PHASE0_LAUNCHED
