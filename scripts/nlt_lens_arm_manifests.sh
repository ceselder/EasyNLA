#!/bin/bash
# Score redteam's control manifests (board #180) with one critic arm, so the form / depth-generic / content decomposition and
# P(z > z_dm) come out of nlt.evals.summarize_scored. Output: /vol/evals/scored_<TAG>_manifest2_<src>.parquet (+ .summary.json).
#   usage: nlt_lens_arm_manifests.sh <TAG> [srcs...]     (TAG = critic dir under /vol/critic, e.g. enc_e0_c)
set -u
TAG=$1; shift
SRCS=${*:-"lensdiff_jlens_L1 lensdiff_jlens_L2 teacher_v1"}
cd /home/celeste/nlt
export NLT_GPU=${NLT_GPU:-H100} NLT_APP=${NLT_APP:-nlt-lens-critic}
LOG=~/nlt-lens-logs
for src in $SRCS; do
  echo "[$(date -u +%H:%M:%S)] manifest2_$src x $TAG" | tee -a $LOG/enc_ablation.log
  modal run scripts/modal_nlt_critic.py --task manifest --tag ${TAG}_manifest2_$src --data /vol/data/qwen3_8b \
    --extra "--ckpt /vol/critic/$TAG/ckpt_final.pt --manifest /vol/evals/manifest2_$src.parquet --n 8000 --ode-steps 32 --data-device cpu" \
    > $LOG/manifest_${TAG}_$src.log 2>&1
  grep -E "summary|verdict|content|P\(" $LOG/manifest_${TAG}_$src.log | tail -6 | cut -c1-300
done
echo "[$(date -u +%H:%M:%S)] MANIFESTS DONE $TAG" | tee -a $LOG/enc_ablation.log
