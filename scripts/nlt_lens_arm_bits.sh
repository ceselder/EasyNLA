#!/bin/bash
# Exact-bits eval (+ redteam manifests) for one finished critic arm; waits until the checkpoint is visible on the volume first
# (the train container's volume commit can land after `modal run` returns -> FileNotFoundError race seen on enc_e0).
#   usage: nlt_lens_arm_bits.sh <TAG>          env: VALSETS (label:path,...), NLT_GPU (H100), MIX (optional --mix-ckpt path)
set -u
TAG=$1
cd /home/celeste/nlt
export NLT_GPU=${NLT_GPU:-H100} NLT_APP=${NLT_APP:-nlt-lens-critic}
LOG=~/nlt-lens-logs; D=/vol/data/qwen3_8b
T=/vol/z/teacher-sonnet-v1/val/part_*.parquet
VALSETS=${VALSETS:-"lensL1:/vol/z/lensdiff_v1/val/L1.parquet,lensL2:/vol/z/lensdiff_v1/val/L2.parquet,lensL2m:/vol/z/lensdiff_v1/val/L2m.parquet,lensL3:/vol/z/lensdiff_v1/val/L3.parquet,teacher0:$T@0,teacher1:$T@1,teacher2:$T@2"}
MIXARG=${MIX:+--mix-ckpt $MIX}
for try in 1 2 3 4 5 6; do
  modal volume ls nlt critic/$TAG 2>/dev/null | grep -q "ckpt_final.pt" && break
  echo "[$(date -u +%H:%M:%S)] $TAG: ckpt_final.pt not visible yet (try $try)"; sleep 60
done
echo "[$(date -u +%H:%M:%S)] bits $TAG" | tee -a $LOG/enc_ablation.log
modal run scripts/modal_nlt_critic.py --task bits --tag $TAG --data $D --extra "--ckpts text:/vol/critic/$TAG/ckpt_final.pt --text-parquet $VALSETS --n 1024 --batch 64 --ode-steps 32 --data-device cpu $MIXARG" > $LOG/bits_$TAG.log 2>&1
grep "\[bits\]" $LOG/bits_$TAG.log | grep -v "rows\|set " | tail -10 | cut -c1-400
bash scripts/nlt_lens_arm_manifests.sh $TAG
