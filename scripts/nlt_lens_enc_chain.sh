#!/bin/bash
# critic-scale chain (lens agent): wait for infra's final blind prior on the volume, then run the four text-encoder arms
# in parallel (each = adapters on the frozen prior + exact-bits eval; see nlt_lens_enc_ablation.sh).
# Pool (identical for every arm): lens-diff J-lens L0-L3 (train part00, 856k rows) + teacher-sonnet-v1 train (x4 oversampled, ~210k rows).
# Val sets (identical): lens L1 / L2 / L2m / L3 + teacher v0 / v1 / v2 on the fixed 4096, n=1024 per set, paired subset reported.
set -u
cd /home/celeste/nlt
LOG=~/nlt-lens-logs; mkdir -p $LOG
PRIOR=${PRIOR:-/vol/critic/none_v1_pooled/ckpt_final.pt}      # D2 v1.7: the POOLED prior
PRIOR_DIR=$(dirname "$PRIOR" | sed "s|^/vol/||")
T=/vol/z/teacher-sonnet-v1/train/part_*.parquet
export TRAIN="/vol/z/lensdiff_v1/train/L0_part00.parquet,/vol/z/lensdiff_v1/train/L1_part00.parquet,/vol/z/lensdiff_v1/train/L2_part00.parquet,/vol/z/lensdiff_v1/train/L3_part00.parquet,$T,$T,$T,$T"
export VALSETS="lensL1:/vol/z/lensdiff_v1/val/L1.parquet,lensL2:/vol/z/lensdiff_v1/val/L2.parquet,lensL2m:/vol/z/lensdiff_v1/val/L2m.parquet,lensL3:/vol/z/lensdiff_v1/val/L3.parquet,teacher0:/vol/z/teacher-sonnet-v1/val/part_*.parquet@0,teacher1:/vol/z/teacher-sonnet-v1/val/part_*.parquet@1,teacher2:/vol/z/teacher-sonnet-v1/val/part_*.parquet@2"
export PRIOR NLT_GPU=${NLT_GPU:-H100} NLT_APP=${NLT_APP:-nlt-lens-critic}
echo "[$(date -u +%H:%M:%S)] waiting for $PRIOR" | tee -a $LOG/enc_ablation.log
until modal volume ls nlt "$PRIOR_DIR" 2>/dev/null | grep -q "$(basename "$PRIOR")"; do sleep 60; done
echo "[$(date -u +%H:%M:%S)] prior present; launching arms ${ARMS:-e0 e1 e2 e3}" | tee -a $LOG/enc_ablation.log
for arm in ${ARMS:-e0 e1 e2 e3}; do
  setsid nohup bash scripts/nlt_lens_enc_ablation.sh $arm ${STEPS:-4000} > $LOG/enc_arm_$arm.out 2>&1 < /dev/null &
  sleep 5
done
wait
echo "[$(date -u +%H:%M:%S)] ALL ARMS DONE" | tee -a $LOG/enc_ablation.log
