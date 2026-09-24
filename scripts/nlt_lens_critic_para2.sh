#!/bin/bash
# DECISIONS v1.24: critic_para = the enc_e2 recipe (Qwen3-8B L24 encoder, FROZEN squash prior none_v1_squash, plain FM, no null term,
# no text-in-proj) trained on original texts + paraphrases (same target). Stage 1 (originals) resumes from enc_e2_sq (3000 steps on the
# lens+teacher pool); stage 2 restarts from the latest checkpoint with the paraphrase rows added.
#   usage: nlt_lens_critic_para.sh <TAG> <RESUME ckpt> [extra train files, comma list]      env: STEPS (total incl. resumed), NLT_GPU (B200)
set -u
TAG=$1; RESUME=$2; EXTRA_TRAIN=${3:-}
cd /home/celeste/nlt
export NLT_GPU=${NLT_GPU:-B200} NLT_APP=${NLT_APP:-nlt-lens-critic}
LOG=~/nlt-lens-logs; D=/vol/data/qwen3_8b
PRIOR=${PRIOR:-/vol/critic/none_v1_pooled/ckpt_final.pt}; SQUASH=${SQUASH:-0}   # v1.25: POOLED prior (the squash space breaks text adapters)
T=/vol/z/teacher-sonnet-v1/train/part_*.parquet
TRAIN="/vol/z/lensdiff_v1/train/L0_part00.parquet,/vol/z/lensdiff_v1/train/L1_part00.parquet,/vol/z/lensdiff_v1/train/L2_part00.parquet,/vol/z/lensdiff_v1/train/L3_part00.parquet,$T,$T${EXTRA_TRAIN:+,$EXTRA_TRAIN}"
VAL="/vol/z/lensdiff_v1/val/L0.parquet,/vol/z/lensdiff_v1/val/L1.parquet,/vol/z/lensdiff_v1/val/L2.parquet,/vol/z/lensdiff_v1/val/L3.parquet,/vol/z/teacher-sonnet-v1/val/part_*.parquet"
echo "[$(date -u +%H:%M:%S)] critic_para $TAG: resume $RESUME; extra train: ${EXTRA_TRAIN:-none}" | tee -a $LOG/critic_para.log
# --resume copies the checkpoint dir? no: --out is the new dir, --resume the file to continue from (model + opt + step)
modal run scripts/modal_nlt_critic.py --task train --tag $TAG --data $D --extra "--cond text --init-from $PRIOR --freeze-prior 1 --src-rms 0 --squash $SQUASH --resume $RESUME --text-parquet $TRAIN --val-text-parquet $VAL --steps ${STEPS:-20000} --batch 512 --lr ${LR:-2e-5} --lr-decay none --warmup 300 --eval-every 500 --eval-n 2048 --save-every 500 --keep-every 500 --d-model 3072 --d-mlp 12288 --n-layers 12 --n-slots 16 --n-heads 8 --d-head 64 --gate-rank 256 --p-uncond 0.3 --enc-max-len 256 --data-device cpu --enc-model Qwen/Qwen3-8B --enc-layer 24 --max-hours ${MAX_HOURS:-3.5}" > $LOG/critic_$TAG.log 2>&1
echo "[$(date -u +%H:%M:%S)] critic_para $TAG train exited rc=$?" | tee -a $LOG/critic_para.log
