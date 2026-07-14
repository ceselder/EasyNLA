#!/bin/bash
# AV + AR SFT warmstart for Gemma-4-26B-A4B-it, run in parallel (GPU0=AV, GPU1=AR).
# bf16 base + LoRA r=128 (no 4-bit — B200s have room; keeps the later merge lossless).
# Hyperparameters follow docs/train_new_model.md (nanoNLA defaults).
set -euo pipefail
DATA=${DATA:-/workspace/nla/data/gemma4-26b-a4b-L20}
MODEL=${MODEL:-/workspace/nla/models/gemma-4-26B-A4B-it-text}
CKPT=${CKPT:-/workspace/nla/ckpts}
STEPS=${STEPS:-1000}
PY=/workspace/nla/venv/bin/python
LOGS=/workspace/nla/logs
cd /workspace/nla/EasyNLA
source /workspace/nla/env.sh
mkdir -p "$CKPT" "$LOGS"

# AV batch 16 × accum 4 (effective 64, the nanoNLA default): the lm-head logits
# tensor scales with batch × 262k vocab — batch 64 OOM'd a B200 at backward.
CUDA_VISIBLE_DEVICES=${AV_GPU:-0} nohup $PY -m nla.train_sft --mode av --base-ckpt "$MODEL" \
  --parquet $DATA/av_sft_train.parquet --sidecar $DATA/av_sft_train.parquet \
  --save-dir $CKPT/av_sft \
  --num-steps "$STEPS" --batch-size 16 --gradient-accumulation-steps 4 \
  --use-lora --lora-r 128 --lora-alpha 16 \
  --lr 1e-4 --min-lr 1e-5 --lr-warmup-steps 50 --max-grad-norm 1.0 \
  --heldout-parquet $DATA/av_sft_val.parquet --heldout-rows 1000 --heldout-every 100 \
  --save-every 500 --wandb-project easynla-gemma4 --wandb-name av_sft_26b_L20 --seed 0 \
  > "$LOGS/av_sft.log" 2>&1 &
echo "AV SFT launched on GPU ${AV_GPU:-0} (log: $LOGS/av_sft.log)"

# --ar-num-layers 24 (not layer_index+1=21): gemma4 FORCES the last layer to
# full_attention at config init. Block 20 is sliding — a 21-block critic saves
# sliding weights but reloads as forced-full (head dim 512, no v_proj) and
# crashes with shape mismatches. Block 23 is naturally full_attention, so a
# 24-block critic roundtrips cleanly; the 3 extra pretrained blocks are legal
# (critic needs AT LEAST layer_index+1 blocks).
CUDA_VISIBLE_DEVICES=${AR_GPU:-1} nohup $PY -m nla.train_sft --mode ar --base-ckpt "$MODEL" \
  --parquet $DATA/ar_sft.parquet --sidecar $DATA/ar_sft.parquet \
  --save-dir $CKPT/ar_sft \
  --num-steps "$STEPS" --batch-size 64 --gradient-accumulation-steps 1 \
  --use-lora --lora-r 128 --lora-alpha 16 --ar-num-layers 24 \
  --lr 2e-5 --min-lr 2e-6 --lr-warmup-steps 50 --max-grad-norm 1.0 \
  --heldout-parquet $DATA/av_sft_val.parquet --heldout-rows 1000 --heldout-every 100 \
  --save-every 500 --wandb-project easynla-gemma4 --wandb-name ar_sft_26b_L20 --seed 0 \
  > "$LOGS/ar_sft.log" 2>&1 &
echo "AR SFT launched on GPU ${AR_GPU:-1} (log: $LOGS/ar_sft.log)"
