#!/bin/bash
# Re-harvest Gemma-4 activations for all three warmstart splits (6-GPU sharded),
# then build the stage3 training parquets + AV train/val doc-hash split.
# Run on the 6×B200 box: bash scripts/gemma4_run_rebuild.sh
set -euo pipefail
DATA_IN=${DATA_IN:-/workspace/nla/data/qwen3-8b-nla-L24}
DATA_OUT=${DATA_OUT:-/workspace/nla/data/gemma4-26b-a4b-L20}
MODEL=${MODEL:-/workspace/nla/models/gemma-4-26B-A4B-it-text}
LAYER=${LAYER:-20}
DTYPE=${DTYPE:-bf16}
PY=/workspace/nla/venv/bin/python
LOGS=/workspace/nla/logs
mkdir -p "$DATA_OUT" "$LOGS"
cd /workspace/nla/EasyNLA
source /workspace/nla/env.sh

declare -A INPUTS=(
  [rl]=$DATA_IN/rl_shuf.parquet
  [av_sft]=$DATA_IN/av_sft_shuf.parquet
  [ar_sft]=$DATA_IN/ar_sft_shuf.parquet
)

for SPLIT in rl av_sft ar_sft; do
  IN=${INPUTS[$SPLIT]}
  OUT=$DATA_OUT/${SPLIT}_base.parquet
  if [[ -f "$OUT" ]]; then echo "=== $SPLIT already done, skipping ==="; continue; fi
  echo "=== reharvest $SPLIT ($(date)) ==="
  pids=()
  for G in 0 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES=$G $PY scripts/gemma4_reharvest.py \
      --input "$IN" --split $SPLIT --model "$MODEL" --layer-index $LAYER \
      --output "$OUT" --shard-index $G --num-shards 6 --dtype "$DTYPE" \
      --token-budget 131072 \
      > "$LOGS/reharvest_${SPLIT}_g${G}.log" 2>&1 &
    pids+=($!)
  done
  fail=0
  for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  if [[ $fail -ne 0 ]]; then echo "shard failure for $SPLIT — see $LOGS/reharvest_${SPLIT}_g*.log"; exit 1; fi
  $PY scripts/gemma4_reharvest.py --concat --input "$IN" --split $SPLIT \
    --model "$MODEL" --layer-index $LAYER --output "$OUT" --num-shards 6
done

echo "=== stage3 build ($(date)) ==="
$PY -m nla.datagen.stage3_build --input $DATA_OUT/av_sft_base.parquet --stage av_sft --output $DATA_OUT/av_sft.parquet
$PY -m nla.datagen.stage3_build --input $DATA_OUT/ar_sft_base.parquet --stage ar_sft --output $DATA_OUT/ar_sft.parquet
$PY -m nla.datagen.stage3_build --input $DATA_OUT/rl_base.parquet --stage rl --output $DATA_OUT/rl.parquet

echo "=== av train/val doc-hash split ==="
$PY scripts/split_val_by_dochash.py --parquet $DATA_OUT/av_sft.parquet \
  --out-train $DATA_OUT/av_sft_train.parquet --out-val $DATA_OUT/av_sft_val.parquet --permille 20

echo "REBUILD DONE ($(date))"
