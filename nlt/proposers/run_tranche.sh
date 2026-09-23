#!/usr/bin/env bash
# Warm-start tranche driver (proposer agent). Usage:
#   bash nlt/proposers/run_tranche.sh <split> <start> <end> [chunk] [concurrency]
# 1) features on Modal (nlt-prop, chunked across containers) -> /vol/z/features_v1/<split>/
# 2) download features -> ~/nlt-prop-data/features_v1/<split>/
# 3) Sonnet teacher, both variants (with / without the model's final top-10), sync transport -> local parts
# 4) upload parts -> /vol/z/teacher-sonnet-v1{,-nofinal}/<split>/part_<start>_<end>.parquet
set -euo pipefail
SPLIT=$1; START=$2; END=$3; CHUNK=${4:-2048}; CONC=${5:-48}
DATA_DIR=${DATA_DIR:-/vol/data/qwen3_8b}
ROOT=/home/celeste/nlt; LOCAL=/home/celeste/nlt-prop-data
TAG=$(printf "%07d_%07d" "$START" "$END")
mkdir -p "$LOCAL/features_v1/$SPLIT" "$LOCAL/teacher/$SPLIT" "$ROOT/nlt/proposers/logs"
cd "$ROOT"
echo "[tranche] $(date -u +%H:%M:%S) features $SPLIT $START:$END chunk $CHUNK"
modal run nlt/proposers/modal_features.py --data-dir "$DATA_DIR" --split "$SPLIT" --start "$START" --end "$END" --chunk "$CHUNK" \
  > "nlt/proposers/logs/feat_${SPLIT}_${TAG}.log" 2>&1
FEATS=()
for ((s=START; s<END; s+=CHUNK)); do e=$(( s+CHUNK < END ? s+CHUNK : END )); f=$(printf "feat_%07d_%07d.parquet" "$s" "$e")
  modal volume get nlt "/z/features_v1/$SPLIT/$f" "$LOCAL/features_v1/$SPLIT/$f" --force > /dev/null; FEATS+=("$LOCAL/features_v1/$SPLIT/$f"); done
echo "[tranche] $(date -u +%H:%M:%S) features downloaded: ${#FEATS[@]} files"
for VAR in final nofinal; do
  EXTRA=""; SRC="teacher-sonnet-v1"; [[ $VAR == nofinal ]] && { EXTRA="--no-final"; SRC="teacher-sonnet-v1-nofinal"; }
  OUT="$LOCAL/teacher/$SPLIT/${SRC}_part_${TAG}.parquet"
  echo "[tranche] $(date -u +%H:%M:%S) teacher $VAR -> $OUT"
  with-local-keys python3 -m nlt.proposers.teacher_sonnet --features "${FEATS[@]}" --out "$OUT" --mode sync --concurrency "$CONC" $EXTRA \
    2>&1 | grep -v "takes precedence" | tee "nlt/proposers/logs/teacher_${VAR}_${SPLIT}_${TAG}.log" | tail -12
  modal volume put nlt "$OUT" "/z/$SRC/$SPLIT/part_${TAG}.parquet" --force > /dev/null && echo "[tranche] uploaded /z/$SRC/$SPLIT/part_${TAG}.parquet"
done
echo "[tranche] $(date -u +%H:%M:%S) done $SPLIT $START:$END"
