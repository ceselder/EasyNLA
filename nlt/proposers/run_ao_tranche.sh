#!/usr/bin/env bash
# AO proposer tranche (proposer agent). Usage:
#   bash nlt/proposers/run_ao_tranche.sh <split> <start> <end> [chunk] [concurrency_per_process]
# 1) AO src/tgt/delta raw generations on Modal (nlt-prop, <=4 containers) -> /vol/z/ao_raw_v1/<split>/
# 2) download raw + the matching features (for the copy filter's prefix)
# 3) Sonnet register rewrite, one process per chunk -> local ao-*-v1 parts
# 4) upload -> /vol/z/ao-{src,tgt,delta}-v1/<split>/part_<s>_<e>.parquet
set -uo pipefail
SPLIT=$1; START=$2; END=$3; CHUNK=${4:-2048}; CONC=${5:-16}
DATA_DIR=${DATA_DIR:-/vol/data/qwen3_8b}
PERM_SEED=${PERM_SEED:--1}       # >=0: rows of a fixed permutation of the pairs table (use 0 for train); val uses -1 = first rows
ROOT=/home/celeste/nlt; LOCAL=/home/celeste/nlt-prop-data
TAG=$(printf "%07d_%07d" "$START" "$END")
mkdir -p "$LOCAL/ao_raw_v1/$SPLIT" "$LOCAL/ao_rewrite/$SPLIT" "$ROOT/nlt/proposers/logs"
cd "$ROOT"
if [[ "${SKIP_AO:-0}" != "1" ]]; then
  echo "[ao-tranche] $(date -u +%H:%M:%S) AO raw $SPLIT $START:$END chunk $CHUNK"
  modal run nlt/proposers/modal_ao_proposers.py::run_ao --data-dir "$DATA_DIR" --split "$SPLIT" --start "$START" --end "$END" --chunk "$CHUNK" --perm-seed "$PERM_SEED" \
    > "nlt/proposers/logs/ao_${SPLIT}_${TAG}.log" 2>&1 || { echo "[ao-tranche] AO FAILED, see log"; exit 1; }
fi
PIDS=(); declare -a CTS
for ((s=START; s<END; s+=CHUNK)); do e=$(( s+CHUNK < END ? s+CHUNK : END )); ct=$(printf "%07d_%07d" "$s" "$e"); CTS+=("$ct")
  RAW="$LOCAL/ao_raw_v1/$SPLIT/ao_${ct}.parquet"; FEAT="$LOCAL/features_v1/$SPLIT/feat_${ct}.parquet"
  [[ -s "$RAW" ]] || modal volume get nlt "/z/ao_raw_v1/$SPLIT/ao_${ct}.parquet" "$RAW" --force > /dev/null || { echo "[ao-tranche] download FAILED $RAW"; exit 1; }
  [[ -s "$FEAT" ]] || modal volume get nlt "/z/features_v1/$SPLIT/feat_${ct}.parquet" "$FEAT" --force > /dev/null || { echo "[ao-tranche] features missing for $ct (run the teacher tranche first)"; exit 1; }
  OUTD="$LOCAL/ao_rewrite/$SPLIT/$ct"
  if [[ -d "$OUTD/ao-tgt-v1" ]]; then echo "[ao-tranche] exists $OUTD"; continue; fi
  ( with-local-keys python3 -m nlt.proposers.rewrite_register --raw "$RAW" --features "$FEAT" --out-dir "$OUTD" --mode sync --concurrency "$CONC" --tag "part_${ct}" \
      2>&1 | grep -v "takes precedence" > "nlt/proposers/logs/rewrite_${SPLIT}_${ct}.log" ) &
  PIDS+=($!)
done
echo "[ao-tranche] $(date -u +%H:%M:%S) launched ${#PIDS[@]} rewrite processes"
[[ ${#PIDS[@]} -gt 0 ]] && wait "${PIDS[@]}"
for ct in "${CTS[@]}"; do
  for SRC in ao-src-v1 ao-tgt-v1 ao-delta-v1; do
    F="$LOCAL/ao_rewrite/$SPLIT/$ct/$SRC/part_${ct}.parquet"
    if [[ -s "$F" ]]; then modal volume put nlt "$F" "/z/$SRC/$SPLIT/part_${ct}.parquet" --force > /dev/null && echo "[ao-tranche] uploaded /z/$SRC/$SPLIT/part_${ct}.parquet ($(python3 -c "import pyarrow.parquet as q;print(q.read_metadata('$F').num_rows)") rows)"
    else echo "[ao-tranche] no rows for $SRC $ct"; fi
  done
done
echo "[ao-tranche] $(date -u +%H:%M:%S) done $SPLIT $START:$END"
