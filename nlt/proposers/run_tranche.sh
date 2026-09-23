#!/usr/bin/env bash
# Warm-start tranche driver (proposer agent). Usage:
#   bash nlt/proposers/run_tranche.sh <split> <start> <end> [chunk] [concurrency_per_process] [variants]
#   variants: "final nofinal" (default for val) or "final" (train)
# 1) features on Modal (nlt-prop, one container per chunk) -> /vol/z/features_v1/<split>/
# 2) download features -> ~/nlt-prop-data/features_v1/<split>/
# 3) Sonnet teacher: one process per (chunk, variant), all in parallel, sync transport -> local parts
# 4) upload parts -> /vol/z/teacher-sonnet-v1{,-nofinal}/<split>/part_<s>_<e>.parquet
set -uo pipefail
SPLIT=$1; START=$2; END=$3; CHUNK=${4:-2048}; CONC=${5:-32}; VARIANTS=${6:-"final nofinal"}
DATA_DIR=${DATA_DIR:-/vol/data/qwen3_8b}
PERM_SEED=${PERM_SEED:--1}       # >=0: rows of a fixed permutation of the pairs table (use 0 for train); val uses -1 = first rows
ZROOT=${ZROOT:-/vol/z}          # override for rehearsals so smoke pair_ids never land under /vol/z
ROOT=/home/celeste/nlt; LOCAL=${LOCAL:-/home/celeste/nlt-prop-data}
TAG=$(printf "%07d_%07d" "$START" "$END")
vget() {  # modal volume get with retries: a just-committed file can take a minute to become visible to the client
  local remote=$1 local_=$2 n=0
  until modal volume get nlt "$remote" "$local_" --force > /dev/null 2>&1; do n=$((n+1)); [[ $n -ge 12 ]] && return 1; sleep 10; done; return 0
}
mkdir -p "$LOCAL/features_v1/$SPLIT" "$LOCAL/teacher/$SPLIT" "$ROOT/nlt/proposers/logs"
cd "$ROOT"
if [[ "${SKIP_FEATURES:-0}" != "1" ]]; then
  echo "[tranche] $(date -u +%H:%M:%S) features $SPLIT $START:$END chunk $CHUNK"
  modal run nlt/proposers/modal_features.py --data-dir "$DATA_DIR" --split "$SPLIT" --start "$START" --end "$END" --chunk "$CHUNK" --out-dir "$ZROOT/features_v1" --perm-seed "$PERM_SEED" \
    > "nlt/proposers/logs/feat_${SPLIT}_${TAG}.log" 2>&1 || { echo "[tranche] features FAILED, see log"; exit 1; }
fi
declare -a CH_S CH_E
for ((s=START; s<END; s+=CHUNK)); do e=$(( s+CHUNK < END ? s+CHUNK : END )); CH_S+=("$s"); CH_E+=("$e")
  f=$(printf "feat_%07d_%07d.parquet" "$s" "$e")
  [[ -s "$LOCAL/features_v1/$SPLIT/$f" ]] || vget "$ZROOT/features_v1/$SPLIT/$f" "$LOCAL/features_v1/$SPLIT/$f" || { echo "[tranche] download FAILED $f"; exit 1; }
done
echo "[tranche] $(date -u +%H:%M:%S) features ready: ${#CH_S[@]} chunks"
PIDS=()
for VAR in $VARIANTS; do
  EXTRA=""; SRC="teacher-sonnet-v1"; [[ $VAR == nofinal ]] && { EXTRA="--no-final"; SRC="teacher-sonnet-v1-nofinal"; }
  for k in "${!CH_S[@]}"; do
    s=${CH_S[$k]}; e=${CH_E[$k]}; ct=$(printf "%07d_%07d" "$s" "$e")
    F="$LOCAL/features_v1/$SPLIT/feat_${ct}.parquet"; OUT="$LOCAL/teacher/$SPLIT/${SRC}_part_${ct}.parquet"
    if [[ -s "$OUT" ]]; then echo "[tranche] exists $OUT"; continue; fi
    ( with-local-keys python3 -m nlt.proposers.teacher_sonnet --features "$F" --out "$OUT" --mode sync --concurrency "$CONC" $EXTRA \
        2>&1 | grep -v "takes precedence" > "nlt/proposers/logs/teacher_${VAR}_${SPLIT}_${ct}.log" ) &
    PIDS+=($!)
  done
done
echo "[tranche] $(date -u +%H:%M:%S) launched ${#PIDS[@]} teacher processes (conc $CONC each)"
wait "${PIDS[@]}"
for VAR in $VARIANTS; do
  SRC="teacher-sonnet-v1"; [[ $VAR == nofinal ]] && SRC="teacher-sonnet-v1-nofinal"
  for k in "${!CH_S[@]}"; do
    ct=$(printf "%07d_%07d" "${CH_S[$k]}" "${CH_E[$k]}"); OUT="$LOCAL/teacher/$SPLIT/${SRC}_part_${ct}.parquet"
    if [[ -s "$OUT" ]]; then
      modal volume put nlt "$OUT" "$ZROOT/$SRC/$SPLIT/part_${ct}.parquet" --force > /dev/null && echo "[tranche] uploaded $ZROOT/$SRC/$SPLIT/part_${ct}.parquet ($(python3 -c "import pyarrow.parquet as q;print(q.read_metadata('$OUT').num_rows)") rows)"
    else echo "[tranche] MISSING $OUT"; fi
  done
done
echo "[tranche] $(date -u +%H:%M:%S) done $SPLIT $START:$END"
