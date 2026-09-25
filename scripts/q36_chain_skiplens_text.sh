#!/usr/bin/env bash
# skip-lens(J̄Δ) as a text source on the phase-1 VAL pairs (shard 0 = the fixed eval set): vectors -> skip-lens rollout -> text pool skiplens_jd__<part>.parquet
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; MAXG=${MAXG:-8}
GPUS_HF=${GPUS_HF:-"H100"}; GPUS_VLLM=${GPUS_VLLM:-"H100"}; GPUS_BIG=${GPUS_BIG:-"H200"}   # Modal 1.5.4 takes ONE gpu type per function (no fallback lists): route around the B200 queue explicitly
log(){ echo "[sltext] $(date -u +%H:%M) $*"; }
nfiles(){ timeout 120 modal volume ls nlt "$1" 2>/dev/null | grep -cE "$2" || true; }
waitn(){ for i in $(seq 1 300); do n=$(nfiles "$1" "$2"); [ "$n" -ge "$3" ] && { log "ready: $1"; return 0; }; sleep 90; done; log "TIMEOUT $1"; return 1; }
gpus_in_use(){ timeout 90 modal app list 2>/dev/null | grep "nlt-q36" | grep -vE "stopped" | awk -F'│' '{gsub(/ /,"",$5); s+=$5} END {print s+0}'; }
wait_gpu(){ for i in $(seq 1 600); do g=$(gpus_in_use); [ $((g + 1)) -le $MAXG ] && return 0; sleep 120; done; return 1; }
run(){ NLT_Q36_GPU="${GT:-$GPUS_HF}" timeout 900 modal run --detach scripts/modal_nlt_q36.py --task "$1" --gpus 1 --script "$2" --args "$3" 2>&1 | grep -E "SPAWNED|modal.com/apps|rror" | sed "s/^/[$4] /" | tee -a /home/celeste/nlt-q36-logs/apps.txt; }
PART=$(python3 -c "import json,os; print(os.path.basename(json.load(open('/home/celeste/nlt-q36-data/splits.json'))['val'][0]).replace('.parquet',''))" 2>/dev/null || echo shard29_part0000)
if [ "$(nfiles q36/phase0b 'jdc_val0.parquet$')" -lt 1 ]; then wait_gpu || exit 1; run hf skiplens_pairs.py "--data-dir /vol/q36/data --split val --shard 0 --out /vol/q36/phase0b/jdc_val0.parquet" jdc; waitn q36/phase0b "jdc_val0.parquet$" 1 || exit 1; fi
if [ "$(nfiles q36/phase0b/skiplens_val0 'v_jdc.parquet$')" -lt 1 ]; then wait_gpu || exit 1; GT="$GPUS_VLLM" run vllm rollout_vllm.py "--data /vol/q36/phase0b/jdc_val0.parquet --adapter /vol/q36/ckpt/skiplens_repeat --prompt skiplens --max-tokens 20 --min-tokens 3 --specs v_jdc --n-samples 0 --out-dir /vol/q36/phase0b/skiplens_val0" slroll; waitn q36/phase0b/skiplens_val0 "v_jdc.parquet$" 1 || exit 1; fi
mkdir -p /home/celeste/nlt-q36-data/sltext; timeout 300 modal volume get nlt q36/phase0b/skiplens_val0/v_jdc.parquet /home/celeste/nlt-q36-data/sltext/v_jdc.parquet --force >/dev/null 2>&1
systemd-run --user --scope -q -p MemoryMax=1500M python3 - <<'PY'
import pyarrow as pa, pyarrow.parquet as pq
t = pq.read_table("/home/celeste/nlt-q36-data/sltext/v_jdc.parquet").to_pandas(); t = t[t["sample"] == 0]
out = pa.table({"pair_id": t["pair_id"].tolist(), "text": [x.strip() for x in t["text"]], "source": ["skiplens_jd"] * len(t), "sample": pa.array([0] * len(t), pa.int32())})
pq.write_table(out, "/home/celeste/nlt-q36-data/sltext/skiplens_jd__val0.parquet"); print("pool rows", len(t))
PY
timeout 300 modal volume put nlt /home/celeste/nlt-q36-data/sltext/skiplens_jd__val0.parquet q36/text/v1/val/skiplens_jd__$PART.parquet --force >/dev/null 2>&1 && log "uploaded skiplens_jd__$PART.parquet"
log "SLTEXT DONE"
