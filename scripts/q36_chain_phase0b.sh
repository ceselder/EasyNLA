#!/usr/bin/env bash
# Phase 0b: skip-lens(J-bar . Delta) side experiment. 1 GPU at a time, launched only when the nlt-q36 apps use <= MAXG-1 GPUs.
#   extract_writes on the phase-0 acts (A/M prefix sums for the 4096 rows) -> jlens_vectors (h_62 + next-token top-10 + J-transported states / diffs / writes)
#   -> skip-lens rollouts batch 1 (h_L62 home turf, Jh_L*, Jd_* ; greedy + 3 samples) -> score + examples (message point)
#   -> skip-lens rollouts batch 2 (Jdj_*, Jdc_*, JA_*, JM_* ; greedy + 1 sample) -> final score
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt
MAXG=${MAXG:-8}; GAPS="24-42,30-42,36-48,42-54,42-60,30-54"; LAYERS="24,30,36,42,48,54,60"
log(){ echo "[chain0b] $(date -u +%H:%M) $*"; }
nfiles(){ timeout 120 modal volume ls nlt "$1" 2>/dev/null | grep -cE "$2" || true; }
waitn(){ for i in $(seq 1 400); do n=$(nfiles "$1" "$2"); [ "$n" -ge "$3" ] && { log "ready: $1 ($n >= $3)"; return 0; }; [ $((i % 5)) -eq 0 ] && log "waiting $1: $n/$3"; sleep 120; done; log "TIMEOUT waiting $1"; return 1; }
gpus_in_use(){ timeout 90 modal app list 2>/dev/null | grep "nlt-q36" | grep -vE "stopped" | awk -F'│' '{gsub(/ /,"",$5); s+=$5} END {print s+0}'; }
wait_gpu(){ for i in $(seq 1 400); do g=$(gpus_in_use); [ "$g" -le $((MAXG - 1)) ] && { log "GPU headroom: $g in use"; return 0; }; [ $((i % 5)) -eq 0 ] && log "waiting for GPU headroom ($g in use)"; sleep 120; done; return 1; }
run(){ timeout 900 modal run --detach scripts/modal_nlt_q36.py --task "$1" --gpus 1 --script "$2" --args "$3" 2>&1 | grep -E "SPAWNED|modal.com/apps|Error|rror:" | sed "s/^/[$4] /" | tee -a /home/celeste/nlt-q36-logs/apps.txt; }

# 1. writes for the phase-0 rows
if [ "$(nfiles q36/phase0/writes 'acts_4k.parquet$')" -lt 1 ]; then
  wait_gpu || exit 1; run hf extract_writes.py "--acts-dir /vol/q36/phase0 --files acts_4k.parquet --out-dir /vol/q36/phase0/writes --layers $LAYERS,42" writes0
  waitn q36/phase0/writes "acts_4k.parquet$" 1 || exit 1
fi
# 2. J-transported vectors (+ h_62, next-token top-10)
if [ "$(nfiles q36/phase0 'jvecs.parquet$')" -lt 1 ]; then
  wait_gpu || exit 1; run hf jlens_vectors.py "--acts /vol/q36/phase0/acts_4k.parquet --n-rows 2048 --layers $LAYERS --gaps $GAPS --writes /vol/q36/phase0/writes/acts_4k.parquet --out /vol/q36/phase0/jvecs.parquet" jvecs
  waitn q36/phase0 "jvecs.parquet$" 1 || exit 1
fi
# 3. skip-lens rollouts, batch 1 (examples ASAP), then score
SPECS1="h_L62;Jh_L42;Jh_L24;Jd_24_42;Jd_30_42;Jh_L30;Jh_L36;Jh_L48;Jh_L54;Jh_L60;Jd_36_48;Jd_42_54;Jd_42_60;Jd_30_54"
C="--data /vol/q36/phase0/jvecs.parquet --n-rows 2048 --adapter /vol/q36/ckpt/skiplens_repeat --prompt skiplens --max-tokens 20 --min-tokens 3 --out-dir /vol/q36/phase0/skiplens"
if [ "$(nfiles q36/phase0/skiplens 'Jd_30_54.parquet$')" -lt 1 ]; then
  wait_gpu || exit 1; run vllm rollout_vllm.py "$C --specs '$SPECS1' --n-samples 3" skiplens1
fi
waitn q36/phase0/skiplens "Jd_24_42.parquet$" 1 || exit 1; waitn q36/phase0/skiplens "Jd_30_42.parquet$" 1 || exit 1
# early examples + metrics on what exists (h_L62, Jh_L42, Jh_L24, Jd_24_42, Jd_30_42)
wait_gpu || exit 1; run hf phase0b_score.py "--acts /vol/q36/phase0/acts_4k.parquet --jvecs /vol/q36/phase0/jvecs.parquet --rollouts-dir /vol/q36/phase0/skiplens --olens-dir /vol/q36/phase0/rollouts --out /vol/q36/phase0/phase0b_metrics_early.json --examples-out /vol/q36/phase0/phase0b_examples_early.json" score0b_early
waitn q36/phase0 "phase0b_examples_early.json" 1 || exit 1; log "EARLY EXAMPLES READY"
waitn q36/phase0/skiplens "Jd_30_54.parquet$" 1 || exit 1
# 4. batch 2: single-map / centred variants + pooled writes (greedy + 1 sample)
SPECS2=""; for g in 24_42 30_42 36_48 42_54 42_60 30_54; do SPECS2+="Jdj_$g;Jdc_$g;JA_$g;JM_$g;"; done
if [ "$(nfiles q36/phase0/skiplens 'JM_30_54.parquet$')" -lt 1 ]; then
  wait_gpu || exit 1; run vllm rollout_vllm.py "$C --specs '${SPECS2%;}' --n-samples 1" skiplens2
fi
waitn q36/phase0/skiplens "JM_30_54.parquet$" 1 || exit 1
wait_gpu || exit 1; run hf phase0b_score.py "--acts /vol/q36/phase0/acts_4k.parquet --jvecs /vol/q36/phase0/jvecs.parquet --rollouts-dir /vol/q36/phase0/skiplens --olens-dir /vol/q36/phase0/rollouts --out /vol/q36/phase0/phase0b_metrics.json --examples-out /vol/q36/phase0/phase0b_examples.json --n-examples 12" score0b
waitn q36/phase0 "phase0b_metrics.json" 1 || exit 1
log "PHASE 0B CHAIN DONE"
