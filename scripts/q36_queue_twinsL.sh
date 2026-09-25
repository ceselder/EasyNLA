#!/usr/bin/env bash
# Orchestrator 13:50: re-score the saved critic checkpoints on the LARGE distinct-position twin set (>= 1024 pairs, <= 1 per position, v1 val shards 29-31 + v3 val shard 32; exact Heun 32 + FM view;
# CIs bootstrapped by position). Waits until no eval worker launched BEFORE the code epoch (13:55) is live (they lack --fixed-from-twins), then enqueues prio-1 specs for critic v5's saves and prio-3
# comparison specs for v4 / v3c / v1b. The gate watcher (restarted 13:55) adds the same spec automatically for every NEW save.
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; source /home/celeste/nlt-q36-logs/gpu_lib.sh
LOGD=/home/celeste/nlt-q36-logs; log(){ echo "[twinsL] $(date -u +%H:%M) $*"; }; EPOCH=${EPOCH:-13:55}
TW='craft_twins:/vol/q36/text/v1/val/twins__*.parquet;/vol/q36/text/v3/val/twins__*.parquet'
for i in $(seq 1 120); do live=0; for a in $(awk -v t="$EPOCH" '$1 ~ /^ap-/ && $3 ~ /^evalq_w/ && $4 < t {print $1}' $LOGD/gpu_ledger.txt); do app_live "$a" && live=$((live + 1)); done; [ $live -eq 0 ] && break; [ $((i % 5)) -eq 0 ] && log "waiting: $live pre-epoch worker(s) still live"; sleep 60; done
log "no pre-epoch worker live -> enqueueing"
q(){ QUEUE_MODE=1 enqueue_eval "--data-dir /vol/q36/data --ckpt $1 --out /vol/q36/results/$2.json --sets '' --twins '$TW' --fixed-from-twins --n-fixed 1024 --twins-n 1024 --ode-steps 32 --skip-samples --skip-sw" "$3" "$4" | tee -a $LOGD/watch_critic_${5}.out | cut -c1-140; }
for st in 000500 000849 001000 001166 001483; do [ "$(timeout 120 modal volume ls nlt q36/critic/v5 2>/dev/null | grep -c ckpt_step$st.pt)" -ge 1 ] && q /vol/q36/critic/v5/ckpt_step$st.pt bits_v5_step${st}_twinsL v5eval_step${st}_twinsL 1 v5; done
q /vol/q36/critic/v4/ckpt_step002000.pt bits_v4_step002000_twinsL v4eval_step002000_twinsL 3 v4
q /vol/q36/critic/v4/ckpt_step001791.pt bits_v4_step001791_twinsL v4eval_step001791_twinsL 3 v4
q /vol/q36/critic/v4/ckpt_step000500.pt bits_v4_step000500_twinsL v4eval_step000500_twinsL 3 v4
q /vol/q36/critic/v3c/ckpt_step001791.pt bits_v3c_step001791_twinsL v3ceval_step001791_twinsL 3 v3c
q /vol/q36/critic/v1b/ckpt_step000500.pt bits_v1bs500_twinsL v1bs500_twinsL 3 v4
q /vol/q36/critic/v1b/ckpt_step001500.pt bits_v1bs1500_twinsL v1bs1500_twinsL 3 v4
log "TWINSL QUEUED"
