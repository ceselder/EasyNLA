#!/usr/bin/env bash
# Critic v3b (orchestrator 07:17) = EXACTLY critic v1b's config (no unconditional phase, uncond-frac 0.10, v1b's 8 pools + weights, same val sets, default seed, same lr schedule, 3000 steps)
# + the ACTIVATION-ANCHORED contrast (--anchor 0.5 --anchor-tau 0.05 --anchor-frac 0.5). v3b vs v1b isolates the anchored contrast. Saves every 500 steps.
# Pre-registered pass rule (same as v3): twin_shift or twin_new P(true > twin) >= 0.60 with craft_full content >= 25 bits at some saved checkpoint (held-out rows, exact Heun 64).
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; source /home/celeste/nlt-q36-logs/gpu_lib.sh
BAND=12,16,20,24,28,30,32,36,40,42,44,48,52,54,56,60; TX=/vol/q36/text/v1; LOGD=/home/celeste/nlt-q36-logs
ANCHOR=${ANCHOR:-0.5}; ATAU=${ATAU:-0.05}; AFRAC=${AFRAC:-0.5}
log(){ echo "[v3b] $(date -u +%H:%M) $*"; }
run(){ out=$(NLT_Q36_GPU=H100 timeout 900 modal run --detach scripts/modal_nlt_q36.py --task hf --gpus 1 --script "$1" --args "$2" 2>&1 | grep -E "SPAWNED|modal.com/apps|Error|rror:"); echo "$out" | sed "s/^/[$3] /" | tee -a $LOGD/apps.txt; ledger_add "$out" 1 "$3"; }
POOLS="craft_full=0.30:$TX/train/craft_full__*.parquet,describer=0.20:$TX/train/describer_sonnet5_A__*.parquet,craft_nojl=0.08:$TX/train/craft_nojl__*.parquet,craft_delta=0.12:$TX/train/craft_delta__*.parquet,craft_newfaded=0.08:$TX/train/craft_newfaded__*.parquet,jlens=0.12:$TX/train/jlens__*.parquet,olens_j=0.10:$TX/train/olens_j__*.parquet,raw_all=0.10:$TX/train/raw_all__*.parquet"
VALS="craft_full:$TX/val/craft_full__*.parquet,describer:$TX/val/describer_sonnet5_A__*.parquet,raw_all:$TX/val/raw_all__*.parquet"
wait_gpu 1 || exit 1
run train_critic.py "--data-dir /vol/q36/data --out /vol/q36/critic/v3b --tag critic_v3b --pools '$POOLS' --val-sets '$VALS' --band $BAND --width 1536 --depth 16 --heads 16 --param v --uncond-steps 0 --uncond-frac 0.10 --steps 3000 --keep-every 500 --batch 1024 --micro-batch 128 --eval-every 500 --eval-n 256 --spot-exact-n 64 --spot-ode-steps 16 --max-hours 3.5 --anchor $ANCHOR --anchor-tau $ATAU --anchor-frac $AFRAC" critic_v3b
log "critic v3b launched"
