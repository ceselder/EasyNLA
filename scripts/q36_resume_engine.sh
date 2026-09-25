#!/usr/bin/env bash
source /home/celeste/nlt-q36-logs/gpu_lib.sh 2>/dev/null
# resume a paused harvest engine slot when a given app is no longer live: resume_engine.sh <slot> <app-id>
LOGD=/home/celeste/nlt-q36-logs; slot=$1; app=$2; cd /home/celeste/nlt; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET
wait_app_end "$app"     # 3 consecutive definitive not-live JSON reads
rm -f $LOGD/harvest_pause_$slot; echo "[resume$slot] $(date -u +%H:%M) engine $slot unpaused ($app ended)"
