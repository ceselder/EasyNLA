#!/usr/bin/env bash
# resume a paused harvest engine slot when a given app is no longer live: resume_engine.sh <slot> <app-id>
LOGD=/home/celeste/nlt-q36-logs; slot=$1; app=$2; cd /home/celeste/nlt; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET
for i in $(seq 1 600); do timeout 90 modal app list 2>/dev/null | grep -vE "stopped|stopping" | grep -q "$app" || break; sleep 300; done
rm -f $LOGD/harvest_pause_$slot; echo "[resume$slot] $(date -u +%H:%M) engine $slot unpaused ($app ended)"
