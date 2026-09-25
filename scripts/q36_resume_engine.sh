#!/usr/bin/env bash
source /home/celeste/nlt-q36-logs/gpu_lib.sh 2>/dev/null
# resume a paused harvest engine slot when a given app is no longer live: resume_engine.sh <slot> <app-id>
LOGD=/home/celeste/nlt-q36-logs; slot=$1; app=$2; cd /home/celeste/nlt; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET
miss=0; for i in $(seq 1 600); do L=$(app_list); if [ -z "$L" ]; then sleep 60; continue; fi; if echo "$L" | grep -vE "stopped|stopping" | grep -q "$app"; then miss=0; else miss=$((miss + 1)); [ $miss -ge 3 ] && break; fi; sleep 300; done     # 3 consecutive absences (an empty list is a transient read)
rm -f $LOGD/harvest_pause_$slot; echo "[resume$slot] $(date -u +%H:%M) engine $slot unpaused ($app ended)"
