#!/usr/bin/env bash
# resume the paused harvest engine 5 at 10:00 UTC or when the eval queue is empty (no eval chain waiting), whichever is first (orchestrator 07:57)
LOGD=/home/celeste/nlt-q36-logs; log(){ echo "[resume5] $(date -u +%H:%M) $*"; }
while true; do
  now=$(date -u +%H%M); waiting=$(ls $LOGD/gpu_want/want_[1-5]_* 2>/dev/null | while read f; do pid=${f##*_}; kill -0 $pid 2>/dev/null && echo live; done | wc -l)
  chains=$(pgrep -fc "bash .*(watch_critic_v3|bits_v1b_main|bits_v1b_trainval|fair_describers|bits_v2_calibrated)\.sh")
  if [ "$now" -ge 1000 ] || { [ "$waiting" -eq 0 ] && [ "$chains" -eq 0 ]; }; then rm -f $LOGD/harvest_pause_5; log "engine 5 unpaused (time $now, waiting evals $waiting, eval chains alive $chains)"; break; fi
  sleep 300
done
