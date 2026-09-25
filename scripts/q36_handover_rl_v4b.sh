#!/usr/bin/env bash
# orchestrator 09:28: once RL v4's eval@10 is recorded, stop RL v4 (stale replay from text/v1 = re-epoching inside RL) and launch RL v4b with fresh replay; keep v4's evals 0-10 as the "stale replay" comparison.
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; source /home/celeste/nlt-q36-logs/gpu_lib.sh
LOGD=/home/celeste/nlt-q36-logs; D=/home/celeste/shared/reports/nlt-27b-olens/data; log(){ echo "[handover4b] $(date -u +%H:%M) $*"; }
for i in $(seq 1 120); do [ -f $D/rl_rl_v4/eval_0010.json ] && break; sleep 60; done; [ -f $D/rl_rl_v4/eval_0010.json ] || { log "no eval_0010 after 2 h; abort"; exit 1; }
python3 -c "
import json; e=json.load(open('$D/rl_rl_v4/eval_0010.json')); print('RL v4 eval@10:', {k: (round(v,3) if isinstance(v,float) else v) for k,v in e.items() if k!='examples'})"
A4=$(grep -oE "ap-[A-Za-z0-9]+" $LOGD/rl_app_rl_v4.txt | head -1); timeout 120 modal app stop -y $A4 >/dev/null 2>&1; sed -i "s/^$A4 /# $A4 (stopped $(date -u +%H:%M): stale replay -> RL v4b) /" $LOGD/gpu_ledger.txt; log "RL v4 $A4 stopped"
bash $LOGD/killchain.sh watch_rl_v3.sh >/dev/null 2>&1 || true; sleep 20
CRITIC_CK=/vol/q36/critic/v1b/ckpt_step000500.pt FROZEN_CK=/vol/q36/critic/v1b/ckpt_step000500.pt PRIO=1 bash $LOGD/launch_rl_v4b.sh 2>&1 | sed "s/^/[handover4b] /"
(systemd-run --user --scope -q -p MemoryMax=1G --setenv=RL_TAG=rl_v4b --setenv=APPFILE=$LOGD/rl_app_rl_v4b.txt bash $LOGD/watch_rl_v3.sh >> $LOGD/watch_rl_v4b.out 2>&1 &)
log "HANDOVER 4b DONE: $(cat $LOGD/rl_app_rl_v4b.txt 2>/dev/null)"
