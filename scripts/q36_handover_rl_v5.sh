#!/usr/bin/env bash
# RL v5 (14:22): launch with the first judge that passes the amended claim test on the large twin set (critic v5 @849), then arm the FM-view watcher (collusion / length rules) and the
# exact-view eval chain (frozen judge = v5 @849, other-lineage judge = v2 step 3000; prio-0 specs; pre-registered step-40 rule). The launcher waits for 4 free GPUs (cap 12).
set -uo pipefail; LOGD=/home/celeste/nlt-q36-logs; log(){ echo "[ho5] $(date -u +%H:%M) $*"; }
rm -f $LOGD/rl_app_rl_v5.txt $LOGD/.rl_rl_v5_stopped
systemd-run --user --scope -q -p MemoryMax=1G --setenv=RL_TAG=rl_v5 --setenv=CRITIC_CK=/vol/q36/critic/v5/ckpt_step000500.pt --setenv=FROZEN_CK=/vol/q36/critic/v5/ckpt_step000500.pt --setenv=GATE_DIR=q36/critic/v5 --setenv=PRIO=1 bash $LOGD/launch_rl_v5.sh >> $LOGD/launch_rl_v5.out 2>&1 &
log "launcher started (waits for 4 GPUs)"
for i in $(seq 1 720); do [ -s $LOGD/rl_app_rl_v5.txt ] && break; sleep 30; done; [ -s $LOGD/rl_app_rl_v5.txt ] || { log "no RL app after 6 h; giving up"; exit 1; }
A=$(cat $LOGD/rl_app_rl_v5.txt); log "RL v5 app $A"
systemd-run --user --scope -q -p MemoryMax=1G --setenv=RL_TAG=rl_v5 --setenv=APPFILE=$LOGD/rl_app_rl_v5.txt bash $LOGD/watch_rl_v3.sh >> $LOGD/watch_rl_v5.out 2>&1 &
systemd-run --user --scope -q -p MemoryMax=1G --setenv=RL_TAG=rl_v5 --setenv=FROZEN=/vol/q36/critic/v5/ckpt_step000500.pt --setenv=APPFILE=$LOGD/rl_app_rl_v5.txt bash $LOGD/rl_exact_evals.sh >> $LOGD/rl_exact_evals_v5.out 2>&1 &
log "HANDOVER 5 DONE: watcher + exact-view chain armed for $A"
