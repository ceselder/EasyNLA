#!/usr/bin/env bash
# RL v6 IN PARALLEL with RL v5 (orchestrator 16:40): judge critic v5@500 (frozen + co-trained start), judge2 v4@2000 for the exact view / policy twins, per-bullet credit on Now-present + Shift,
# t-weighted FM reward on the judge's claim-sensitive bands, groups of 16. Harvest v5 engines paused (harvest5_pause_*) until v6 holds its 4 GPUs; eval workers 2.
set -uo pipefail; LOGD=/home/celeste/nlt-q36-logs; J=/vol/q36/critic/v5/ckpt_step000500.pt; J2=/vol/q36/critic/v4/ckpt_step002000.pt; log(){ echo "[rl6] $(date -u +%H:%M) $*"; }
rm -f $LOGD/rl_app_rl_v6.txt $LOGD/.rl_rl_v6_stopped
systemd-run --user --scope -q -p MemoryMax=1G --setenv=RL_TAG=rl_v6 --setenv=CRITIC_CK=$J --setenv=FROZEN_CK=$J --setenv=GATE_DIR=q36/critic/v5 --setenv=PRIO=1 --setenv=SAVE_EVERY=20 --setenv=EXTRA_RL_ARGS="--group 16 --batch 8 --t-grid 0.5,0.7,0.9 --t-weights 0.268,0.361,0.371 --eps-draws 4 --bullet-beta 1.0 --bullet-lines \"Now present,Shift\"" bash $LOGD/launch_rl_v5.sh >> $LOGD/launch_rl_v6.out 2>&1 &
log "RL v6 launcher started (prio 1, waits for 4 free GPUs; t-weights 0.268,0.361,0.371)"
for i in $(seq 1 720); do [ -s $LOGD/rl_app_rl_v6.txt ] && break; sleep 30; done; [ -s $LOGD/rl_app_rl_v6.txt ] || { log "no RL v6 app after 6 h"; exit 1; }
A=$(cat $LOGD/rl_app_rl_v6.txt); log "RL v6 app $A"
systemd-run --user --scope -q -p MemoryMax=1G --setenv=RL_TAG=rl_v6 --setenv=APPFILE=$LOGD/rl_app_rl_v6.txt bash $LOGD/watch_rl_v3.sh >> $LOGD/watch_rl_v6.out 2>&1 &
systemd-run --user --scope -q -p MemoryMax=1G --setenv=RL_TAG=rl_v6 --setenv=FROZEN=$J --setenv=JUDGE2=$J2 --setenv=APPFILE=$LOGD/rl_app_rl_v6.txt bash $LOGD/rl_exact_evals.sh >> $LOGD/rl_exact_evals_v6.out 2>&1 &
systemd-run --user --scope -q -p MemoryMax=1G --setenv=RL_TAG=rl_v6 --setenv=FROZEN=$J --setenv=JUDGE2=$J2 --setenv=EVERY=20 --setenv=APPFILE=$LOGD/rl_app_rl_v6.txt bash $LOGD/rl_policy_twins.sh >> $LOGD/rl_policy_twins_v6.out 2>&1 &
touch $LOGD/.rl_rl_v6_stopped      # no content-based auto-stop for v6 either (amendment #2): the twin watcher + FM guards decide
sleep 120; for p in 0 1 2 3 4 5; do rm -f $LOGD/harvest5_pause_$p; done; log "harvest v5 engines unpaused (they fill remaining slots at prio 3)"
log "RL6 DONE: watcher + exact chain (monitor only) + policy twins armed for $A"
