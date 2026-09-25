#!/usr/bin/env bash
# RL v6 GATE (orchestrator 15:50): launch RL v6 only if RL v5's step-40 EXACT verdict is flat under the FROZEN judge (gain within 1 sem); never stop v5 early.
# v6 = v5's recipe + t-weighted FM reward on the judge's claim-sensitive bands + per-bullet leave-one-out credit + groups of 16. Also requires the bullet-credit smoke test to have
# the right sign (P(true slot > swapped slot) > .5 for twin_new on the v6 grid) - else it logs and waits for a human.
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; source /home/celeste/nlt-q36-logs/gpu_lib.sh
LOGD=/home/celeste/nlt-q36-logs; D=/home/celeste/shared/reports/nlt-27b-olens/data; log(){ echo "[rl6gate] $(date -u +%H:%M) $*"; }
JUDGE=${JUDGE:-/vol/q36/critic/v5/ckpt_step000500.pt}
for i in $(seq 1 720); do
  V=$(python3 -c "
import json, sys
try: d=json.load(open('$D/rl_rl_v5_exact.json')); g=d.get('gains_at_last') or {}; print(g.get('step', 0), g.get('frozen', {}).get('significant'), round(g.get('frozen', {}).get('gain', 0), 2), round(g.get('frozen', {}).get('sem', 0), 2))
except Exception: print(0, None, 0, 0)")
  set -- $V; st=$1; sig=$2; gain=$3; sem=$4
  if [ "$st" -ge 40 ]; then break; fi
  [ $((i % 10)) -eq 0 ] && log "waiting for RL v5's exact step-40 verdict (last step $st)"; sleep 60
done
log "RL v5 exact verdict at step $st: frozen gain $gain ± $sem, significant=$sig"
if [ "$sig" != "False" ]; then log "frozen-judge gain is NOT flat -> RL v6 not launched (RL v5 is moving exact content); exit"; exit 0; fi
SM=$D/bullet_smoke_v5s500.json; for i in $(seq 1 120); do [ -f $SM ] && break; sleep 60; done
OK=$(python3 -c "
import json
try: d=json.load(open('$SM')); x=d['grids']['v6']['swapped_slot'].get('twin_new', {}); print('yes' if x.get('p_true_bullet_gain_gt_swapped', 0) > 0.5 and d['grids']['v6']['true_bullets']['gain_mean'] > 0 else 'no', x.get('p_true_bullet_gain_gt_swapped'), d['grids']['v6']['true_bullets']['gain_mean'])
except Exception as e: print('unknown', e)")
log "bullet-credit smoke: $OK"; case "$OK" in yes*) ;; *) log "smoke test not passed/unknown -> not launching v6 automatically; human decision"; exit 0;; esac
# weights from the judge's t-band table (P - .5 per band at t .5/.7/.9 for twin_new + twin_shift averaged), fallback .28/.37/.36
W=$(python3 -c "
import json
try:
    v=json.load(open('$D/bits_v5_step000500_twinsLt.json'))['twins']['craft_twins']['variants']; ts=['0.5','0.7','0.9']
    w=[max(1e-3, sum(v[k]['fm_by_t'][t]['p_true_better'] - 0.5 for k in ('twin_new','twin_shift')) / 2) for t in ts]; s=sum(w); print(','.join(f'{x/s:.3f}' for x in w))
except Exception: print('0.28,0.37,0.36')")
log "t weights for 0.5,0.7,0.9: $W"
rm -f $LOGD/rl_app_rl_v6.txt $LOGD/.rl_rl_v6_stopped
systemd-run --user --scope -q -p MemoryMax=1G --setenv=RL_TAG=rl_v6 --setenv=CRITIC_CK=$JUDGE --setenv=FROZEN_CK=$JUDGE --setenv=GATE_DIR=q36/critic/v5 --setenv=PRIO=1 --setenv=EXTRA_RL_ARGS="--group 16 --batch 8 --t-grid 0.5,0.7,0.9 --t-weights $W --eps-draws 4 --bullet-beta 1.0 --bullet-lines \"Now present,Shift\" --save-every 20" bash $LOGD/launch_rl_v5.sh >> $LOGD/launch_rl_v6.out 2>&1 &
log "RL v6 launcher started (waits for 4 GPUs; RL v5 keeps running under its own rules)"
for i in $(seq 1 720); do [ -s $LOGD/rl_app_rl_v6.txt ] && break; sleep 30; done; [ -s $LOGD/rl_app_rl_v6.txt ] || { log "no RL v6 app after 6 h"; exit 1; }
A=$(cat $LOGD/rl_app_rl_v6.txt); log "RL v6 app $A"
systemd-run --user --scope -q -p MemoryMax=1G --setenv=RL_TAG=rl_v6 --setenv=APPFILE=$LOGD/rl_app_rl_v6.txt bash $LOGD/watch_rl_v3.sh >> $LOGD/watch_rl_v6.out 2>&1 &
systemd-run --user --scope -q -p MemoryMax=1G --setenv=RL_TAG=rl_v6 --setenv=FROZEN=$JUDGE --setenv=APPFILE=$LOGD/rl_app_rl_v6.txt bash $LOGD/rl_exact_evals.sh >> $LOGD/rl_exact_evals_v6.out 2>&1 &
systemd-run --user --scope -q -p MemoryMax=1G --setenv=RL_TAG=rl_v6 --setenv=FROZEN=$JUDGE --setenv=APPFILE=$LOGD/rl_app_rl_v6.txt bash $LOGD/rl_policy_twins.sh >> $LOGD/rl_policy_twins_v6.out 2>&1 &
log "RL6 GATE DONE: watcher + exact chain + policy twins armed for $A"
