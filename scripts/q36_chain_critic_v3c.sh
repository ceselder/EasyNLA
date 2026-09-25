#!/usr/bin/env bash
# CRITIC v3c (orchestrator 08:15 + 09:00: alongside v4 in the FIRST free slot, pausing a harvest engine if needed): v4's stage-1 config and pools (v1b recipe, no unconditional phase, one pass) + the plain margin HINGE anchor
# relu(FM_own - FM_other + m) with m small (0.02 per-dim MSE vs a natural own-vs-dm gap of ~0.1), weight 0.1 with a 300-step warm-up, and the hard reconstruction GUARD: held-out cos(E[u_j|z],u_j) and
# held-out FM loss with text within 5% of critic v1b's at the matched step (ref_evals.json), auto-stop otherwise. FM_other is logged separately (rising while FM_own does not fall = the shortcut).
# Same pre-registered pass criterion (twin_shift or twin_new P >= 0.60 at content >= 25) + the guard. v3c vs v4 isolates the anchor.
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; source /home/celeste/nlt-q36-logs/gpu_lib.sh
BAND=12,16,20,24,28,30,32,36,40,42,44,48,52,54,56,60; LOGD=/home/celeste/nlt-q36-logs; D=/home/celeste/shared/reports/nlt-27b-olens/data
log(){ echo "[v3c] $(date -u +%H:%M) $*"; }
run(){ out=$(NLT_Q36_GPU=H100 timeout 900 modal run --detach scripts/modal_nlt_q36.py --task hf --gpus 1 --script "$1" --args "$2" 2>&1 | grep -E "SPAWNED|modal.com/apps|Error|rror:"); echo "$out" | sed "s/^/[$3] /" | tee -a $LOGD/apps.txt; ledger_add "$out" 1 "$3"; }
for i in $(seq 1 600); do grep -q "^\[critic_v4_s1\] SPAWNED" $LOGD/apps.txt && [ -f $D/critic_v4_stage1.json ] && break; [ $((i % 6)) -eq 0 ] && log "waiting for critic v4 stage 1 to launch"; sleep 300; done
A4=$(grep -E "^\[critic_v4_s1\] https" $LOGD/apps.txt | tail -n 1 | grep -oE "ap-[A-Za-z0-9]+")
# same pools / val sets as v4 stage 1 (read back from its launch line)
LINE=$(grep -E "^\[critic_v4_s1\] SPAWNED" $LOGD/apps.txt | tail -n 1); SPEC=$(timeout 120 modal app logs $A4 2>&1 | grep -oE "\-\-pools '[^']+'" | head -n 1 | sed -E "s/^--pools '//; s/'$//"); VALS=$(timeout 120 modal app logs $A4 2>&1 | grep -oE "\-\-val-sets '[^']+'" | head -n 1 | sed -E "s/^--val-sets '//; s/'$//"); STEPS=$(timeout 120 modal app logs $A4 2>&1 | grep -oE "\-\-steps [0-9]+" | head -n 1 | awk '{print $2}')
[ -z "$SPEC" ] && { log "could not read v4's pools from its log; abort"; exit 1; }
log "v3c = v4 stage-1 pools ($(echo "$SPEC" | tr ',' '\n' | wc -l) pools), steps $STEPS, + hinge anchor w 0.1 m 0.02 warm-up 300, guard 5% vs v1b"
# orchestrator 09:00: v3c alongside v4 in the FIRST free slot; if the cap makes that impossible, pause one more harvest engine (highest running slot) rather than delay v3c
if [ $(( $(gpus_in_use) + 1 )) -gt $(cap_now) ]; then
  for p_ in 4 3 2 1 0; do a_=$(cat $LOGD/harvest_app_$p_.txt 2>/dev/null || true); [ -n "$a_" ] || continue; timeout 90 modal app list 2>/dev/null | grep -vE "stopped|stopping" | grep -q "$a_" || continue
    touch $LOGD/harvest_pause_$p_; timeout 120 modal app stop -y $a_ >/dev/null 2>&1; sed -i "s/^$a_ /# $a_ (paused $(date -u +%H:%M) for critic v3c) /" $LOGD/gpu_ledger.txt; log "paused harvest engine $p_ ($a_) for v3c"; PAUSED=$p_; sleep 20; break; done
fi
PRIO=1 wait_gpu 1 || exit 1
run train_critic.py "--data-dir /vol/q36/data --out /vol/q36/critic/v3c --tag critic_v3c --pools '$SPEC' --val-sets '$VALS' --band $BAND --width 1536 --depth 16 --heads 16 --param v --uncond-steps 0 --uncond-frac 0.10 --steps $STEPS --keep-every 500 --max-passes 1 --batch 1024 --micro-batch 128 --eval-every 500 --eval-n 256 --spot-exact-n 64 --spot-ode-steps 16 --max-hours 8.0 --anchor 0.1 --anchor-mode hinge --anchor-margin 0.02 --anchor-frac 0.5 --anchor-warmup 300 --guard-ref /vol/q36/critic/v1b/ref_evals.json --guard-tol 0.05" critic_v3c
A3C=$(grep -E "^\[critic_v3c\] https" $LOGD/apps.txt | tail -n 1 | grep -oE "ap-[A-Za-z0-9]+"); log "critic v3c launched: $A3C"
[ -n "${PAUSED:-}" ] && (systemd-run --user --scope -q -p MemoryMax=512M bash $LOGD/resume_engine.sh $PAUSED $A3C >> $LOGD/resume_engine.out 2>&1 &)
(TAG=v3c MINSTEP=500 systemd-run --user --scope -q -p MemoryMax=1G --setenv=TAG=v3c --setenv=MINSTEP=500 bash $LOGD/watch_critic_v3.sh >> $LOGD/watch_critic_v3c.out 2>&1 &)
