#!/usr/bin/env bash
# DRAIN-RECYCLE the eval workers without interrupting a job (11:40): the three live workers (launched 10:43-10:50) predate eval_bits --pair-ids-file (critic v5's train probe) and the
# .code_epoch check. Sequence: (a) move every PENDING spec into q36/evalq/hold/ so idle workers find nothing new; (b) wait until no spec is .running. (current jobs finish; new prio-0 RL
# exact-view specs enqueued meanwhile are still picked up - fine, old code handles them); (c) stop the old worker apps + comment the ledger; (d) move the held specs back; eval_workers.sh
# spawns fresh workers with the current code snapshot.
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; source /home/celeste/nlt-q36-logs/gpu_lib.sh
LOGD=/home/celeste/nlt-q36-logs; log(){ echo "[drain] $(date -u +%H:%M) $*"; }
Q=q36/evalq; H=q36/evalq/hold
ls_q(){ timeout 120 modal volume ls nlt $Q 2>/dev/null | grep -oE "[0-9]_[0-9]+_[0-9]+_[A-Za-z0-9_.-]+\.json"; }
# (a) hold the pending specs
for f in $(ls_q | grep -vE "\.running\.|\.done"); do
  timeout 120 modal volume cp nlt $Q/$f $H/$f >/dev/null 2>&1 && timeout 120 modal volume rm nlt $Q/$f >/dev/null 2>&1 && log "held $f" || log "hold of $f FAILED (left in place)"
done
# (b) wait for the running jobs to finish (prio-0 RL exact specs may arrive and run meanwhile)
for i in $(seq 1 90); do
  for f in $(ls_q | grep -vE "\.running\.|\.done" | grep -vE "^0_"); do timeout 120 modal volume cp nlt $Q/$f $H/$f >/dev/null 2>&1 && timeout 120 modal volume rm nlt $Q/$f >/dev/null 2>&1 && log "held (arrived during the drain) $f"; done   # keep holding what the watchers enqueue meanwhile; prio-0 RL exact specs stay live
  r=$(ls_q | grep -c "\.running\." || true); [ "$r" -eq 0 ] && break; [ $((i % 5)) -eq 0 ] && log "waiting: $r job(s) still running"; sleep 60; done
# (c) stop the old workers
OLD_BEFORE=${OLD_BEFORE:-11:29}     # only workers launched BEFORE the drain started are stale; fresh ones spawned meanwhile (e.g. w1 12:19) already run the current code
for a in $(awk -v t="$OLD_BEFORE" '$1 ~ /^ap-/ && $3 ~ /^evalq_w/ && $4 < t {print $1}' $LOGD/gpu_ledger.txt); do
  app_live "$a" || continue
  timeout 120 modal app stop -y "$a" >/dev/null 2>&1 && log "stopped old worker $a" || log "stop of $a failed"
  sed -i -E "s|^($a 1 evalq_w[^ ]* [0-9:]+)|# \1 (drain-recycled $(date -u +%H:%M): code predates --pair-ids-file)|" $LOGD/gpu_ledger.txt
done
# a job claimed in the last seconds before the stop would be left as .running -> restore it to pending
sleep 15; for r in $(ls_q | grep "\.running\."); do orig=$(echo "$r" | sed -E 's/\.running\.w[0-9]+\.json$/.json/'); timeout 120 modal volume cp nlt $Q/$r $Q/$orig >/dev/null 2>&1 && timeout 120 modal volume rm nlt $Q/$r >/dev/null 2>&1 && log "restored interrupted $orig"; done
# (d) release the held specs
for f in $(timeout 120 modal volume ls nlt $H 2>/dev/null | grep -oE "[0-9]_[0-9]+_[0-9]+_[A-Za-z0-9_.-]+\.json"); do
  timeout 120 modal volume cp nlt $H/$f $Q/$f >/dev/null 2>&1 && timeout 120 modal volume rm nlt $H/$f >/dev/null 2>&1 && log "released $f" || log "release of $f FAILED"
done
log "DRAIN DONE: $(ls_q | grep -vcE '\.running\.|\.done') pending specs; eval_workers.sh spawns fresh workers"
