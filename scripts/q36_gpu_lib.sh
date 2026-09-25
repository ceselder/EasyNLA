# shared GPU accounting for the nlt-q36 chains: a ledger (app id -> GPUs PER TASK; hf-many = 1, torchrun RL/SFT = nproc) x live task count.
# A failed / empty `modal app list` (transient) returns 99 so callers keep waiting instead of launching into an unknown state.
LEDGER=/home/celeste/nlt-q36-logs/gpu_ledger.txt; touch $LEDGER
app_list(){ local o; o=$(timeout 90 modal app list 2>/dev/null) || return 1; echo "$o" | grep -q "└" || return 1; echo "$o"; }   # complete listing or nothing: 10:59-11:09 three consecutive truncated/failed reads made two chains declare live apps "ended"
live_apps(){ app_list | grep "nlt-q36" | grep -vE "stopped|stopping" | awk -F'│' '{gsub(/ /,"",$2); gsub(/ /,"",$5); print $2, $5}'; }
gpus_in_use(){ raw=$(app_list); [ -z "$raw" ] && { echo 99; return; }; echo "$raw" | grep "nlt-q36" | grep -vE "stopped|stopping" | awk -F'│' '{gsub(/ /,"",$2); gsub(/ /,"",$5); print $2, $5}' | while read -r id tasks; do g=$(awk -v a="$id" '$1==a {print $2; exit}' $LEDGER); echo $(( tasks * ${g:-1} )); done | awk '{s+=$1} END {print s+0}'; }
ledger_add(){ id=$(echo "$1" | grep -oE "ap-[A-Za-z0-9]+" | head -1); [ -n "$id" ] && echo "$id $2 $3 $(date -u +%H:%M)" >> $LEDGER; [ -n "$id" ] && for i in 1 2 3 4 5 6 7 8 9 10 11 12; do app_list 9>&- | grep -q "$id" && break; sleep 5 9>&-; done; release_gpu_lock; }   # hold the lock until the new app is listed (a spawned app takes seconds to appear; without this two waiters over-launch)
LOCK=/home/celeste/nlt-q36-logs/gpu_launch.lock
harvest_active(){ for p_ in 0 1 2 3 4 5; do a_=$(cat /home/celeste/nlt-q36-logs/harvest_app_$p_.txt 2>/dev/null || true); [ -n "$a_" ] && app_list | grep -vE "stopped|stopping" | grep -q "$a_" && return 0; done; pgrep -f "bash .*harvest\.sh" >/dev/null 2>&1 && [ ! -f /home/celeste/nlt-q36-logs/.harvest_done ]; }
cap_now(){ if harvest_active; then echo 12; else echo 8; fi; }   # orchestrator 08:05: cap 12 until the HARVEST is done (6 engines also during RL v4); 8 after
# PRIORITY (orchestrator 07:57): a waiter with a lower PRIO number goes first. While waiting, each chain advertises $LOGD/gpu_want_<PRIO>_<pid>; a waiter yields whenever a LIVE higher-priority waiter exists.
#   PRIO 0 harvest engines | 1 v3b gate evals | 2 v1b twins+neighbours (RL v4 judge) | 3 v1b train-vs-val | 4 fair describers | 5 v2 LOO/singles | 9 default
PRIO=${PRIO:-9}; WANTDIR=/home/celeste/nlt-q36-logs/gpu_want; mkdir -p $WANTDIR
higher_waiting(){ local f b pr pid; for f in $WANTDIR/want_*; do [ -e "$f" ] || continue; b=$(basename $f); pr=${b#want_}; pr=${pr%%_*}; pid=${b##*_}; kill -0 $pid 2>/dev/null || { rm -f $f; continue; }; [ "$pr" -lt "$PRIO" ] && return 0; done; return 1; }
APPCAP=${APPCAP:-12}; nlt_apps_live(){ app_list | grep -E "nlt-" | grep -vE "stopped|stopping" | wc -l; }   # orchestrator 09:12: <= 12 live nlt-* apps (workspace limit is 100 ephemeral apps, shared)
wait_gpu(){ [ "${QUEUE_MODE:-0}" = 1 ] && return 0; need=${1:-1}; cap=${2:-$(cap_now)}; touch $WANTDIR/want_${PRIO}_$$; trap 'rm -f $WANTDIR/want_${PRIO}_$$' EXIT
  exec 9>$LOCK; flock 9; for i in $(seq 1 900); do g=$(gpus_in_use); cap=${2:-$(cap_now)}
    na=$(nlt_apps_live)
    if [ $((g + need)) -le $cap ] && [ "$na" -lt "$APPCAP" ] && ! higher_waiting; then rm -f $WANTDIR/want_${PRIO}_$$; echo "[gpu] $(date -u +%H:%M) headroom: $g in use (cap $cap), $na live nlt apps (cap $APPCAP), launching $need (prio $PRIO)"; return 0; fi
    [ $((i % 5)) -eq 0 ] && echo "[gpu] $(date -u +%H:%M) waiting: $g in use (cap $cap), need $need, prio $PRIO$(higher_waiting && echo ' (yielding to a higher-priority waiter)')"; flock -u 9; sleep 120 9>&-; flock 9; done; flock -u 9; rm -f $WANTDIR/want_${PRIO}_$$; return 1; }   # 'sleep 9>&-': children must NOT inherit the lock fd (an orphaned sleep kept the lock after its parent died)
# release_gpu_lock: call right AFTER the launch has been ledgered (the lock is held from a successful wait_gpu until then, so a second waiter sees the new app in the ledger)
release_gpu_lock(){ flock -u 9 2>/dev/null || true; }
# completeness of an eval_bits result: the JSON is rewritten after every set, `elapsed_min` is only present once the whole job finished
bits_complete(){ timeout 180 modal volume get nlt "q36/results/$1.json" "/tmp/q36_chk_$1.json" --force >/dev/null 2>&1 && grep -q '"elapsed_min"' "/tmp/q36_chk_$1.json"; }
wait_bits(){ # wait_bits name1 name2 ... -> returns when every result is complete
  for i in $(seq 1 600); do ok=0; for n in "$@"; do bits_complete "$n" && ok=$((ok + 1)); done; [ "$ok" -ge $# ] && return 0; [ $((i % 5)) -eq 0 ] && echo "[bits-wait] $(date -u +%H:%M) complete: $ok/$#"; sleep 180; done; return 1; }

# spawn_retry env VAR=.. timeout 900 modal run --detach ... : retries while Modal refuses the launch ("reached limit of 100 ephemeral apps", transient errors); releases the launch lock while sleeping
spawn_retry(){ local i out; for i in $(seq 1 24); do out=$("$@" 2>&1 9>&- | grep -E "SPAWNED|modal.com/apps|rror|limit"); if echo "$out" | grep -q SPAWNED; then echo "$out"; return 0; fi
  echo "[spawn] $(date -u +%H:%M) launch refused ($(echo "$out" | grep -oE 'reached limit[^│]*|rror[^│]*' | head -1 | cut -c1-80)); retry $i/24 in 5 min" >&2; flock -u 9 2>/dev/null; sleep 300 9>&-; flock 9 2>/dev/null; done; echo "$out"; return 1; }

# EVAL QUEUE (orchestrator 09:12): enqueue_eval "<eval_bits args>" <label> [prio] [script] -> one spec file on the volume; eval_workers.sh keeps <= EVAL_WORKERS long-lived worker apps (task evalq) draining it in priority order.
EVALQ_LOG=/home/celeste/nlt-q36-logs/evalq.txt
enqueue_eval(){ local args="$1" label="$2" prio="${3:-9}" script="${4:-eval_bits.py}" f; f=/tmp/q36_evalq_${prio}_$(date -u +%Y%m%d%H%M%S)_$$_${label}.json
  python3 -c "import json,sys; json.dump({'label': sys.argv[2], 'args': sys.argv[1], 'prio': int(sys.argv[3]), 'script': sys.argv[4]}, open(sys.argv[5], 'w'))" "$args" "$label" "$prio" "$script" "$f"
  if timeout 180 modal volume put nlt "$f" "q36/evalq/$(basename $f | sed 's/^q36_evalq_//')" 9>&- >/dev/null 2>&1; then echo "[$label] QUEUED prio $prio $(date -u +%H:%M) $(basename $f)" | tee -a $EVALQ_LOG; else echo "[$label] ENQUEUE FAILED" | tee -a $EVALQ_LOG; return 1; fi; }
