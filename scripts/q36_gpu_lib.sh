# shared GPU accounting for the nlt-q36 chains: a ledger (app id -> GPUs PER TASK; hf-many = 1, torchrun RL/SFT = nproc) x live task count.
# A failed / empty `modal app list` (transient) returns 99 so callers keep waiting instead of launching into an unknown state.
LEDGER=/home/celeste/nlt-q36-logs/gpu_ledger.txt; touch $LEDGER
live_apps(){ timeout 90 modal app list 2>/dev/null | grep "nlt-q36" | grep -vE "stopped|stopping" | awk -F'│' '{gsub(/ /,"",$2); gsub(/ /,"",$5); print $2, $5}'; }
gpus_in_use(){ raw=$(timeout 90 modal app list 2>/dev/null); [ -z "$raw" ] && { echo 99; return; }; echo "$raw" | grep "nlt-q36" | grep -vE "stopped|stopping" | awk -F'│' '{gsub(/ /,"",$2); gsub(/ /,"",$5); print $2, $5}' | while read -r id tasks; do g=$(awk -v a="$id" '$1==a {print $2; exit}' $LEDGER); echo $(( tasks * ${g:-1} )); done | awk '{s+=$1} END {print s+0}'; }
ledger_add(){ id=$(echo "$1" | grep -oE "ap-[A-Za-z0-9]+" | head -1); [ -n "$id" ] && echo "$id $2 $3 $(date -u +%H:%M)" >> $LEDGER; [ -n "$id" ] && for i in 1 2 3 4 5 6 7 8 9 10 11 12; do timeout 60 modal app list 2>/dev/null | grep -q "$id" && break; sleep 5; done; release_gpu_lock; }   # hold the lock until the new app is listed (a spawned app takes seconds to appear; without this two waiters over-launch)
LOCK=/home/celeste/nlt-q36-logs/gpu_launch.lock
harvest_active(){ for p_ in 0 1 2 3 4 5; do a_=$(cat /home/celeste/nlt-q36-logs/harvest_app_$p_.txt 2>/dev/null || true); [ -n "$a_" ] && timeout 90 modal app list 2>/dev/null | grep -vE "stopped|stopping" | grep -q "$a_" && return 0; done; pgrep -f "bash .*harvest\.sh" >/dev/null 2>&1 && [ ! -f /home/celeste/nlt-q36-logs/.harvest_done ]; }
cap_now(){ if harvest_active; then echo 12; else echo 8; fi; }   # orchestrator 08:05: cap 12 until the HARVEST is done (6 engines also during RL v4); 8 after
wait_gpu(){ need=${1:-1}; cap=${2:-$(cap_now)}; exec 9>$LOCK; flock 9; for i in $(seq 1 900); do g=$(gpus_in_use); [ $((g + need)) -le $cap ] && { echo "[gpu] $(date -u +%H:%M) headroom: $g in use (cap $cap), launching $need"; return 0; }; [ $((i % 5)) -eq 0 ] && echo "[gpu] $(date -u +%H:%M) waiting: $g in use (cap $cap), need $need"; flock -u 9; sleep 120; flock 9; done; flock -u 9; return 1; }
# release_gpu_lock: call right AFTER the launch has been ledgered (the lock is held from a successful wait_gpu until then, so a second waiter sees the new app in the ledger)
release_gpu_lock(){ flock -u 9 2>/dev/null || true; }
# completeness of an eval_bits result: the JSON is rewritten after every set, `elapsed_min` is only present once the whole job finished
bits_complete(){ timeout 180 modal volume get nlt "q36/results/$1.json" "/tmp/q36_chk_$1.json" --force >/dev/null 2>&1 && grep -q '"elapsed_min"' "/tmp/q36_chk_$1.json"; }
wait_bits(){ # wait_bits name1 name2 ... -> returns when every result is complete
  for i in $(seq 1 600); do ok=0; for n in "$@"; do bits_complete "$n" && ok=$((ok + 1)); done; [ "$ok" -ge $# ] && return 0; [ $((i % 5)) -eq 0 ] && echo "[bits-wait] $(date -u +%H:%M) complete: $ok/$#"; sleep 180; done; return 1; }
