# shared GPU accounting for the nlt-q36 chains: a ledger (app id -> GPUs) so multi-GPU apps count correctly; task count as the fallback.
# A failed / empty `modal app list` (transient) returns 99 so callers keep waiting instead of launching into an unknown state.
LEDGER=/home/celeste/nlt-q36-logs/gpu_ledger.txt; touch $LEDGER
live_apps(){ timeout 90 modal app list 2>/dev/null | grep "nlt-q36" | grep -vE "stopped|stopping" | awk -F'│' '{gsub(/ /,"",$2); gsub(/ /,"",$5); print $2, $5}'; }
gpus_in_use(){ raw=$(timeout 90 modal app list 2>/dev/null); [ -z "$raw" ] && { echo 99; return; }; echo "$raw" | grep "nlt-q36" | grep -vE "stopped|stopping" | awk -F'│' '{gsub(/ /,"",$2); gsub(/ /,"",$5); print $2, $5}' | while read -r id tasks; do g=$(awk -v a="$id" '$1==a {print $2; exit}' $LEDGER); echo "${g:-$tasks}"; done | awk '{s+=$1} END {print s+0}'; }
ledger_add(){ id=$(echo "$1" | grep -oE "ap-[A-Za-z0-9]+" | head -1); [ -n "$id" ] && echo "$id $2 $3 $(date -u +%H:%M)" >> $LEDGER; }
wait_gpu(){ need=${1:-1}; cap=${2:-8}; for i in $(seq 1 900); do g=$(gpus_in_use); [ $((g + need)) -le $cap ] && { echo "[gpu] $(date -u +%H:%M) headroom: $g in use (cap $cap), launching $need x ${GT:-H100}"; return 0; }; [ $((i % 5)) -eq 0 ] && echo "[gpu] $(date -u +%H:%M) waiting ($g in use, cap $cap, need $need)"; sleep 120; done; return 1; }
