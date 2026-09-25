# shared GPU accounting for the nlt-q36 chains: a ledger (app id -> GPUs PER TASK; hf-many = 1, torchrun RL/SFT = nproc) x live task count.
# A failed / empty `modal app list` (transient) returns 99 so callers keep waiting instead of launching into an unknown state.
LEDGER=/home/celeste/nlt-q36-logs/gpu_ledger.txt; touch $LEDGER
live_apps(){ timeout 90 modal app list 2>/dev/null | grep "nlt-q36" | grep -vE "stopped|stopping" | awk -F'│' '{gsub(/ /,"",$2); gsub(/ /,"",$5); print $2, $5}'; }
gpus_in_use(){ raw=$(timeout 90 modal app list 2>/dev/null); [ -z "$raw" ] && { echo 99; return; }; echo "$raw" | grep "nlt-q36" | grep -vE "stopped|stopping" | awk -F'│' '{gsub(/ /,"",$2); gsub(/ /,"",$5); print $2, $5}' | while read -r id tasks; do g=$(awk -v a="$id" '$1==a {print $2; exit}' $LEDGER); echo $(( tasks * ${g:-1} )); done | awk '{s+=$1} END {print s+0}'; }
ledger_add(){ id=$(echo "$1" | grep -oE "ap-[A-Za-z0-9]+" | head -1); [ -n "$id" ] && echo "$id $2 $3 $(date -u +%H:%M)" >> $LEDGER; }
wait_gpu(){ need=${1:-1}; cap=${2:-8}; for i in $(seq 1 900); do g=$(gpus_in_use); [ $((g + need)) -le $cap ] && { echo "[gpu] $(date -u +%H:%M) headroom: $g in use (cap $cap), launching $need x ${GT:-H100}"; return 0; }; [ $((i % 5)) -eq 0 ] && echo "[gpu] $(date -u +%H:%M) waiting ($g in use, cap $cap, need $need)"; sleep 120; done; return 1; }
# completeness of an eval_bits result: the JSON is rewritten after every set, `elapsed_min` is only present once the whole job finished
bits_complete(){ timeout 180 modal volume get nlt "q36/results/$1.json" "/tmp/q36_chk_$1.json" --force >/dev/null 2>&1 && grep -q '"elapsed_min"' "/tmp/q36_chk_$1.json"; }
wait_bits(){ # wait_bits name1 name2 ... -> returns when every result is complete
  for i in $(seq 1 600); do ok=0; for n in "$@"; do bits_complete "$n" && ok=$((ok + 1)); done; [ "$ok" -ge $# ] && return 0; [ $((i % 5)) -eq 0 ] && echo "[bits-wait] $(date -u +%H:%M) complete: $ok/$#"; sleep 180; done; return 1; }
