# shared GPU accounting for the nlt-q36 chains: a ledger (app id -> GPUs) so multi-GPU apps count correctly; task count as the fallback
LEDGER=/home/celeste/nlt-q36-logs/gpu_ledger.txt; touch $LEDGER
live_apps(){ timeout 90 modal app list 2>/dev/null | grep "nlt-q36" | grep -vE "stopped|stopping" | awk -F'│' '{gsub(/ /,"",$2); gsub(/ /,"",$5); print $2, $5}'; }
gpus_in_use(){ live_apps | while read -r id tasks; do g=$(awk -v a="$id" '$1==a {print $2; exit}' $LEDGER); echo "${g:-$tasks}"; done | awk '{s+=$1} END {print s+0}'; }
ledger_add(){ id=$(echo "$1" | grep -oE "ap-[A-Za-z0-9]+" | head -1); [ -n "$id" ] && echo "$id $2 $3 $(date -u +%H:%M)" >> $LEDGER; }
