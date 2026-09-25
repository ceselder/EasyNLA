#!/usr/bin/env bash
# mechanics smoke of rl_verbalizer.py (random policy LoRA + random critic): waits for GPU headroom (<= MAXG total across nlt-q36 apps)
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; MAXG=${MAXG:-8}
log(){ echo "[rlsmoke] $(date -u +%H:%M) $*"; }
gpus_in_use(){ timeout 90 modal app list 2>/dev/null | grep "nlt-q36" | grep -vE "stopped" | awk -F'│' '{gsub(/ /,"",$5); s+=$5} END {print s+0}'; }
for i in $(seq 1 600); do g=$(gpus_in_use); [ $((g + 1)) -le $MAXG ] && break; [ $((i % 5)) -eq 0 ] && log "waiting for headroom ($g in use)"; sleep 120; done
log "launching smoke ($g in use)"
timeout 900 modal run --detach scripts/modal_nlt_q36.py --task hf --gpus 1 --script rl_verbalizer.py --args "--data-dir /vol/q36/data --policy none --critic none --out /vol/q36/rl/_smoke --band 24,30,36,42,48,54 --max-train-pos 12000 --steps 3 --batch 4 --group 4 --n-tok 32 --heldout 16 --eval-every 3 --save-every 3 --gen-chunk 16 --bwd-chunk 8 --no-wandb" 2>&1 | grep -E "SPAWNED|modal.com/apps|rror" | sed "s/^/[rlsmoke] /" | tee -a /home/celeste/nlt-q36-logs/apps.txt
