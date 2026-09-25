#!/usr/bin/env bash
# Orchestrator 14:45 ("the harvest end -> the positions harvest (go_harvest_v5)"): open the harvest v5 gate the moment the current harvest is complete.
LOGD=/home/celeste/nlt-q36-logs; for i in $(seq 1 720); do [ -f $LOGD/.harvest_done ] && break; sleep 60; done
[ -f $LOGD/.harvest_done ] && { touch $LOGD/.go_harvest_v5; echo "[go5] $(date -u +%H:%M) .harvest_done seen -> .go_harvest_v5 touched (harvest_v5_chain.sh proceeds: extract -> finalize_v5 -> harvest_v5.sh + craft_v5.sh)"; } || echo "[go5] $(date -u +%H:%M) no .harvest_done after 12 h"
