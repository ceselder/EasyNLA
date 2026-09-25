#!/usr/bin/env bash
# waits for the four bits_v1_*.json (chain1) -> pulls them + critic eval into the report data dir -> plots + HTML
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; D=/home/celeste/shared/reports/nlt-27b-olens/data
log(){ echo "[pullv1] $(date -u +%H:%M) $*"; }
for i in $(seq 1 400); do n=$(timeout 120 modal volume ls nlt q36/results 2>/dev/null | grep -cE "bits_v1_(main|components|verbalizer|describers)\.json"); [ $((i % 5)) -eq 0 ] && log "bits_v1 files: $n/4"; [ "$n" -ge 4 ] && break; sleep 120; done
for f in main components verbalizer describers; do timeout 300 modal volume get nlt q36/results/bits_v1_$f.json $D/bits_v1_$f.json --force >/dev/null 2>&1 && log "pulled bits_v1_$f.json"; done
timeout 120 modal volume get nlt q36/critic/v1/eval_latest.json $D/critic_v1_eval_latest.json --force >/dev/null 2>&1
systemd-run --user --scope -q -p MemoryMax=2G python3 scripts/plot_nlt_q36_phase1.py --tag v1 2>&1 | tail -30 | tee /home/celeste/nlt-q36-logs/plot_phase1_v1.out
cd /home/celeste/shared/reports/nlt-27b-olens && systemd-run --user --scope -q -p MemoryMax=1G python3 build_html.py && log "REPORT BUILT"
log "PULLV1 DONE"
