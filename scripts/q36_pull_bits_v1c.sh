#!/usr/bin/env bash
# waits until the four bits_v1 results are COMPLETE (elapsed_min present) -> pulls them + the critic eval -> plots (tag v1) + report
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; source /home/celeste/nlt-q36-logs/gpu_lib.sh; D=/home/celeste/shared/reports/nlt-27b-olens/data
log(){ echo "[pullv1] $(date -u +%H:%M) $*"; }
wait_bits bits_v1_main bits_v1_components bits_v1_verbalizer bits_v1_describers || { log "TIMEOUT"; exit 1; }
for f in main components verbalizer describers; do cp /tmp/q36_chk_bits_v1_$f.json $D/bits_v1_$f.json && log "pulled bits_v1_$f.json"; done
timeout 120 modal volume get nlt q36/critic/v1/eval_latest.json $D/critic_v1_eval_latest.json --force >/dev/null 2>&1
systemd-run --user --scope -q -p MemoryMax=2G python3 scripts/plot_nlt_q36_phase1.py --tag v1 2>&1 | tail -30 | tee /home/celeste/nlt-q36-logs/plot_phase1_v1.out
cd /home/celeste/shared/reports/nlt-27b-olens && systemd-run --user --scope -q -p MemoryMax=1G python3 build_html.py && log "REPORT BUILT"; log "PULLV1 DONE"
