#!/usr/bin/env bash
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; source /home/celeste/nlt-q36-logs/gpu_lib.sh; D=/home/celeste/shared/reports/nlt-27b-olens/data
log(){ echo "[pullv2] $(date -u +%H:%M) $*"; }
wait_bits bits_v2_sources bits_v2_loo bits_v2_singles || { log "TIMEOUT"; exit 1; }
for f in sources loo singles; do cp /tmp/q36_chk_bits_v2_$f.json $D/bits_v2_$f.json && log "pulled bits_v2_$f.json"; done
timeout 120 modal volume get nlt q36/critic/v2/eval_latest.json $D/critic_v2_eval_latest.json --force >/dev/null 2>&1
systemd-run --user --scope -q -p MemoryMax=2G python3 scripts/plot_nlt_q36_phase1.py --tag v2 2>&1 | tail -30 | tee /home/celeste/nlt-q36-logs/plot_phase1_v2.out
cd /home/celeste/shared/reports/nlt-27b-olens && systemd-run --user --scope -q -p MemoryMax=1G python3 build_html.py && log "REPORT BUILT"; log "PULLV2 DONE"
