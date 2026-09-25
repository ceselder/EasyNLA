#!/usr/bin/env bash
# every 10 min: re-parse the critic trainers' spot evals (v1, v2, v1b, v3b) -> data/critic_train_curves.json + fig_critic_uncond_ablation, rebuild the report; stops when v1b and v3b have their final evals
set -uo pipefail; cd /home/celeste/nlt; log(){ echo "[curves] $(date -u +%H:%M) $*"; }
for i in $(seq 1 60); do
  out=$(systemd-run --user --scope -q -p MemoryMax=2G python3 scripts/plot_nlt_q36_critic_train_curves.py --tags v1,v2,v1b,v3b,v4,v3c 2>&1 | grep -E "^v1b|^v3b" | tail -n 2 | tr '\n' ';')
  (cd /home/celeste/shared/reports/nlt-27b-olens && systemd-run --user --scope -q -p MemoryMax=1G python3 build_html.py >/dev/null 2>&1); log "$out"
  python3 -c "
import json,sys; d=json.load(open('/home/celeste/shared/reports/nlt-27b-olens/data/critic_train_curves.json'))
ok=all((d.get(t,{}).get('evals') or [{}])[-1].get('step',0)>=3000 for t in ("v4","v3c")); sys.exit(0 if ok else 1)" && { log "CURVES DONE"; break; }
  sleep 600
done
