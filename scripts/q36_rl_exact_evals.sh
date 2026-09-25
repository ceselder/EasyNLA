#!/usr/bin/env bash
# orchestrator 10:10: exact-view RL evals. For every saved held-out dump of an RL run (dumps_STEP.parquet written at each eval) at steps 0, 20, 40, ...: one eval_bits job (Heun 32, the run's 128 held-out pairs)
# with sets teacher / policy@step under the FROZEN judge and under an OTHER-lineage critic (v2 step 3000); queued at PRIO 1. Results -> data/rl_<tag>_exact_<judge>_<step>.json -> plot + report.
# Decision rule (orchestrator): the exact view decides; if after ~40 steps the exact-view gain is within one sem under BOTH judges -> stop the run and report "FM-reward RL does not move exact content at this scale".
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; source /home/celeste/nlt-q36-logs/gpu_lib.sh
RL_TAG=${RL_TAG:-rl_v4b}; FROZEN=${FROZEN:-/vol/q36/critic/v1b/ckpt_step000500.pt}; OTHER=${OTHER:-/vol/q36/critic/v2/ckpt_step3000.pt}; EVERY=${EVERY:-20}
LOGD=/home/celeste/nlt-q36-logs; D=/home/celeste/shared/reports/nlt-27b-olens/data; TX=/vol/q36/text/v1; APPFILE=${APPFILE:-$LOGD/rl_app_$RL_TAG.txt}
log(){ echo "[rlexact] $(date -u +%H:%M) $*"; }
declare -A Q
for i in $(seq 1 400); do
  dumps=$(timeout 120 modal volume ls nlt q36/rl/$RL_TAG 2>/dev/null | grep -oE "dumps_[0-9]+\.parquet" | sort -u)
  for d_ in $dumps; do st=${d_#dumps_}; st=${st%.parquet}; stn=$((10#$st)); [ $((stn % EVERY)) -eq 0 ] || continue
    for J in frozen other; do
      out=rl_${RL_TAG}_exact_${J}_$st; [ -n "${Q[$out]:-}" ] && continue; [ -f $D/$out.json ] && { Q[$out]=1; continue; }
      CK=$FROZEN; [ $J = other ] && CK=$OTHER
      enqueue_eval "--data-dir /vol/q36/data --ckpt $CK --out /vol/q36/results/$out.json --sets 'teacher:$TX/val/craft_full__*.parquet,policy:/vol/q36/rl/$RL_TAG/$d_' --n 128 --n-fixed 128 --ode-steps 32 --skip-samples --skip-sw" $out 1 && Q[$out]=1
    done
  done
  new=0
  for f in $(timeout 120 modal volume ls nlt q36/results 2>/dev/null | grep -oE "rl_${RL_TAG}_exact_(frozen|other)_[0-9]+\.json" | sort -u); do
    [ -f $D/$f ] && continue; timeout 120 modal volume get nlt q36/results/$f /tmp/q36_$f --force >/dev/null 2>&1; grep -q '"elapsed_min"' /tmp/q36_$f 2>/dev/null || continue; cp /tmp/q36_$f $D/$f; new=1
    log "pulled $f: $(python3 -c "
import json; d=json.load(open('$D/$f')); s=d['sets']; t=s.get('teacher',{}); p=s.get('policy',{})
print(f\"teacher {t['content_bits']['mean']:.1f}±{t['content_bits']['sem']:.1f} (P {t['p_z_gt_dm']:.2f}) | policy {p['content_bits']['mean']:.1f}±{p['content_bits']['sem']:.1f} (P {p['p_z_gt_dm']:.2f}) | policy PMI {p['pmi_bits']['mean'] if isinstance(p['pmi_bits'],dict) else p['pmi_bits']:+.1f} tok {p['n_tokens_mean']:.0f}\")" 2>&1 | tail -n 1)"
  done
  if [ $new -eq 1 ]; then systemd-run --user --scope -q -p MemoryMax=2G python3 scripts/plot_nlt_q36_rl_exact.py --tag $RL_TAG 2>&1 | grep -E "VERDICT|error|Traceback" | head -3; (cd /home/celeste/shared/reports/nlt-27b-olens && systemd-run --user --scope -q -p MemoryMax=1G python3 build_html.py >/dev/null 2>&1); fi
  A=$(grep -oE "ap-[A-Za-z0-9]+" $APPFILE 2>/dev/null | head -1); [ -n "$A" ] && ! timeout 90 modal app list 2>/dev/null | grep -vE "stopped|stopping" | grep -q "$A" && [ $i -gt 3 ] && { log "RL app $A no longer live; final pass done"; break; }
  sleep 300
done
log "RLEXACT DONE"
