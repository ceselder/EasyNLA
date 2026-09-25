#!/usr/bin/env bash
# POLICY-SIDE LARGE TWINS for an RL run (orchestrator 14:26): every 20 steps (policy adapter saved at step_STEP; step 0 = the SFT policy), (1) dump the policy's greedy output for the
# 1,024 twinsL pairs (dump_verbalizer --pair-ids-file), (2) build twins from those outputs (policy_twins.py: one bullet swapped with another pair at the same (i, j)), (3) score them
# under the FROZEN judge and the OTHER-lineage judge (eval_bits --fixed-from-twins, exact Heun 32 + FM, position-bootstrap CIs) -> data/rl_<tag>_ptwinsL_<judge>_<step>.json.
# All three run on the eval-queue workers (script field); the chain sequences them per step by polling the volume. Headline: does RL against a claim-sensitive reward raise the
# policy's claim-level accuracy (P(true > own twin)), not just content?
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; source /home/celeste/nlt-q36-logs/gpu_lib.sh
RL_TAG=${RL_TAG:-rl_v5}; FROZEN=${FROZEN:-/vol/q36/critic/v5/ckpt_step000500.pt}; OTHER=${OTHER:-/vol/q36/critic/v2/ckpt_step3000.pt}; EVERY=${EVERY:-20}; SFT=${SFT:-/vol/q36/verbalizer/v1b/final}
PAIRS=${PAIRS:-/vol/q36/twinsL/pairs.txt}; LOGD=/home/celeste/nlt-q36-logs; D=/home/celeste/shared/reports/nlt-27b-olens/data; RLD=/vol/q36/rl/$RL_TAG; APPFILE=${APPFILE:-$LOGD/rl_app_$RL_TAG.txt}
log(){ echo "[ptwins] $(date -u +%H:%M) $*"; }
for i in $(seq 1 200); do [ "$(timeout 120 modal volume ls nlt ${PAIRS%/*} 2>/dev/null | grep -c pairs.txt)" -ge 1 ] && break; [ $((i % 5)) -eq 0 ] && log "waiting for $PAIRS"; sleep 60; done
declare -A Q; miss=0
for it in $(seq 1 600); do
  RLS=$(timeout 120 modal volume ls nlt ${RLD#/vol/} 2>/dev/null); RES=$(timeout 120 modal volume ls nlt q36/results 2>/dev/null)
  steps="0000 $(echo "$RLS" | grep -oE "step_[0-9]+" | sed 's/step_//' | sort -u | tr '\n' ' ')"
  for st in $steps; do stn=$((10#$st)); [ $((stn % EVERY)) -eq 0 ] || continue
    AD=$SFT; [ $stn -gt 0 ] && AD=$RLD/step_$st
    if ! echo "$RLS" | grep -q "dumpsL_$st.parquet"; then
      [ -n "${Q[dump_$st]:-}" ] || { enqueue_eval "--data-dir /vol/q36/data --adapter $AD --out $RLD/dumpsL_$st.parquet --split val --pair-ids-file $PAIRS --n 1024 --batch 64 --max-new 208 --source ${RL_TAG}_step$st" ptw_dump_${RL_TAG}_$st 1 dump_verbalizer.py && Q[dump_$st]=1; }; continue; fi
    if ! echo "$RLS" | grep -q "twinsL_$st.parquet"; then
      [ -n "${Q[tw_$st]:-}" ] || { enqueue_eval "--dump $RLD/dumpsL_$st.parquet --out $RLD/twinsL_$st.parquet" ptw_build_${RL_TAG}_$st 1 policy_twins.py && Q[tw_$st]=1; }; continue; fi
    for J in frozen other; do out=rl_${RL_TAG}_ptwinsL_${J}_$st; [ -f $D/$out.json ] && continue
      if echo "$RES" | grep -q "$out.json"; then timeout 120 modal volume get nlt q36/results/$out.json /tmp/q36_$out.json --force >/dev/null 2>&1; grep -q '"elapsed_min"' /tmp/q36_$out.json 2>/dev/null && { cp /tmp/q36_$out.json $D/$out.json; log "pulled $out: $(python3 -c "
import json; v=json.load(open('$D/$out.json'))['twins']['policy_twins']['variants']
print(' | '.join(f\"{k}: P {x['p_true_gt_twin']:.3f} [{x['ci95_p'][0]:.3f},{x['ci95_p'][1]:.3f}] FM {x['proxy_p_true_gt_twin']:.3f} [{x['proxy_ci95_p'][0]:.3f},{x['proxy_ci95_p'][1]:.3f}] n {x['n_positions']}\" for k, x in v.items() if k in ('twin_shift', 'twin_new', 'dm_full')))" 2>/dev/null)"; systemd-run --user --scope -q -p MemoryMax=2G python3 scripts/plot_nlt_q36_policy_twins.py --tag $RL_TAG 2>&1 | grep -E "^saved|Traceback" | head -2; (cd /home/celeste/shared/reports/nlt-27b-olens && systemd-run --user --scope -q -p MemoryMax=1G python3 build_html.py >/dev/null 2>&1); }; continue; fi
      [ -n "${Q[$out]:-}" ] && continue; CK=$FROZEN; [ $J = other ] && CK=$OTHER
      enqueue_eval "--data-dir /vol/q36/data --ckpt $CK --out /vol/q36/results/$out.json --sets '' --twins 'policy_twins:$RLD/twinsL_$st.parquet' --fixed-from-twins --n-fixed 1024 --twins-n 1024 --ode-steps 32 --skip-samples --skip-sw" $out 1 && Q[$out]=1
    done
  done
  A=$(grep -oE "ap-[A-Za-z0-9]+" $APPFILE 2>/dev/null | head -1); if [ -n "$A" ]; then app_live "$A"; r=$?; [ $r -eq 0 ] && miss=0; [ $r -eq 1 ] && miss=$((miss + 1)); fi
  if [ $miss -ge 3 ]; then pend=0; for k in "${!Q[@]}"; do case $k in rl_*) [ -f $D/$k.json ] || pend=$((pend + 1));; esac; done; [ $pend -eq 0 ] && { log "RL app ended and every policy-twin eval pulled; done"; break; }; after=$((${after:-0} + 1)); [ $after -ge 36 ] && { log "RL app ended; $pend evals still missing after 3 h; giving up"; break; }; fi
  sleep 300
done
log "PTWINS DONE"
