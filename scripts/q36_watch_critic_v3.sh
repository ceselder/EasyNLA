#!/usr/bin/env bash
# critic ${TAG} GATE evals (orchestrator 06:58): for every saved checkpoint from step 2000 (after the 1500 unconditional steps), one eval_bits job on held-out rows -
# craft_full content / P(z>z_dm) / rp, Sonnet describer, claim twins (exact + FM view), same-document neighbour double differences. 1 H100 each, ledger-aware (yields to RL: never exceeds 8).
# Pre-registered pass rule: twin_shift or twin_new P(true > twin) >= 0.60 with craft_full content >= 25 bits at some checkpoint.
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; source /home/celeste/nlt-q36-logs/gpu_lib.sh
TAG=${TAG:-v3b}; MINSTEP=${MINSTEP:-500}; TX=/vol/q36/text/v1; TX1=/vol/q36/text/v1; LOGD=/home/celeste/nlt-q36-logs; D=/home/celeste/shared/reports/nlt-27b-olens/data; REP=/home/celeste/shared/reports/nlt-27b-olens
log(){ echo "[${TAG}eval] $(date -u +%H:%M) $*"; }
run(){ if [ "${QUEUE_MODE:-0}" = 1 ]; then enqueue_eval "$1" "$2" "${PRIO:-9}"; return; fi; out=$(spawn_retry env NLT_Q36_GPU=H100 timeout 900 modal run --detach scripts/modal_nlt_q36.py --task hf --gpus 1 --script eval_bits.py --args "$1"); echo "$out" | sed "s/^/[$2] /" | tee -a $LOGD/apps.txt; ledger_add "$out" 1 "$2"; }
declare -A launched
for i in $(seq 1 400); do
  for ck in $(timeout 120 modal volume ls nlt q36/critic/$TAG 2>/dev/null | grep -oE "ckpt_step[0-9]+\.pt" | sort -u); do
    st=${ck#ckpt_}; st=${st%.pt}; stn=$((10#${st#step})); [ $stn -lt $MINSTEP ] && continue; [ -n "${launched[$st]:-}" ] && continue
    [ -f $D/bits_${TAG}_$st.json ] && { launched[$st]=1; continue; }
    grep -q "${TAG}eval_$st\] SPAWNED" $LOGD/apps.txt && { launched[$st]=1; continue; }        # already launched by an earlier incarnation of this watcher
    PRIO=1 wait_gpu 1 || exit 1
    run "--data-dir /vol/q36/data --ckpt /vol/q36/critic/$TAG/$ck --out /vol/q36/results/bits_${TAG}_$st.json --sets 'craft_full:$TX/val/craft_full__*.parquet,describer_A:$TX1/val/describer_sonnet5_A__*.parquet' --twins 'craft_twins:$TX/val/twins__*.parquet' --neighbors /vol/q36/data/neigh --neighbor-n 256 --n 256 --ode-steps 64 --skip-samples" ${TAG}eval_$st
    if grep -q "${TAG}eval_$st\] SPAWNED" $LOGD/apps.txt; then launched[$st]=1; log "gate eval launched for $st"; else log "gate eval launch for $st FAILED (retry next round)"; fi
  done
  new=0
  for f in $(timeout 120 modal volume ls nlt q36/results 2>/dev/null | grep -oE "bits_${TAG}_step[0-9]+\.json" | sort -u); do
    [ -f $D/$f ] && continue; timeout 180 modal volume get nlt q36/results/$f /tmp/q36_$f --force >/dev/null 2>&1; grep -q '"elapsed_min"' /tmp/q36_$f 2>/dev/null || continue; cp /tmp/q36_$f $D/$f; new=1
    log "pulled $f: $(python3 -c "
import json; d=json.load(open('$D/$f')); s=d['sets']['craft_full']; tw=d.get('twins',{}).get('craft_twins',{}).get('variants',{})
m=lambda x: x['mean'] if isinstance(x, dict) else x
print(f\"content {s['content_bits']['mean']:.1f} P {s['p_z_gt_dm']:.3f} rp-content {m(s['content_rp_bits']):.1f} | twin_shift P {tw.get('twin_shift',{}).get('p_true_gt_twin',float('nan')):.3f} twin_new P {tw.get('twin_new',{}).get('p_true_gt_twin',float('nan')):.3f} (FM view {tw.get('twin_shift',{}).get('proxy_p_true_gt_twin',float('nan')):.3f} / {tw.get('twin_new',{}).get('proxy_p_true_gt_twin',float('nan')):.3f}) | neigh m1/m4/m16 dd {' / '.join(f\"{s['neighbours'][k]['double_diff']:.1f}\" for k in ('m1','m4','m16') if k in s.get('neighbours',{}))}\")" 2>&1 | tail -n 1)"
  done
  if [ $new -eq 1 ]; then systemd-run --user --scope -q -p MemoryMax=2G python3 scripts/plot_nlt_q36_critic_curve.py --tag $TAG 2>&1 | grep -iE "error|traceback|PASS|FAIL" | head -3; (cd $REP && systemd-run --user --scope -q -p MemoryMax=1G python3 build_html.py >/dev/null 2>&1); log "curve replotted + report built"; fi
  [ -f $D/bits_${TAG}_step003000.json ] || [ -f $D/bits_${TAG}_step004500.json ] && { log "${TAG} GATE EVALS DONE"; break; }
  sleep 300
done
