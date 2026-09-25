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
  # the ONE-PASS stop saves ckpt_final.pt at an arbitrary step (v4 stage 1: 1791) with no ckpt_stepNNNNNN.pt -> give it a step-named copy so it gets a gate eval (the definitive one-pass-end row) and survives the next stage's ckpt_final
  LS=$(timeout 120 modal volume ls nlt q36/critic/$TAG 2>/dev/null)
  if echo "$LS" | grep -q one_pass_stop.json && timeout 60 modal volume get nlt q36/critic/$TAG/one_pass_stop.json /tmp/q36_${TAG}_ops.json --force >/dev/null 2>&1; then
    OPS=$(python3 -c "import json; print('%06d' % json.load(open('/tmp/q36_${TAG}_ops.json'))['stopped_at_step'])" 2>/dev/null)
    if [ -n "$OPS" ] && ! echo "$LS" | grep -q "ckpt_step$OPS.pt"; then timeout 600 modal volume cp nlt q36/critic/$TAG/ckpt_final.pt q36/critic/$TAG/ckpt_step$OPS.pt >/dev/null 2>&1 && log "one-pass stop at step $OPS: ckpt_final.pt copied to ckpt_step$OPS.pt (gate eval follows)"; LS=$(timeout 120 modal volume ls nlt q36/critic/$TAG 2>/dev/null); fi
  fi
  for ck in $(echo "$LS" | grep -oE "ckpt_step[0-9]+\.pt" | sort -u); do
    st=${ck#ckpt_}; st=${st%.pt}; stn=$((10#${st#step})); [ $stn -lt $MINSTEP ] && continue; [ -n "${launched[$st]:-}" ] && continue
    [ -f $D/bits_${TAG}_$st.json ] && { launched[$st]=1; continue; }
    { grep -q "${TAG}eval_$st\] SPAWNED" $LOGD/apps.txt || grep -q "\[${TAG}eval_$st\] QUEUED" $LOGD/evalq.txt; } && { launched[$st]=1; continue; }        # already launched / queued by an earlier incarnation
    PRIO=1 wait_gpu 1 || exit 1
    run "--data-dir /vol/q36/data --ckpt /vol/q36/critic/$TAG/$ck --out /vol/q36/results/bits_${TAG}_$st.json --sets 'craft_full:$TX/val/craft_full__*.parquet,describer_A:$TX1/val/describer_sonnet5_A__*.parquet' --twins 'craft_twins:$TX/val/twins__*.parquet' --neighbors /vol/q36/data/neigh --neighbor-n 256 --n 256 --ode-steps 64 --skip-samples" ${TAG}eval_$st
    # per-pool held-out content (orchestrator 09:12): the single-line and partial pools on the v1 held-out texts, n 128 (a separate lower-priority spec so the gate eval stays fast)
    PRIO=2 run "--data-dir /vol/q36/data --ckpt /vol/q36/critic/$TAG/$ck --out /vol/q36/results/bits_${TAG}_${st}_pools.json --sets 'craft_delta:$TX1/val/craft_delta__*.parquet,craft_newfaded:$TX1/val/craft_newfaded__*.parquet,craft_nojl:$TX1/val/craft_nojl__*.parquet,jlens:$TX1/val/jlens__*.parquet,olens_j:$TX1/val/olens_j__*.parquet,raw_all:$TX1/val/raw_all__*.parquet' --n 128 --n-fixed 1024 --ode-steps 64 --skip-samples --skip-sw" ${TAG}eval_${st}_pools
    # (e) of the amended criterion (orchestrator 09:25): a small TRAIN-row eval on the pools this critic trains on -> train - held-out PMI gap
    # the one-pass critics (v4 stages, v3c = v4 stage 1) train on harvested text/v3 shards whose pair ids are NOT in pairs_train -> the fixed set is built from the shard files themselves (--fixed-from-texts);
    # the glob = exactly the shards the run has trained on (stage files), so the probe never scores rows the critic has not seen at all. At 0.x passes only that fraction of the probe rows has been seen -> gap diluted; the definitive (e) test is the one-pass end.
    STG=$D/critic_v4_stage*.json; [ "$TAG" = v3c ] && STG=$D/critic_v4_stage1.json
    [ "$TAG" = v5 ] && STG=$D/critic_v5_stage*.json
    if [ "$TAG" = v4 ] || [ "$TAG" = v3c ] || [ "$TAG" = v5 ]; then TRAIN_GLOB=$(python3 -c "import json,glob; sh=sorted({s for f in glob.glob('$STG') for s in json.load(open(f))['shards']}); print(';'.join(f'/vol/q36/text/v3/train/craft_full__{s}.parquet' for s in sh))"); FFT="--fixed-from-texts"; else TRAIN_GLOB=${TRAIN_GLOB:-$TX1/train/craft_full__*.parquet}; FFT=""; fi
    [ "$TAG" = v5 ] && FFT="$FFT --pair-ids-file /vol/q36/critic/$TAG/pair_ids_trained.txt"     # pair-capped critic: probe only the pairs it trained on (chain merges every stage's slice file)
    # orchestrator 13:50: the decisive twin statistic = >= 1024 pairs with DISTINCT positions (<= 1 per position, pairs carrying every variant first) from the held-out twin manifests (v1 val shards 29-31 + the fresh v3 val shard 32),
    # exact Heun 32 + FM view, 95% CIs bootstrapped by position; (a) = point >= .60 AND lower CI > .55 in BOTH views. twins-only spec (no sets), prio 1, ~20 min/checkpoint.
    run "--data-dir /vol/q36/data --ckpt /vol/q36/critic/$TAG/$ck --out /vol/q36/results/bits_${TAG}_${st}_twinsL.json --sets '' --twins 'craft_twins:$TX1/val/twins__*.parquet;/vol/q36/text/v3/val/twins__*.parquet' --fixed-from-twins --n-fixed 1024 --twins-n 1024 --ode-steps 32 --skip-samples --skip-sw" ${TAG}eval_${st}_twinsL
    run "--data-dir /vol/q36/data --ckpt /vol/q36/critic/$TAG/$ck --split train --out /vol/q36/results/bits_${TAG}_${st}_train.json --sets 'craft_full:$TRAIN_GLOB' --n 128 --n-fixed 512 --ode-steps 64 --skip-samples --skip-sw $FFT" ${TAG}eval_${st}_train
    if grep -q "${TAG}eval_$st\] SPAWNED" $LOGD/apps.txt || grep -q "\[${TAG}eval_$st\] QUEUED" $LOGD/evalq.txt; then launched[$st]=1; log "gate eval launched/queued for $st"; else log "gate eval launch for $st FAILED (retry next round)"; fi
  done
  # (d) passes per pool at each step, from the trainer's log lines ("passes max X" every 100 steps) -> data/critic_${TAG}_passes.json
  # every stage's app (staged one-pass critics resume across apps: stage 1 steps 0-1791, stage 2 from 1791, ...), later stages override on equal steps; passes restart at 0 in each stage because the pools are new rows
  TAS=$(grep -E "^\[critic_${TAG}(_s[0-9]+)?\] https" $LOGD/apps.txt | grep -oE "ap-[A-Za-z0-9]+" | awk '!seen[$0]++')
  [ -n "$TAS" ] && for TA in $TAS; do timeout 120 modal app logs $TA 2>/dev/null | grep -E "^\[train\] step [0-9]+ .*passes max"; done | python3 -c "
import sys, re, json
d = {}; e = {}
for l in sys.stdin:
    m = re.search(r'step (\d+) .*passes max ([0-9.]+)', l)
    if m: d[int(m.group(1))] = float(m.group(2))
    x = re.search(r'expo pos ([0-9.]+) \(seen ([0-9.]+), max ([0-9.]+)\) posj ([0-9.]+)', l)
    if m and x: e[int(m.group(1))] = {'per_position_mean': float(x.group(1)), 'per_position_seen_mean': float(x.group(2)), 'per_position_max': float(x.group(3)), 'per_pos_j_mean': float(x.group(4))}
json.dump(d, open('$D/critic_${TAG}_passes.json', 'w')); json.dump(e, open('$D/critic_${TAG}_exposures.json', 'w'))" 2>/dev/null
  new=0
  for f in $(timeout 120 modal volume ls nlt q36/results 2>/dev/null | grep -oE "bits_${TAG}_step[0-9]+(_train|_pools|_twinsL)?\.json" | sort -u); do
    [ -f $D/$f ] && continue; timeout 180 modal volume get nlt q36/results/$f /tmp/q36_$f --force >/dev/null 2>&1; grep -q '"elapsed_min"' /tmp/q36_$f 2>/dev/null || continue; cp /tmp/q36_$f $D/$f; new=1
    case "$f" in *_train.json) log "pulled $f (train rows)"; continue;; *_pools.json) log "pulled $f (per-pool held-out content)"; continue;; *_twinsL.json) log "pulled $f (twins, >= 1024 distinct positions, CIs): $(python3 -c "
import json; d=json.load(open('$D/$f')); v=d['twins']['craft_twins']['variants']
print(' | '.join(f"{k}: P {x['p_true_gt_twin']:.3f} CI [{x['ci95_p'][0]:.3f},{x['ci95_p'][1]:.3f}] / FM {x['proxy_p_true_gt_twin']:.3f} CI [{x['proxy_ci95_p'][0]:.3f},{x['proxy_ci95_p'][1]:.3f}] n {x['n_positions']}" for k, x in v.items() if k in ('twin_shift', 'twin_new')))" 2>/dev/null)"; new=1; continue;; esac
    log "pulled $f: $(python3 -c "
import json; d=json.load(open('$D/$f')); s=d['sets']['craft_full']; tw=d.get('twins',{}).get('craft_twins',{}).get('variants',{})
m=lambda x: x['mean'] if isinstance(x, dict) else x
print(f\"content {s['content_bits']['mean']:.1f} P {s['p_z_gt_dm']:.3f} rp-content {m(s['content_rp_bits']):.1f} | twin_shift P {tw.get('twin_shift',{}).get('p_true_gt_twin',float('nan')):.3f} twin_new P {tw.get('twin_new',{}).get('p_true_gt_twin',float('nan')):.3f} (FM view {tw.get('twin_shift',{}).get('proxy_p_true_gt_twin',float('nan')):.3f} / {tw.get('twin_new',{}).get('proxy_p_true_gt_twin',float('nan')):.3f}) | neigh m1/m4/m16 dd {' / '.join(f\"{s['neighbours'][k]['double_diff']:.1f}\" for k in ('m1','m4','m16') if k in s.get('neighbours',{}))}\")" 2>&1 | tail -n 1)"
  done
  if [ $new -eq 1 ]; then systemd-run --user --scope -q -p MemoryMax=2G python3 scripts/plot_nlt_q36_twin_curve.py 2>&1 | grep -E "^saved|Traceback|Error" | head -2; systemd-run --user --scope -q -p MemoryMax=2G python3 scripts/plot_nlt_q36_critic_curve.py --tag $TAG 2>&1 | grep -iE "error|traceback|PASS|FAIL" | head -3; (cd $REP && systemd-run --user --scope -q -p MemoryMax=1G python3 build_html.py >/dev/null 2>&1); log "curve replotted + report built"; fi
  [ -f $D/bits_${TAG}_step003000.json ] || [ -f $D/bits_${TAG}_step004500.json ] && { log "${TAG} GATE EVALS DONE"; break; }
  sleep 300
done
