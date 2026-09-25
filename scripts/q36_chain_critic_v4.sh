#!/usr/bin/env bash
# CRITIC v4 (orchestrator 07:52 #5): v1b's config (no unconditional phase, uncond-frac 0.10) [+ --anchor 0.5 if critic v3b PASSES its gate], trained ONE PASS (--max-passes 1)
# on the enlarged text/v3 pools, pool weights proportional to distinct rows (no pool gets re-epoched while others are under-sampled). Launches when >= MIN_SHARDS harvested shards are crafted
# and the v3b verdict exists (PASS/FAIL; 'pending' waits up to WAIT_V3B_MIN minutes, then launches without the anchor). Step count follows the data: steps = ceil(total rows / text rows per step) + margin, max-passes stops it.
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; source /home/celeste/nlt-q36-logs/gpu_lib.sh
BAND=12,16,20,24,28,30,32,36,40,42,44,48,52,54,56,60; TX3=/vol/q36/text/v3; TX1=/vol/q36/text/v1; LOGD=/home/celeste/nlt-q36-logs; D=/home/celeste/shared/reports/nlt-27b-olens/data
MIN_SHARDS=${MIN_SHARDS:-8}; WAIT_V3B_MIN=${WAIT_V3B_MIN:-180}; TAG=${TAG:-v4}
log(){ echo "[v4] $(date -u +%H:%M) $*"; }
run(){ out=$(NLT_Q36_GPU=H100 timeout 900 modal run --detach scripts/modal_nlt_q36.py --task hf --gpus 1 --script "$1" --args "$2" 2>&1 | grep -E "SPAWNED|modal.com/apps|Error|rror:"); echo "$out" | sed "s/^/[$3] /" | tee -a $LOGD/apps.txt; ledger_add "$out" 1 "$3"; }
t_start=$(date +%s)
for i in $(seq 1 600); do
  n=$(timeout 120 modal volume ls nlt q36/text/v3/train 2>/dev/null | grep -c "stats__"); v=$(timeout 120 modal volume ls nlt q36/text/v3/val 2>/dev/null | grep -c "stats__")
  verdict=$(python3 -c "import json; print(json.load(open('$D/critic_v3b_curve.json')).get('verdict','pending'))" 2>/dev/null || echo pending)
  waited=$(( ($(date +%s) - t_start) / 60 ))
  if [ "$n" -ge $MIN_SHARDS ] && [ "$v" -ge 1 ] && { [ "$verdict" != pending ] || [ $waited -ge $WAIT_V3B_MIN ]; }; then break; fi
  [ $((i % 6)) -eq 0 ] && log "waiting: $n train + $v val shards crafted (need $MIN_SHARDS + 1), v3b verdict $verdict (waited $waited min)"; sleep 300
done
log "launch conditions: $n train + $v val crafted shards, v3b verdict = $verdict"
ANCH=""; [ "$verdict" = PASS ] && ANCH="--anchor 0.5 --anchor-tau 0.05 --anchor-frac 0.5"
# pool weights proportional to distinct rows (from the crafted stats files)
mkdir -p /tmp/q36_v3stats; for f in $(timeout 120 modal volume ls nlt q36/text/v3/train 2>/dev/null | grep -oE "stats__[^ ]+\.json"); do [ -f /tmp/q36_v3stats/$f ] || timeout 60 modal volume get nlt q36/text/v3/train/$f /tmp/q36_v3stats/$f --force >/dev/null 2>&1; done
POOLS=$(python3 - <<'PY'
import json, glob
tot = {}
for f in glob.glob("/tmp/q36_v3stats/stats__*.json"):
    for k, v in json.load(open(f)).get("pool_rows", {}).items(): tot[k] = tot.get(k, 0) + v
use = ["craft_full", "craft_nodelta", "raw_all", "raw_nodelta", "craft_delta", "craft_newfaded", "craft_nojl", "jlens", "olens_j"]
rows = {k: tot.get(k, 0) for k in use if tot.get(k, 0) > 0}; S = sum(rows.values())
spec = ",".join(f"{k}={rows[k] / S:.4f}:/vol/q36/text/v3/train/{k}__*.parquet" for k in rows)
# the Sonnet describer pool (15k rows, v1 val/train dirs) is added at its natural share so it is not re-epoched
spec += f",describer={15199 / (S + 15199):.4f}:/vol/q36/text/v1/train/describer_sonnet5_A__*.parquet"
print(spec); print(json.dumps({"rows": rows, "total_text_rows": S + 15199, "one_pass_steps": int((S + 15199) / (1024 * 0.9 * 0.9)) + 1}), file=open("/tmp/q36_v4_pools.json", "w"))
PY
)
STEPS=$(python3 -c "import json; d=json.load(open('/tmp/q36_v4_pools.json')); print(int(d['one_pass_steps'] * 1.15) + 50)"); log "pools: $POOLS"; log "rows: $(cat /tmp/q36_v4_pools.json) -> steps $STEPS (max-passes 1 stops earlier)"
VALS="craft_full:$TX3/val/craft_full__*.parquet,craft_nodelta:$TX3/val/craft_nodelta__*.parquet,describer:$TX1/val/describer_sonnet5_A__*.parquet,raw_all:$TX3/val/raw_all__*.parquet,craft_full_v1:$TX1/val/craft_full__*.parquet"
wait_gpu 1 || exit 1
run train_critic.py "--data-dir /vol/q36/data --out /vol/q36/critic/$TAG --tag critic_$TAG --pools '$POOLS' --val-sets '$VALS' --band $BAND --width 1536 --depth 16 --heads 16 --param v --uncond-steps 0 --uncond-frac 0.10 --steps $STEPS --keep-every 500 --max-passes 1 --batch 1024 --micro-batch 128 --eval-every 500 --eval-n 256 --spot-exact-n 64 --spot-ode-steps 16 --max-hours 8.0 $ANCH" critic_$TAG
cp /tmp/q36_v4_pools.json $D/critic_${TAG}_pools.json; log "critic $TAG launched (anchor: '${ANCH:-none}')"
