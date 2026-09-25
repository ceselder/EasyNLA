#!/usr/bin/env bash
# CRITIC v4 (orchestrator 08:15, FIRST): v1b's config (no unconditional phase, uncond-frac 0.10), NO anchor, ONE PASS over the harvested text/v3 pools, STREAMING: stage 1 starts when >= MIN_NEW crafted
# train shards exist; every later stage resumes (--resume ckpt_latest, optimizer state) on the shards crafted since, so every (pair, text) row is seen exactly once overall. Pool weights within a stage
# are proportional to distinct rows. Val sets = the v1 held-out texts (fixed across stages, comparable to v1b's spot evals) + text/v3 val when it exists. Steps per stage = one pass + margin; --max-passes 1 stops it.
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; source /home/celeste/nlt-q36-logs/gpu_lib.sh
BAND=12,16,20,24,28,30,32,36,40,42,44,48,52,54,56,60; TX3=/vol/q36/text/v3; TX1=/vol/q36/text/v1; LOGD=/home/celeste/nlt-q36-logs; D=/home/celeste/shared/reports/nlt-27b-olens/data
MIN_NEW=${MIN_NEW:-8}; MIN_NEW_LATER=${MIN_NEW_LATER:-4}; TAG=${TAG:-v4}; EXTRA=${EXTRA:-}
log(){ echo "[v4] $(date -u +%H:%M) $*"; }
nfiles(){ timeout 120 modal volume ls nlt "$1" 2>/dev/null | grep -cE "$2" || true; }
run(){ out=$(spawn_retry env NLT_Q36_GPU=H100 timeout 900 modal run --detach scripts/modal_nlt_q36.py --task hf --gpus 1 --script "$1" --args "$2"); echo "$out" | sed "s/^/[$3] /" | tee -a $LOGD/apps.txt; ledger_add "$out" 1 "$3"; }
USED=$LOGD/critic_${TAG}_used_shards.txt; touch $USED; stage=$(( $(grep -c "^stage" $USED) + 1 ))
mkdir -p /tmp/q36_v3stats
while true; do
  need=$MIN_NEW; [ $stage -gt 1 ] && need=$MIN_NEW_LATER
  for i in $(seq 1 600); do
    crafted=$(timeout 120 modal volume ls nlt q36/text/v3/train 2>/dev/null | grep -oE "stats__shard[0-9]+_part0000\.json" | sed -E 's/stats__(.*)\.json/\1/' | sort -u)
    new=$(comm -23 <(echo "$crafted" | sort) <(grep -v "^stage" $USED | sort) | grep -v '^$'); n=$(echo "$new" | grep -c .)
    [ "$n" -ge $need ] && break
    if [ $stage -gt 1 ] && [ "$n" -gt 0 ] && [ -f $LOGD/.harvest_done ] && [ "$(nfiles q36/text/v3/train 'stats__')" -ge 26 ]; then break; fi     # harvest + craft finished: take what is left
    [ $((i % 6)) -eq 0 ] && log "stage $stage waiting: $n new crafted shards (need $need)"; sleep 300
  done
  [ "$n" -eq 0 ] && { log "no new shards and the harvest is done -> CRITIC $TAG COMPLETE after $((stage - 1)) stages"; break; }
  log "stage $stage: $n new shards: $(echo $new | tr '\n' ' ')"
  for f in $new; do timeout 60 modal volume get nlt q36/text/v3/train/stats__$f.json /tmp/q36_v3stats/stats__$f.json --force >/dev/null 2>&1; done
  SPEC=$(python3 - "$stage" $new <<'PY'
import json, sys
stage = int(sys.argv[1]); shards = sys.argv[2:]; use = ["craft_full", "craft_nodelta", "raw_all", "raw_nodelta", "craft_delta", "craft_newfaded", "craft_nojl", "jlens", "olens_j"]
rows = {}
for sh in shards:
    for k, v in json.load(open(f"/tmp/q36_v3stats/stats__{sh}.json")).get("pool_rows", {}).items():
        if k in use: rows[k] = rows.get(k, 0) + v
S = sum(rows.values()); spec = ",".join(f"{k}={rows[k] / S:.4f}:" + ";".join(f"/vol/q36/text/v3/train/{k}__{sh}.parquet" for sh in shards) for k in rows if rows[k] > 0)
if stage == 1: spec += f",describer={15199 / (S + 15199):.4f}:/vol/q36/text/v1/train/describer_sonnet5_A__*.parquet"; S += 15199
one_pass = int(S / (1024 * 0.9 * 0.9)) + 1
json.dump({"stage": stage, "shards": shards, "rows": rows, "total_text_rows": S, "one_pass_steps": one_pass}, open("/tmp/q36_v4_stage.json", "w")); print(spec)
PY
)
  ONE=$(python3 -c "import json; print(json.load(open('/tmp/q36_v4_stage.json'))['one_pass_steps'])"); STEP0=0; RESUME=""
  if [ $stage -gt 1 ]; then
    PREV=$(grep -E "^\[critic_${TAG}_s$((stage - 1))\] https" $LOGD/apps.txt | tail -n 1 | grep -oE "ap-[A-Za-z0-9]+")
    for i in $(seq 1 600); do L=$(app_list); [ -n "$L" ] && ! echo "$L" | grep -vE "stopped|stopping" | grep -q "$PREV" && break; [ $((i % 6)) -eq 0 ] && log "stage $stage: previous stage $PREV still running"; sleep 300; done
    for i in $(seq 1 60); do [ "$(timeout 120 modal volume ls nlt q36/critic/$TAG 2>/dev/null | grep -c one_pass_stop.json)" -ge 1 ] && break; sleep 60; done
  fi
  if [ $stage -gt 1 ]; then STEP0=$(timeout 60 modal volume get nlt q36/critic/$TAG/one_pass_stop.json /tmp/q36_v4_stop.json --force >/dev/null 2>&1 && python3 -c "import json; print(json.load(open('/tmp/q36_v4_stop.json'))['stopped_at_step'])" || echo 0); RESUME="--resume /vol/q36/critic/$TAG/ckpt_latest.pt"; fi
  STEPS=$(( STEP0 + ONE * 115 / 100 + 50 ))
  VALS="craft_full:$TX1/val/craft_full__*.parquet,describer:$TX1/val/describer_sonnet5_A__*.parquet,raw_all:$TX1/val/raw_all__*.parquet,craft_delta:$TX1/val/craft_delta__*.parquet,craft_newfaded:$TX1/val/craft_newfaded__*.parquet,craft_nojl:$TX1/val/craft_nojl__*.parquet,jlens:$TX1/val/jlens__*.parquet,olens_j:$TX1/val/olens_j__*.parquet"   # per-pool held-out content (orchestrator 09:12); stage 1 launched with the first three only
  [ "$(nfiles q36/text/v3/val 'craft_full__')" -ge 1 ] && VALS="$VALS,craft_full_v3:$TX3/val/craft_full__*.parquet,craft_nodelta_v3:$TX3/val/craft_nodelta__*.parquet"
  log "stage $stage: rows $(cat /tmp/q36_v4_stage.json | cut -c1-300) -> steps $STEP0 + $ONE (x1.15) = $STEPS"
  PRIO=1 wait_gpu 1 || exit 1
  run train_critic.py "--data-dir /vol/q36/data --out /vol/q36/critic/$TAG --tag critic_${TAG}_s$stage --pools '$SPEC' --val-sets '$VALS' --band $BAND --width 1536 --depth 16 --heads 16 --param v --uncond-steps 0 --uncond-frac 0.10 --steps $STEPS --keep-every 500 --max-passes 1 --batch 1024 --micro-batch 128 --eval-every 500 --eval-n 256 --spot-exact-n 64 --spot-ode-steps 16 --max-hours 8.0 $RESUME $EXTRA" critic_${TAG}_s$stage
  grep -q "critic_${TAG}_s$stage\] SPAWNED" $LOGD/apps.txt || { log "stage $stage launch FAILED; retrying in 5 min"; sleep 300; continue; }
  cp /tmp/q36_v4_stage.json $D/critic_${TAG}_stage$stage.json; echo "stage $stage $(date -u +%H:%M)" >> $USED; for f in $new; do echo $f >> $USED; done
  printf 'SPEC=%q\nVALS=%q\nSTEPS=%q\n' "$SPEC" "$VALS" "$STEPS" > $LOGD/critic_${TAG}_s${stage}.launch          # exact launch config for sibling runs (v3c mirrors stage 1)
  A=$(grep -E "^\[critic_${TAG}_s$stage\] https" $LOGD/apps.txt | tail -n 1 | grep -oE "ap-[A-Za-z0-9]+"); log "stage $stage launched: $A"
  miss=0; for i in $(seq 1 600); do L=$(app_list); if [ -z "$L" ]; then sleep 60; continue; fi; if echo "$L" | grep -vE "stopped|stopping" | grep -q "$A"; then miss=0; else miss=$((miss + 1)); [ $miss -ge 3 ] && break; fi; sleep 300; done; log "stage $stage app ended (3 consecutive absences)"
  stage=$((stage + 1))
done
