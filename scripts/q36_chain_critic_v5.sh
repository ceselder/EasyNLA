#!/usr/bin/env bash
# CRITIC v5 (orchestrator 11:05 rule (2), after critic v4's train - held-out PMI gap went 2.6 -> 21.5 -> ~51 bits within ONE row-pass): v1b recipe (no unconditional phase, uncond-frac 0.10), NO anchor,
# ONE PASS with a PAIR CAP: every position contributes <= 4 (i, j) pairs per pass (all pools' text variants of those pairs; train_critic --max-pairs-per-pos 4 --pair-slice s), the other 12 pairs are
# LATER slices' data. Staged like v4: stage 1 = every crafted train shard so far, slice 0; later stages prefer NEW shards (slice 0 = new positions) and only when the harvest is done and no new shard
# is left do they take slice s+1 over all shards (--resume ckpt_latest; passes counted per slice). The trained pair ids of every stage are merged into /vol/q36/critic/v5/pair_ids_trained.txt for the
# train-row probe (criterion (e) on exactly the rows the critic saw). Exposures per position are logged by the trainer (expo pos ...) and by the gate watcher.
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; source /home/celeste/nlt-q36-logs/gpu_lib.sh
BAND=12,16,20,24,28,30,32,36,40,42,44,48,52,54,56,60; TX3=/vol/q36/text/v3; TX1=/vol/q36/text/v1; LOGD=/home/celeste/nlt-q36-logs; D=/home/celeste/shared/reports/nlt-27b-olens/data
MIN_NEW_LATER=${MIN_NEW_LATER:-4}; TAG=${TAG:-v5}; CAP=${CAP:-4}; KEEP_FRAC=${KEEP_FRAC:-0.27}; EXTRA=${EXTRA:-}
log(){ echo "[v5] $(date -u +%H:%M) $*"; }
nfiles(){ timeout 120 modal volume ls nlt "$1" 2>/dev/null | grep -cE "$2" || true; }
run(){ out=$(spawn_retry env NLT_Q36_GPU=H100 timeout 900 modal run --detach scripts/modal_nlt_q36.py --task hf --gpus 1 --script "$1" --args "$2"); echo "$out" | sed "s/^/[$3] /" | tee -a $LOGD/apps.txt; ledger_add "$out" 1 "$3"; }
USED=$LOGD/critic_${TAG}_used.txt; touch $USED; stage=$(( $(grep -c "^stage" $USED) + 1 ))
mkdir -p /tmp/q36_v3stats
while true; do
  # which shards + slice for this stage: new shards at slice 0 first; else (harvest + craft done) the next slice over all shards
  for i in $(seq 1 600); do
    crafted=$(timeout 120 modal volume ls nlt q36/text/v3/train 2>/dev/null | grep -oE "stats__shard[0-9]+_part0000\.json" | sed -E 's/stats__(.*)\.json/\1/' | sort -u)
    seen0=$(grep -E "^slice0 " $USED | awk '{print $2}' | sort -u); new=$(comm -23 <(echo "$crafted" | sort) <(echo "$seen0") | grep -v '^$'); n=$(echo "$new" | grep -c .)
    SLICE=0; SH="$new"
    if [ $stage -eq 1 ] && [ "$n" -ge 1 ]; then break; fi
    if [ "$n" -ge $MIN_NEW_LATER ]; then break; fi
    if [ -f $LOGD/.harvest_done ] && [ "$(nfiles q36/text/v3/train 'stats__')" -ge 26 ]; then
      if [ "$n" -gt 0 ]; then break; fi
      SLICE=$(( $(grep -E "^slice[0-9]+ " $USED | sed -E 's/^slice([0-9]+) .*/\1/' | sort -n | tail -n 1) + 1 )); [ $SLICE -ge $(( 16 / CAP )) ] && { log "all $(( 16 / CAP )) slices used -> CRITIC $TAG COMPLETE"; exit 0; }
      SH="$crafted"; n=$(echo "$SH" | grep -c .); break
    fi
    [ $((i % 6)) -eq 0 ] && log "stage $stage waiting: $n new crafted shards (need $MIN_NEW_LATER, or the harvest to finish for the next slice)"; sleep 300
  done
  log "stage $stage: slice $SLICE over $n shards: $(echo $SH | tr '\n' ' ')"
  for f in $SH; do timeout 60 modal volume get nlt q36/text/v3/train/stats__$f.json /tmp/q36_v3stats/stats__$f.json --force >/dev/null 2>&1; done
  SPEC=$(python3 - "$stage" "$KEEP_FRAC" $SH <<'PY'
import json, sys
stage = int(sys.argv[1]); keep = float(sys.argv[2]); shards = sys.argv[3:]; use = ["craft_full", "craft_nodelta", "raw_all", "raw_nodelta", "craft_delta", "craft_newfaded", "craft_nojl", "jlens", "olens_j"]
rows = {}
for sh in shards:
    for k, v in json.load(open(f"/tmp/q36_v3stats/stats__{sh}.json")).get("pool_rows", {}).items():
        if k in use: rows[k] = rows.get(k, 0) + v
S = sum(rows.values()); spec = ",".join(f"{k}={rows[k] / S:.4f}:" + ";".join(f"/vol/q36/text/v3/train/{k}__{sh}.parquet" for sh in shards) for k in rows if rows[k] > 0)   # weights are re-set ∝ kept rows by the trainer
one_pass = int(S * keep / (1024 * 0.9 * 0.9)) + 1
json.dump({"stage": stage, "shards": shards, "rows_uncapped": rows, "total_rows_uncapped": S, "keep_frac_est": keep, "one_pass_steps_est": one_pass}, open("/tmp/q36_v5_stage.json", "w")); print(spec)
PY
)
  ONE=$(python3 -c "import json; print(json.load(open('/tmp/q36_v5_stage.json'))['one_pass_steps_est'])"); STEP0=0; RESUME=""
  if [ $stage -gt 1 ]; then
    PREV=$(grep -E "^\[critic_${TAG}_s$((stage - 1))\] https" $LOGD/apps.txt | tail -n 1 | grep -oE "ap-[A-Za-z0-9]+"); miss=0
    for i in $(seq 1 600); do L=$(app_list); if [ -z "$L" ]; then sleep 60; continue; fi; if echo "$L" | grep -vE "stopped|stopping" | grep -q "$PREV"; then miss=0; else miss=$((miss + 1)); [ $miss -ge 3 ] && break; fi; [ $((i % 6)) -eq 0 ] && log "stage $stage: previous stage $PREV still running"; sleep 300; done
    for i in $(seq 1 60); do [ "$(timeout 120 modal volume ls nlt q36/critic/$TAG 2>/dev/null | grep -c one_pass_stop.json)" -ge 1 ] && break; sleep 60; done
    STEP0=$(timeout 60 modal volume get nlt q36/critic/$TAG/one_pass_stop.json /tmp/q36_v5_stop.json --force >/dev/null 2>&1 && python3 -c "import json; print(json.load(open('/tmp/q36_v5_stop.json'))['stopped_at_step'])" || echo 0); RESUME="--resume /vol/q36/critic/$TAG/ckpt_latest.pt"
  fi
  STEPS=$(( STEP0 + ONE * 115 / 100 + 50 ))
  VALS="craft_full:$TX1/val/craft_full__*.parquet,describer:$TX1/val/describer_sonnet5_A__*.parquet,raw_all:$TX1/val/raw_all__*.parquet,craft_delta:$TX1/val/craft_delta__*.parquet,craft_newfaded:$TX1/val/craft_newfaded__*.parquet,craft_nojl:$TX1/val/craft_nojl__*.parquet,jlens:$TX1/val/jlens__*.parquet,olens_j:$TX1/val/olens_j__*.parquet"
  [ "$(nfiles q36/text/v3/val 'craft_full__')" -ge 1 ] && VALS="$VALS,craft_full_v3:$TX3/val/craft_full__*.parquet,craft_nodelta_v3:$TX3/val/craft_nodelta__*.parquet"
  log "stage $stage: $(cat /tmp/q36_v5_stage.json | cut -c1-260) -> steps $STEP0 + $ONE (x1.15) = $STEPS (the one-pass rule over the slice stops it)"
  PRIO=1 wait_gpu 1 || exit 1
  run train_critic.py "--data-dir /vol/q36/data --out /vol/q36/critic/$TAG --tag critic_${TAG}_s$stage --pools '$SPEC' --val-sets '$VALS' --band $BAND --width 1536 --depth 16 --heads 16 --param v --uncond-steps 0 --uncond-frac 0.10 --steps $STEPS --keep-every 500 --max-passes 1 --max-pairs-per-pos $CAP --pair-slice $SLICE --batch 1024 --micro-batch 128 --eval-every 500 --eval-n 256 --spot-exact-n 64 --spot-ode-steps 16 --max-hours 8.0 $RESUME $EXTRA" critic_${TAG}_s$stage
  grep -q "critic_${TAG}_s$stage\] SPAWNED" $LOGD/apps.txt || { log "stage $stage launch FAILED; retrying in 5 min"; sleep 300; continue; }
  cp /tmp/q36_v5_stage.json $D/critic_${TAG}_stage$stage.json; echo "stage $stage slice $SLICE $(date -u +%H:%M)" >> $USED; for f in $SH; do echo "slice$SLICE $f" >> $USED; done
  A=$(grep -E "^\[critic_${TAG}_s$stage\] https" $LOGD/apps.txt | tail -n 1 | grep -oE "ap-[A-Za-z0-9]+"); log "stage $stage launched: $A"
  # merge every stage's trained pair ids for the train-row probe (the trainer writes pair_slice_<s>_<tag>.txt at start-up)
  for i in $(seq 1 40); do [ "$(timeout 120 modal volume ls nlt q36/critic/$TAG 2>/dev/null | grep -c "pair_slice_${SLICE}_critic_${TAG}_s$stage.txt")" -ge 1 ] && break; sleep 30; done
  rm -rf /tmp/q36_v5_pairs; mkdir -p /tmp/q36_v5_pairs; for f in $(timeout 120 modal volume ls nlt q36/critic/$TAG 2>/dev/null | grep -oE "pair_slice_[0-9]+_critic_${TAG}_s[0-9]+\.txt" | sort -u); do timeout 120 modal volume get nlt q36/critic/$TAG/$f /tmp/q36_v5_pairs/$f --force >/dev/null 2>&1; done
  cat /tmp/q36_v5_pairs/*.txt 2>/dev/null | sort -u > /tmp/q36_v5_pairs_merged.txt; timeout 120 modal volume put -f nlt /tmp/q36_v5_pairs_merged.txt q36/critic/$TAG/pair_ids_trained.txt >/dev/null 2>&1 && log "trained pair ids merged: $(wc -l < /tmp/q36_v5_pairs_merged.txt) pairs -> /vol/q36/critic/$TAG/pair_ids_trained.txt"
  miss=0; for i in $(seq 1 600); do L=$(app_list); if [ -z "$L" ]; then sleep 60; continue; fi; if echo "$L" | grep -vE "stopped|stopping" | grep -q "$A"; then miss=0; else miss=$((miss + 1)); [ $miss -ge 3 ] && break; fi; sleep 300; done; log "stage $stage app ended (3 consecutive complete listings without it)"
  stage=$((stage + 1))
done
