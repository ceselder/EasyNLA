#!/usr/bin/env bash
# Full val-4096 dossier pipeline (featurizer). Stage 1 launches the GPU jobs detached; stage 2 fetches + labels + describes.
#   bash nlt/featurizer/run_val.sh launch      # SAE dossier (1), transcoders (3 layer ranges), maxact (1)
#   bash nlt/featurizer/run_val.sh spec        # after sae + tc parts exist: MAEMM spec, then 3 MAEMM containers
#   bash nlt/featurizer/run_val.sh fetch       # pull /vol/feat outputs to ~/nlt-feat-data
set -euo pipefail
cd ~/nlt
L=~/nlt-feat-logs; mkdir -p $L ~/nlt-feat-data
M="nlt/featurizer/modal_featurizer.py"
SPLIT=${SPLIT:-val}; START=${START:-0}; END=${END:-4096}; PERM=${PERM:--1}
case "$1" in
  launch)
    setsid nohup modal run --detach $M --task sae --split $SPLIT --start $START --end $END --perm-seed $PERM > $L/sae_${SPLIT}_${START}_${END}.log 2>&1 < /dev/null &
    sleep 1
    for R in 10-17 18-25 26-34; do
      setsid nohup modal run --detach $M --task tc --split $SPLIT --start $START --end $END --layers $R --perm-seed $PERM > $L/tc_${SPLIT}_${START}_${END}_$R.log 2>&1 < /dev/null &
      sleep 1
    done
    if [ "${WITH_MAXACT:-1}" = "1" ]; then
      setsid nohup modal run --detach $M --task maxact > $L/maxact.log 2>&1 < /dev/null &
    fi
    echo launched ;;
  spec)
    modal run $M --task spec --split $SPLIT --start $START --end $END 2>&1 | grep "\[spec\]" || true
    S=$(printf "%07d_%07d" $START $END)
    for P in pairs sae tc; do
      setsid nohup modal run --detach $M --task maemm --spec /vol/feat/maemm/$SPLIT/spec_${S}_$P.json --out /vol/feat/maemm/$SPLIT/gen_${S}_$P.parquet > $L/maemm_${SPLIT}_${S}_$P.log 2>&1 < /dev/null &
      sleep 1
    done
    echo launched maemm ;;
  fetch)
    cd ~/nlt-feat-data
    for D in sae_dossier tc_dossier maemm; do
      mkdir -p $D/$SPLIT
      modal volume get nlt feat/$D/$SPLIT $D/ --force 2>&1 | tail -1 || true
    done
    mkdir -p maxact; modal volume get nlt feat/sae_maxact maxact/ --force 2>&1 | tail -1 || true
    # modal volume get of a dir creates a nested dir named after the source; flatten
    for D in sae_dossier tc_dossier maemm; do
      if [ -d $D/$SPLIT/$SPLIT ]; then mv -f $D/$SPLIT/$SPLIT/* $D/$SPLIT/ && rmdir $D/$SPLIT/$SPLIT; fi
    done
    if [ -d maxact/sae_maxact ]; then mv -f maxact/sae_maxact/* maxact/ && rmdir maxact/sae_maxact; fi
    find . -name "*.npy" -path "*sae_dossier*" -delete 2>/dev/null || true   # not needed locally
    du -sh sae_dossier tc_dossier maemm maxact 2>/dev/null ;;
  *) echo "usage: $0 launch|spec|fetch"; exit 1 ;;
esac
