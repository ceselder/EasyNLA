#!/usr/bin/env bash
# Launch the two probe-scaling training jobs (2 B200 total): numbers (other-doc + near-miss + digit decoding) and names+quotes (other-doc).
# usage: bash scripts/decodability_scale_launch.sh [data_dir]
set -u
D="${1:-/vol_glp/decodability/scale}"
cd "$(dirname "$0")/.."
(NLA_APP_NAME=nla-decodability-scale setsid nohup modal run scripts/decodability_modal.py --task train \
  --cmd "python scripts/decodability_scale_train.py --data-dir $D --types number --tasks other_doc,near,digits --out $D/results_number.json" \
  > ~/nla-exp-logs/decodability_scale_train_number.out 2>&1 < /dev/null &)
(NLA_APP_NAME=nla-decodability-scale setsid nohup modal run scripts/decodability_modal.py --task train \
  --cmd "python scripts/decodability_scale_train.py --data-dir $D --types name,quote --tasks other_doc --out $D/results_namequote.json" \
  > ~/nla-exp-logs/decodability_scale_train_namequote.out 2>&1 < /dev/null &)
echo "launched: numbers -> ~/nla-exp-logs/decodability_scale_train_number.out ; names+quotes -> ~/nla-exp-logs/decodability_scale_train_namequote.out"
