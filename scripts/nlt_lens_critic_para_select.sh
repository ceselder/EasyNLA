#!/bin/bash
# critic_para checkpoint selection (v1.24): for every new numbered checkpoint /vol/critic/<TAG>/ckpt_step*.pt, score redteam's
# manifest_para_teacher_v1 and manifest_twinnext2_teacher_v1 (first 1024 rows ~ 256 pairs x 4 variants, exact Heun 32) and append the
# paraphrase retention / P(orig > twin_far) line to ~/nlt-lens-logs/critic_para_select.tsv. Stops when STOP file exists.
#   usage: nlt_lens_critic_para_select.sh <TAG>
set -u
TAG=$1
cd /home/celeste/nlt
export NLT_GPU=${NLT_GPU:-H100} NLT_APP=${NLT_APP:-nlt-lens-critic}
LOG=~/nlt-lens-logs; D=/vol/data/qwen3_8b; TSV=$LOG/critic_para_select.tsv; STOP=$LOG/critic_para_select.STOP
mkdir -p $LOG/para_scored; touch $LOG/critic_para_select.done
[ -s $TSV ] || echo -e "tag\tckpt\tmanifest\tn_pairs\tPMI_orig\tpara_light_ret\tpara_strong_ret\tP_orig_gt_twin\ttwin_ret\tP_orig_gt_twin_far\ttwin_far_ret" > $TSV
while [ ! -f $STOP ]; do
  for ck in $(modal volume ls nlt critic/$TAG 2>/dev/null | grep -oE "ckpt_step[0-9]+\.pt" | sort -u); do
    grep -q "^$ck$" $LOG/critic_para_select.done && continue
    for m in manifest_para_teacher_v1 manifest_twinnext2_teacher_v1; do
      out=${TAG}_${ck%.pt}_$m
      modal run scripts/modal_nlt_critic.py --task manifest --tag $out --data $D --extra "--ckpt /vol/critic/$TAG/$ck --manifest /vol/evals/$m.parquet --n 1024 --ode-steps 32 --data-device cpu" > $LOG/para_scored/$out.log 2>&1
      modal volume get nlt evals/scored_$out.parquet $LOG/para_scored/scored_$out.parquet --force > /dev/null 2>&1
      systemd-run --user --scope -p MemoryMax=1G python3 - "$TAG" "$ck" "$m" "$LOG/para_scored/scored_$out.parquet" "$TSV" <<'EOF' 2>/dev/null
import sys, pandas as pd, numpy as np
tag, ck, m, path, tsv = sys.argv[1:6]; ln2 = np.log(2)
try:
    d = pd.read_parquet(path); piv = d.pivot_table(index="pair_id", columns="variant", values="logp", aggfunc="first").dropna(subset=["orig", "empty"])
except Exception as e:
    open(tsv, "a").write(f"{tag}\t{ck}\t{m}\tERR {e}\n"); sys.exit()
b = lambda v: (piv[v] - piv["empty"]) / ln2
po = b("orig").mean()
def ret(v):
    if v not in piv: return float("nan"), float("nan")
    mm = piv[["orig", v, "empty"]].dropna(); pv = ((mm[v] - mm["empty"]) / ln2).mean(); pp = ((mm["orig"] - mm["empty"]) / ln2).mean()
    return (pv / pp if abs(pp) > 1e-6 else float("nan")), (mm["orig"] > mm[v]).mean()
pl, _ = ret("para_light"); ps, _ = ret("para_strong"); tr, tp = ret("twin"); tfr, tfp = ret("twin_far")
open(tsv, "a").write(f"{tag}\t{ck}\t{m}\t{len(piv)}\t{po:.2f}\t{pl:.2f}\t{ps:.2f}\t{tp:.3f}\t{tr:.2f}\t{tfp:.3f}\t{tfr:.2f}\n")
EOF
      tail -1 $TSV
    done
    echo "$ck" >> $LOG/critic_para_select.done
  done
  sleep 180
done
echo "[$(date -u +%H:%M:%S)] selection loop stopped" >> $LOG/critic_para.log
