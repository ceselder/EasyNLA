#!/bin/bash
# Final acceptance numbers for a chosen critic_para checkpoint (v1.24/v1.25 target: paraphrase retention >= 0.7 AND P(orig > twin_far) >= 0.7
# on V0 AND teacher sentences): scores the para + twinnext2 manifests for BOTH sources (all rows) and prints the paired read.
#   usage: nlt_lens_para_final.sh <TAG> <ckpt file name, e.g. ckpt_step004500.pt>
set -u
TAG=$1; CK=$2
cd /home/celeste/nlt
export NLT_GPU=${NLT_GPU:-H100} NLT_APP=${NLT_APP:-nlt-lens-critic}
LOG=~/nlt-lens-logs; D=/vol/data/qwen3_8b; OUT=$LOG/para_final; mkdir -p $OUT
for m in manifest_para_teacher_v1 manifest_para_v0_ao_tsv1 manifest_twinnext2_teacher_v1 manifest_twinnext2_v0_ao_tsv1 manifest2_v0_ao_tsv1; do
  ( tag=${TAG}_${CK%.pt}_FINAL_$m
    modal run scripts/modal_nlt_critic.py --task manifest --tag $tag --data $D --extra "--ckpt /vol/critic/$TAG/$CK --manifest /vol/evals/$m.parquet --ode-steps 32 --data-device cpu" > $OUT/$tag.log 2>&1
    modal volume get nlt evals/scored_$tag.parquet $OUT/scored_$tag.parquet --force > /dev/null 2>&1 ) &
  sleep 2
done
wait
systemd-run --user --scope -p MemoryMax=2G python3 - "$OUT" "$TAG" "$CK" <<'EOF' 2>/dev/null
import sys, glob, pandas as pd, numpy as np, json
out, tag, ck = sys.argv[1:4]; ln2 = np.log(2); res = {}
for f in sorted(glob.glob(f"{out}/scored_{tag}_{ck[:-3]}_FINAL_*.parquet")):
    m = f.split("_FINAL_")[1][:-8]
    d = pd.read_parquet(f); piv = d.pivot_table(index="pair_id", columns="variant", values="logp", aggfunc="first").dropna(subset=["orig", "empty"])
    b = lambda v: (piv[v] - piv["empty"]) / ln2; r = {"n": int(len(piv)), "PMI_orig": round(float(b("orig").mean()), 2)}
    for v in piv.columns:
        if v in ("orig", "empty"): continue
        mm = piv[["orig", v, "empty"]].dropna(); pv = ((mm[v] - mm["empty"]) / ln2).mean(); po = ((mm["orig"] - mm["empty"]) / ln2).mean()
        r[v] = {"PMI": round(float(pv), 2), "retention": (round(float(pv / po), 2) if abs(po) > 1e-6 else None), "P_orig_gt": round(float((mm["orig"] > mm[v]).mean()), 3), "n": int(len(mm))}
    res[m] = r; print(m, json.dumps(r))
json.dump(res, open(f"{out}/final_{tag}_{ck[:-3]}.json", "w"), indent=1); print("wrote", f"{out}/final_{tag}_{ck[:-3]}.json")
EOF
