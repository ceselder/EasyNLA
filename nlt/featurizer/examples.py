"""~30 example pairs for the report: the dossier, the dossier-sonnet-v1 texts, the teacher text and the passage (for the reader).

  systemd-run --user --scope -p MemoryMax=2G python3 -m nlt.featurizer.examples --split val --n 30 \
      --report-dir ~/shared/reports/natural-language-transcoder
Writes report-dir/data/featurizer_examples_{split}.json and report-dir/featurizer_examples_{split}.html (a fragment the reporter can include).
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def load_parts(pattern, columns=None):
    fs = sorted(glob.glob(pattern))
    return pd.concat([pq.read_table(f, columns=columns).to_pandas() for f in fs], ignore_index=True) if fs else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--data-dir", default=os.path.expanduser("~/nlt-feat-data"))
    ap.add_argument("--dossier-dir", default=os.path.expanduser("~/nlt-feat-data/dossier-sonnet-v1"))
    ap.add_argument("--report-dir", default=os.path.expanduser("~/shared/reports/natural-language-transcoder"))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dossiers = {}
    for f in sorted(glob.glob(f"{a.dossier_dir}/{a.split}/dossier_*.jsonl")):
        for l in open(f):
            d = json.loads(l); dossiers[d["pair_id"]] = d
    z = load_parts(f"{a.dossier_dir}/{a.split}/part_*.parquet")
    zt = {}
    for r in z.itertuples():
        zt.setdefault(r.pair_id, {})[int(r.verbosity)] = r.text
    teacher = load_parts(f"{a.data_dir}/teacher-sonnet-v1/{a.split}/*.parquet", columns=["pair_id", "text", "verbosity"])
    tt = {}
    for r in teacher.itertuples():
        tt.setdefault(r.pair_id, {})[int(r.verbosity)] = r.text
    feats = load_parts(f"{a.data_dir}/features_v1/{a.split}/*.parquet", columns=["pair_id", "context_text", "true_next_token", "final_top10"])
    ctx = feats.set_index("pair_id") if len(feats) else None
    pids = [p for p in dossiers if p in zt]
    rng = np.random.default_rng(a.seed)
    # stratify by band
    by_band = {"pre": [], "ws": [], "motor": []}
    for p in pids:
        j = dossiers[p]["j"]; by_band["pre" if j <= 13 else ("ws" if j <= 32 else "motor")].append(p)
    pick = []
    for b, share in (("pre", 0.25), ("ws", 0.5), ("motor", 0.25)):
        k = int(round(a.n * share)); lst = by_band[b]
        pick += [lst[i] for i in rng.permutation(len(lst))[:k]]
    out = []
    for p in pick:
        d = dossiers[p]
        ex = dict(pair_id=p, i=d["i"], j=d["j"], sae_layer=d["sae_layer"], dossier=d["dossier"], dossier_sonnet=zt.get(p, {}), teacher=tt.get(p, {}))
        if ctx is not None and p in ctx.index:
            ex["context_tail"] = str(ctx.at[p, "context_text"])[-300:]; ex["true_next_token"] = str(ctx.at[p, "true_next_token"])
        out.append(ex)
    os.makedirs(os.path.join(a.report_dir, "data"), exist_ok=True)
    json.dump(out, open(os.path.join(a.report_dir, "data", f"featurizer_examples_{a.split}.json"), "w"), indent=1)
    rows = []
    for ex in out:
        rows.append(
            f"<details><summary><b>{html.escape(ex['pair_id'])}</b> (i={ex['i']}, j={ex['j']}) &mdash; "
            f"<i>{html.escape(ex['dossier_sonnet'].get(0, ''))}</i></summary>"
            f"<p><b>dossier-sonnet-v1:</b> {html.escape(ex['dossier_sonnet'].get(1, ''))}</p>"
            f"<p><b>teacher-sonnet-v1 (sees the passage):</b> {html.escape(ex['teacher'].get(1, ex['teacher'].get(0, '(none)')))}</p>"
            f"<p><b>passage tail (reader reference only, never shown to the describer):</b> <code>{html.escape(ex.get('context_tail', ''))}</code> &rarr; "
            f"<code>{html.escape(ex.get('true_next_token', ''))}</code></p>"
            f"<pre style='white-space:pre-wrap;font-size:11px'>{html.escape(ex['dossier'])}</pre></details>")
    frag = ("<div class='featurizer-examples'><p>The dossier that Sonnet saw (features up/down with autointerp labels, MLP transcoder features, "
            "the activation-oracle readings, the J-lens leanings, the mechanism line and the MAEMM direction texts), its description, and the teacher text.</p>"
            + "\n".join(rows) + "</div>")
    open(os.path.join(a.report_dir, f"featurizer_examples_{a.split}.html"), "w").write(frag)
    print(f"{len(out)} examples -> data/featurizer_examples_{a.split}.json + featurizer_examples_{a.split}.html")


if __name__ == "__main__":
    main()
