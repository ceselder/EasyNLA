"""Single-claim PMI (frozen critic) of every claim in generated verbalizer outputs (ws_sft_eval.py --save-all json) on their own activation:
distribution + the highest-PMI claims with a coarse category (quotes the text's final words / quotes any span / names a token / other), to see what
the RL reward pays for.   python scripts/score_claims_peek.py --adapter <critic> --gen <json> --parquet <av_sft_test.parquet> --out <json>"""
import argparse, json, os, re, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from claims_controls import Scorer


def category(c, prefix):
    q = re.findall(r"[\"“'‘]([^\"”'’]{3,})[\"”'’]", c)
    tail = prefix[-200:] if prefix else ""
    if any(x.strip() and x.strip() in tail for x in q): return "quotes the final text"
    if q and prefix and any(x.strip() in prefix for x in q): return "quotes an earlier span"
    if q: return "quotes something not in the text"
    if re.search(r"\b(next|final|last) (token|word)", c, re.I): return "names a token"
    return "other"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--adapter", required=True); ap.add_argument("--gen", required=True); ap.add_argument("--parquet", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--D", type=int, default=2); ap.add_argument("--mode", default="sample_t1")
    a = ap.parse_args(); dev = "cuda:0"
    import pyarrow.parquet as pq
    from nla.flow.scoring import FlowBundle
    from nla.flow.claims import format_claims, split_claims
    from nla.schema import extract_explanation
    G = [g for g in json.load(open(a.gen))["all"] if g["mode"] == a.mode]
    cols = pq.ParquetFile(a.parquet).schema_arrow.names
    T = pq.read_table(a.parquet, columns=["activation_vector"] + (["prompt"] if "prompt" in cols else [])).to_pylist()
    aa = torch.load(a.adapter, map_location="cpu")["args"]
    fb = FlowBundle(aa["prior"], a.adapter, aa["stats"], dev, base="Qwen/Qwen3.6-27B", enc_layer=aa.get("enc_layer", 42), ar_ckpt=aa.get("ar_ckpt", "/vol/ckpts/qwen36_27b/ar_sft_merged"))
    fmt = (lambda c: format_claims([c])) if aa.get("claim_subsets", 0) > 0 else (lambda c: c)
    sc = Scorer(fb, fmt, dev, a.D); recs = []
    src = {}
    try:
        S = pq.read_table(a.parquet, columns=["source", "id"]).to_pylist(); src = {i: s for i, s in enumerate(S)}
    except Exception: pass
    for g in G:
        cl = split_claims(extract_explanation(g["generated"]) or "")
        if not cl: continue
        X = fb.norm.normalize(torch.tensor([T[g["row"]]["activation_vector"]]).float().to(dev)).float()
        M = sc.pmi_matrix(X, cl, [11_000_003 + g["row"]]).numpy()[0]
        for c, p in zip(cl, M): recs.append({"row": g["row"], "claim": c, "pmi": float(p)})
    P = np.array([r["pmi"] for r in recs]); top = sorted(recs, key=lambda r: -r["pmi"])[:40]
    res = {"n_claims": len(recs), "pmi_median": float(np.median(P)), "pmi_mean": float(P.mean()), "pmi_p90": float(np.percentile(P, 90)), "top": top, "all": recs}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[peek-score] {len(recs)} claims: single PMI median {np.median(P):.1f} mean {P.mean():.1f} p90 {np.percentile(P, 90):.1f}", flush=True)
    for r in top[:25]: print(f"  {r['pmi']:7.1f}  row {r['row']:3d}  {r['claim'][:150]}", flush=True)


if __name__ == "__main__":
    main()
