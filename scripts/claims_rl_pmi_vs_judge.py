"""Does the frozen critic separate ACCURATE from FABRICATED claims on the RL policy's own outputs? Per judged claim (judge_batch per_row: the
judge's atomic claims of each explanation with type + verdict), the single-claim PMI under the critic (same scoring as the reward: bullet format,
5-point t grid, D noise draws, the row's own activation from av_sft_val_clean1) and its duplicate-discounted value within the explanation (semdup
max(NLI, MiniLM, lexical 4-word span), floor 0.5, nla.flow.claim_redundancy = the reward's rule).
Summary per checkpoint key: AUC (supported vs unsupported+contradicted) of the PMI and of the discounted value; share of accurate / fabricated
claims with PMI > lambda, the precision of the claims above lambda and the claims per explanation that would remain, lambda in LAMS; by claim type.
  python scripts/claims_rl_pmi_vs_judge.py --adapter <critic> --judge <judge_batch json> --keys k1,k2 --out <json>"""
import argparse, json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from claims_controls import Scorer
LAMS = (6.93, 20, 40, 80, 120, 200)


def auc(pos, neg):
    if not pos or not neg: return None
    x = np.concatenate([pos, neg]); r = np.argsort(np.argsort(x)) + 1.0
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--adapter", required=True); ap.add_argument("--judge", required=True); ap.add_argument("--keys", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--D", type=int, default=2)
    a = ap.parse_args(); dev = "cuda:0"
    import pyarrow.parquet as pq
    from nla.flow.scoring import FlowBundle
    from nla.flow.claims import format_claims
    from nla.flow.claim_redundancy import EmbSim, NLISim, LexSim, MaxSim, _discounts
    J = json.load(open(a.judge))
    t = pq.read_table("/vol_q36/data/sft/av_sft_val_clean1.parquet", columns=["activation_vector"]); N = t.num_rows
    acts = torch.tensor(np.asarray(t.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(N, -1))
    aa = torch.load(a.adapter, map_location="cpu")["args"]
    fb = FlowBundle(aa["prior"], a.adapter, aa["stats"], dev, base="Qwen/Qwen3.6-27B", enc_layer=aa.get("enc_layer", 42), ar_ckpt=aa.get("ar_ckpt", "/vol/ckpts/qwen36_27b/ar_sft_merged"))
    fmt = (lambda c: format_claims([c])) if aa.get("claim_subsets", 0) > 0 else (lambda c: c)
    sc = Scorer(fb, fmt, dev, a.D); sim = MaxSim(NLISim(device=dev), EmbSim(device=dev), LexSim(4))
    recs = []
    for key in a.keys.split(","):
        pr = J[key]["per_row"]; rows = sorted(pr, key=lambda r: int(r))
        sim.prefetch([[c["claim"] for c in pr[r]["claims"]] for r in rows])
        for q, r in enumerate(rows):
            cl = [c for c in pr[r]["claims"] if c.get("claim")]
            if not cl: continue
            txt = [c["claim"] for c in cl]; X = fb.norm.normalize(acts[int(r)][None].to(dev)).float()
            M = sc.pmi_matrix(X, txt, [13_000_003 + int(r)]).numpy()[0]; d = _discounts(txt, sim, 0.5)
            for c, p, dd in zip(cl, M, d):
                recs.append({"key": key, "row": int(r), "claim": c["claim"], "type": c.get("type", "other"), "verdict": c.get("verdict"), "pmi": float(p), "value": float(p * (1 - dd)), "discount": float(dd)})
            if q % 100 == 0: print(f"[pmi-judge] {key}: row {q + 1}/{len(rows)}", flush=True)
    def summ(R, n_expl):
        acc = [x for x in R if x["verdict"] == "supported"]; bad = [x for x in R if x["verdict"] in ("unsupported", "contradicted")]
        o = {"n": len(R), "n_accurate": len(acc), "n_fabricated": len(bad), "precision": len(acc) / max(len(acc) + len(bad), 1),
             "auc_pmi": auc([x["pmi"] for x in acc], [x["pmi"] for x in bad]), "auc_value": auc([x["value"] for x in acc], [x["value"] for x in bad]),
             "pmi_median_accurate": float(np.median([x["pmi"] for x in acc])) if acc else None, "pmi_median_fabricated": float(np.median([x["pmi"] for x in bad])) if bad else None, "lambda": {}}
        for l in LAMS:
            ka = sum(x["pmi"] > l for x in acc); kb = sum(x["pmi"] > l for x in bad)
            o["lambda"][str(l)] = {"accurate_kept": ka / max(len(acc), 1), "fabricated_kept": kb / max(len(bad), 1), "precision_kept": ka / max(ka + kb, 1),
                                   "claims_per_expl_kept": (sum(x["pmi"] > l for x in R)) / max(n_expl, 1)}
        return o
    res = {"adapter": a.adapter, "D": a.D, "keys": {}}
    for key in a.keys.split(","):
        R = [x for x in recs if x["key"] == key]; ne = len({x["row"] for x in R}); s = summ(R, ne); s["by_type"] = {}
        for ty in sorted({x["type"] for x in R}):
            Rt = [x for x in R if x["type"] == ty]
            if len(Rt) >= 50: s["by_type"][ty] = summ(Rt, ne)
        res["keys"][key] = s
        print(f"[pmi-judge] {key}: {len(R)} claims, precision {s['precision']:.3f}, AUC pmi {s['auc_pmi']:.3f} value {s['auc_value']:.3f}; median PMI accurate {s['pmi_median_accurate']:.1f} fabricated {s['pmi_median_fabricated']:.1f}; "
              + " | ".join(f"lam {l}: prec {v['precision_kept']:.3f} keeps {v['claims_per_expl_kept']:.1f}/expl" for l, v in s["lambda"].items()), flush=True)
    json.dump({"summary": res, "claims": recs}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
