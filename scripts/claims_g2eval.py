"""Programmatic-twin benchmark + specificity-ladder calibration for claim critics, on the peer session's g2 positions (READ ONLY:
/vol_glp/scale/g2/shards, owned by diffusion-AR-nla / its scale agent; nothing is written there).

Per held-out g2 position (doc id not among our anchors / gold rows; shards with fact-sheet parse rate < --min-parse skipped, which drops the
FP8-KV-broken shards still being relabelled): up to --facts verified exact facts, each rendered with ONE template per rung so only the detail
changes: "The text mentions X." (quotes: "The text contains the words “X”.") with X = exact value (+ description) / partial / hedged category;
the omitted rung contributes 0 (the unconditional baseline). The wrong-exact twin is the same fact type from another document (programmatic,
non-LLM). Every claim is scored on the position's own activation and on a WRONG activation (another held-out g2 position, other document).
  twins : P(PMI(exact) > PMI(twin)) own vs wrong activation (the wrong-activation excess over 50 % is claim-only cue use), by fact type, and for
          numbers / dates / quotes by distance of the value from the end of the text (chars: <= 16 / 64 / 256 / 1024 / more)
  ladder: mean single-claim PMI per rung (exact, partial, category, omitted = 0) on the own and on the wrong activation, and the share of facts
          with exact > partial > category on the own activation
  python scripts/claims_g2eval.py --adapter <path> --tag <tag>   -> /vol_glp/cond/compnla/g2eval_<tag>.json"""
import argparse, glob, json, os, random, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from claims_controls import Scorer, OUT

DIST = ((16, "<=16"), (64, "<=64"), (256, "<=256"), (1024, "<=1024"), (10 ** 9, ">1024"))


def render(f, rung, twin=False):
    if rung == "exact" or twin:
        v = f["twin"] if twin else f["value"]
        if f["type"] == "quote": return f"The text contains the words “{v}”."
        return f"The text mentions {v}" + (f" ({f['desc']})." if f.get("desc") else ".")
    x = (f.get("ladder") or {}).get(rung)
    if not x: return None
    return f"The text contains {x}." if f["type"] == "quote" else f"The text mentions {x}."


def our_docs():
    import pyarrow.parquet as pq
    s = set()
    for f in glob.glob("/vol_glp/claims/final/final_*.parquet"): s.update(pq.read_table(f, columns=["doc_id"]).column(0).to_pylist())
    for f in glob.glob("/vol_q36/data/acts_qwen36_L42/shard_*.parquet"): s.update(pq.read_table(f, columns=["doc_id"]).column(0).to_pylist())
    return s


def load_g2(n, per_pos, min_parse, excl, seed):
    import pyarrow.parquet as pq
    rng = random.Random(seed); out, stats = [], {"shards_seen": 0, "shards_skipped_parse": 0, "rows_doc_overlap": 0}
    files = sorted(glob.glob("/vol_glp/scale/g2/shards/shard_*.parquet")); rng.shuffle(files)
    for f in files:
        if len(out) >= n: break
        t = pq.read_table(f, columns=["doc_id", "text", "fact_ladders"]).to_pydict(); stats["shards_seen"] += 1
        fl = [json.loads(x) if x else None for x in t["fact_ladders"]]
        if np.mean([bool(x) for x in fl]) < min_parse: stats["shards_skipped_parse"] += 1; continue
        idx = list(range(len(fl))); rng.shuffle(idx); take = []
        for i in idx:
            if not fl[i]: continue
            if t["doc_id"][i] in excl or str(t["doc_id"][i]).split("/")[-1] in excl: stats["rows_doc_overlap"] += 1; continue
            facts = [x for x in fl[i] if x.get("twin") and x.get("value") and x.get("etype") != "last_words"]
            if not facts: continue
            rng.shuffle(facts); take.append((i, facts[:per_pos]))
            if len(take) >= max(4, n // 20): break
        if not take: continue
        A = pq.read_table(f, columns=["activation_vector"]).column(0)
        for i, facts in take:
            out.append({"doc_id": t["doc_id"][i], "text": t["text"][i], "facts": facts, "act": np.asarray(A[i].values.to_numpy(zero_copy_only=False), dtype=np.float32), "shard": os.path.basename(f)})
    return out[:n], stats


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--adapter", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--D", type=int, default=2)
    ap.add_argument("--n", type=int, default=2000); ap.add_argument("--facts", type=int, default=3); ap.add_argument("--min-parse", type=float, default=0.9); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); dev = "cuda:0"; t0 = time.time()
    from nla.flow.scoring import FlowBundle
    from nla.flow.claims import format_claims
    aa = torch.load(a.adapter, map_location="cpu")["args"]; pov = os.path.join(os.path.dirname(a.adapter), "prior_cotrained_latest.pt")
    fb = FlowBundle(aa["prior"], a.adapter, aa["stats"], dev, base="Qwen/Qwen3.6-27B", enc_layer=aa.get("enc_layer", 42), ar_ckpt=aa.get("ar_ckpt", "/vol/ckpts/qwen36_27b/ar_sft_merged"),
                    prior_override=(pov if os.path.exists(pov) else None))
    fmt = (lambda c: format_claims([c])) if aa.get("claim_subsets", 0) > 0 else (lambda c: c)
    sc = Scorer(fb, fmt, dev, a.D)
    excl = our_docs(); excl |= {str(d).split("/")[-1] for d in list(excl)}
    P, st = load_g2(a.n, a.facts, a.min_parse, excl, a.seed); st["positions"] = len(P); st["our_doc_ids"] = len(excl)
    print(f"[g2eval {a.tag}] {len(P)} held-out g2 positions ({st}), {time.time() - t0:.0f}s", flush=True)
    X = fb.norm.normalize(torch.tensor(np.stack([p["act"] for p in P])).to(dev)).float()
    rng = np.random.default_rng(a.seed); wrong = [(k + 1 + int(rng.integers(len(P) - 1))) % len(P) for k in range(len(P))]
    for k in range(len(P)):
        while P[wrong[k]]["doc_id"] == P[k]["doc_id"]: wrong[k] = (wrong[k] + 1) % len(P)
    recs = []
    for k, p in enumerate(P):
        cl, meta = [], []
        for fi, f in enumerate(p["facts"]):
            for rung in ("exact", "partial", "category", "twin"):
                c = render(f, rung, twin=(rung == "twin"))
                if c: cl.append(c); meta.append((fi, rung))
        M = sc.pmi_matrix(torch.stack([X[k], X[wrong[k]]]), cl, [5_000_003 + k, 5_000_003 + wrong[k]]).numpy()
        for fi, f in enumerate(p["facts"]):
            v = str(f["value"]); pos = p["text"].rfind(v); dist = len(p["text"]) - (pos + len(v)) if pos >= 0 else None
            r = {"type": f["type"], "value": v, "dist_chars": dist, "own": {}, "wrong": {}}
            for (fj, rung), c, m0, m1 in zip(meta, cl, M[0], M[1]):
                if fj == fi: r["own"][rung] = float(m0); r["wrong"][rung] = float(m1)
            recs.append(r)
        if k % 200 == 0: print(f"[g2eval {a.tag}] {k + 1}/{len(P)} positions, {time.time() - t0:.0f}s", flush=True)
    def twin_stats(R):
        R = [r for r in R if "exact" in r["own"] and "twin" in r["own"]]
        return {"n": len(R), "own": float(np.mean([r["own"]["exact"] > r["own"]["twin"] for r in R])) if R else None,
                "wrong_activation": float(np.mean([r["wrong"]["exact"] > r["wrong"]["twin"] for r in R])) if R else None}
    res = {"adapter": a.adapter, "tag": a.tag, "D": a.D, "data": st, "twins": {"all": twin_stats(recs)}, "twins_by_type": {}, "twins_by_distance": {}, "ladder": {}}
    for ty in sorted({r["type"] for r in recs}): res["twins_by_type"][ty] = twin_stats([r for r in recs if r["type"] == ty])
    for ty in ("number", "date", "quote"):
        res["twins_by_distance"][ty] = {}
        for i, (hi, nm) in enumerate(DIST):
            lo = DIST[i - 1][0] if i else -1
            res["twins_by_distance"][ty][nm] = twin_stats([r for r in recs if r["type"] == ty and r["dist_chars"] is not None and lo < r["dist_chars"] <= hi])
    for side in ("own", "wrong"):
        L = {}
        for rung in ("exact", "partial", "category"):
            v = [r[side][rung] for r in recs if rung in r[side]]; L[rung] = {"mean": float(np.mean(v)) if v else None, "median": float(np.median(v)) if v else None, "n": len(v)}
        L["omitted"] = {"mean": 0.0, "median": 0.0, "n": len(recs)}
        res["ladder"][side] = L
    full = [r for r in recs if all(k_ in r["own"] for k_ in ("exact", "partial", "category"))]
    res["ladder"]["monotone_own_share"] = float(np.mean([r["own"]["exact"] > r["own"]["partial"] > r["own"]["category"] for r in full])) if full else None
    res["ladder"]["exact_gt_category_own"] = float(np.mean([r["own"]["exact"] > r["own"]["category"] for r in full])) if full else None
    res["ladder"]["exact_gt_category_wrong"] = float(np.mean([r["wrong"]["exact"] > r["wrong"]["category"] for r in full])) if full else None
    os.makedirs(OUT, exist_ok=True); json.dump({"summary": res, "facts": recs}, open(f"{OUT}/g2eval_{a.tag}.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("data", "twins", "twins_by_type", "ladder")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
