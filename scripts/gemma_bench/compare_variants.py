"""Quality guard for the pipeline bench: production deterministic QC (scripts/g2_qc.py's checks), per-axis breakdown, diversity per
(length, format) cell, fact-sheet field presence, all on the SAME positions across variants.
  python scripts/gemma_bench/compare_variants.py <out.json> <name=parquet> [<name=parquet> ...]
The production g2 labels of the same rows (lab_0000 sliced to the bench rows) are one of the inputs (name prod_lab)."""
import collections, json, os, random, sys
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g2_qc import shingles, distinct  # noqa: E402  (production QC helpers)


def renders_of(rows):
    R = []
    for ri, r in enumerate(rows):
        for j, (t, s, q) in enumerate(zip(r["renders"], r["styles"], r["qc"])):
            R.append({"ri": ri, "j": j, "text": t, "style": json.loads(s), "qc": json.loads(q), "window": r["window"]})
    return R


def det(sub):
    ok = [x for x in sub if x["text"]]
    f = lambda k: float(np.mean([x["qc"].get(k, 0) > 0 for x in ok])) if ok else None
    return {"n": len(sub), "parse": len(ok) / max(1, len(sub)), "exact_missing": f("exact_missing"), "leaked": f("leaked"), "unsupported_numbers": f("unsupported_numbers"),
            "pass": float(np.mean([not any(x["qc"].get(k, 0) for k in ("exact_missing", "leaked", "unsupported_numbers", "parse_fail")) for x in ok])) * len(ok) / max(1, len(sub)) if sub else None,
            "words": float(np.mean([x["qc"].get("words", 0) for x in ok])) if ok else None}


def summarize(rows):
    R = renders_of(rows); rnd = random.Random(0)
    fa = [json.loads(r["facts"]) for r in rows if r["facts"]]; vs = [json.loads(r["validate"]) for r in rows if r["validate"]]
    res = {"n_positions": len(rows), "fact_sheet_parse_rate": len(fa) / len(rows), "values_kept": sum(v["kept"] for v in vs), "values_dropped": sum(v["dropped"] for v in vs),
           "value_drop_rate": sum(v["dropped"] for v in vs) / max(1, sum(v["kept"] + v["dropped"] for v in vs)),
           "facts_per_position": float(np.mean([len(json.loads(r["fact_ladders"] or "[]")) for r in rows if r["facts"]])),
           "claims_per_position": float(np.mean([len(r["claims"]) for r in rows if r["facts"]])),
           "field_presence": {k: float(np.mean([bool(f.get(k)) for f in fa])) for k in ("topic", "genre", "voice", "doing", "next", "sentiment", "format", "last_words")},
           "field_words": {k: float(np.mean([len(str(f.get(k, "")).split()) for f in fa if f.get(k)] or [0])) for k in ("topic", "genre", "voice", "doing", "next", "sentiment", "format")},
           "per_position_lists": {k: float(np.mean([len(f.get(k) or []) for f in fa])) for k in ("entities", "numbers", "dates", "quotes")},
           "renders": len(R), "any_pass_rate": float(np.mean([r["n_pass"] > 0 for r in rows])), "mean_n_pass": float(np.mean([r["n_pass"] for r in rows])),
           "deterministic_all": det(R)}
    axes = {"mode": lambda x: x["style"]["mode"], "length": lambda x: x["style"]["length"], "format": lambda x: x["style"]["format"], "window": lambda x: x["window"],
            "hedge": lambda x: x["style"]["hedge"]}
    res["deterministic"] = {ax: {lv: det([x for x in R if f(x) == lv]) for lv in sorted(set(f(x) for x in R))} for ax, f in axes.items()}
    cells = collections.defaultdict(list)
    for x in R:
        if x["text"]: cells[(x["style"]["length"], x["style"]["format"])].append(x["text"])
    div = {}
    for c, ts in cells.items():
        sh = [shingles(t) for t in ts]; pairs = [(rnd.randrange(len(sh)), rnd.randrange(len(sh))) for _ in range(min(3000, len(sh) ** 2))]
        nd = [len(sh[a] & sh[b]) / max(1, len(sh[a] | sh[b])) >= 0.8 for a, b in pairs if a != b]
        div[f"{c[0]}|{c[1]}"] = {"n": len(ts), "distinct2": distinct(ts, 2), "distinct3": distinct(ts, 3), "near_dup_rate": float(np.mean(nd)) if nd else None}
    res["diversity"] = div
    res["diversity_mean"] = {k: float(np.mean([d[k] for d in div.values() if d[k] is not None])) for k in ("distinct2", "distinct3", "near_dup_rate")}
    return res


def main(out_p, specs):
    res = {}
    for spec in specs:
        name, path = spec.split("=", 1)
        rows = pq.read_table(path).to_pylist(); res[name] = summarize(rows); res[name]["path"] = path
        r = res[name]; d = r["deterministic_all"]
        print(f"{name:22s} n={r['n_positions']} facts_parse={r['fact_sheet_parse_rate']:.4f} drop={r['value_drop_rate']:.3f} facts/pos={r['facts_per_position']:.2f} "
              f"claims/pos={r['claims_per_position']:.2f} | render parse={d['parse']:.3f} pass={d['pass']:.3f} miss={d['exact_missing']:.3f} leak={d['leaked']:.3f} "
              f"badnum={d['unsupported_numbers']:.3f} words={d['words']:.1f} | any_pass={r['any_pass_rate']:.3f} n_pass={r['mean_n_pass']:.2f} | "
              f"d2={r['diversity_mean']['distinct2']:.3f} d3={r['diversity_mean']['distinct3']:.3f} dup={r['diversity_mean']['near_dup_rate']:.4f}")
        print("   presence", {k: round(v, 3) for k, v in r["field_presence"].items()}, "lists", {k: round(v, 2) for k, v in r["per_position_lists"].items()})
    json.dump(res, open(out_p, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
