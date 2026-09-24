"""QC of the g2 pilot (scripts/modal_scale.py g2_pilot): per-style-cell deterministic checks, diversity, and a stratified Sonnet-5 claim judge.
  with-local-keys python scripts/g2_qc.py <g2_pilot.parquet> <out.json> [n_per_cell]
Deterministic (every rendering): parse rate, exact facts missing, hidden exact values leaked, unsupported numbers, words; fact-sheet parse rate and
validation drop rate. Diversity per (length, format) cell: distinct-2 / distinct-3, near-duplicate rate (MinHash-style Jaccard >= 0.8 on word 5-grams,
sampled pairs). Judge (judge_batch.py prompt, full prefix as ground truth): stratified by (mode, length) + the 400 Opus-overlap positions' first
rendering (Opus verdicts reused from data/scale/pilot_judge.json) -> precision / claims per explanation / informativeness per cell and per axis level."""
import asyncio, collections, json, os, random, re, sys
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_batch as jb  # noqa: E402


def shingles(t, n=5):
    w = t.lower().split(); return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


def distinct(texts, n):
    grams = [tuple(t.lower().split()[i:i + n]) for t in texts for i in range(max(0, len(t.split()) - n + 1))]
    return len(set(grams)) / max(1, len(grams))


async def judge_all(items, conc=24):
    import anthropic
    hdr = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else {}
    cl = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=hdr, max_retries=8); sem = asyncio.Semaphore(conc); out = {}

    async def one(k, src, z):
        async with sem:
            try:
                m = await cl.messages.create(model=jb.MODEL, max_tokens=2500, system=[{"type": "text", "text": jb.SYS, "cache_control": {"type": "ephemeral"}}],
                                             messages=[{"role": "user", "content": jb.user_msg(src, z, 8000)}])
                out[k] = jb.parse("".join(b.text for b in m.content if getattr(b, "type", None) == "text"))
            except Exception as e: print("ERR", k, str(e)[:100], flush=True)
    await asyncio.gather(*[one(*it) for it in items]); return out


def main(src, out_p, n_cell=40):
    rows = pq.read_table(src).to_pylist(); rnd = random.Random(0)
    R = []   # one entry per rendering
    for ri, r in enumerate(rows):
        for j, (t, s, q) in enumerate(zip(r["renders"], r["styles"], r["qc"])):
            R.append({"ri": ri, "j": j, "text": t, "style": json.loads(s), "qc": json.loads(q), "src": r["src"], "window": r["window"]})
    fa = [r for r in rows if r["facts"]]; vs = [json.loads(r["validate"]) for r in rows if r["validate"]]
    res = {"n_positions": len(rows), "fact_sheet_parse_rate": len(fa) / len(rows), "fact_values_kept": sum(v["kept"] for v in vs), "fact_values_dropped": sum(v["dropped"] for v in vs),
           "renders": len(R), "claims_per_position": float(np.mean([len(r["claims"]) for r in fa])),
           "facts_per_position": float(np.mean([len(json.loads(r["fact_ladders"] or "[]")) for r in fa]))}
    def det(sub):
        ok = [x for x in sub if x["text"]]
        return {"n": len(sub), "parse": len(ok) / max(1, len(sub)), "exact_missing": float(np.mean([x["qc"].get("exact_missing", 0) > 0 for x in ok])) if ok else None,
                "leaked": float(np.mean([x["qc"].get("leaked", 0) > 0 for x in ok])) if ok else None,
                "unsupported_numbers": float(np.mean([x["qc"].get("unsupported_numbers", 0) > 0 for x in ok])) if ok else None,
                "words": float(np.mean([x["qc"].get("words", 0) for x in ok])) if ok else None}
    axes = {"mode": lambda x: x["style"]["mode"], "hedge": lambda x: x["style"]["hedge"], "focus": lambda x: x["style"]["focus"][0], "length": lambda x: x["style"]["length"],
            "format": lambda x: x["style"]["format"], "window": lambda x: x["window"]}
    for t in ("person", "number", "quote", "date", "place", "organisation"): axes[f"spec_{t}"] = (lambda tt: lambda x: x["style"]["spec"][tt])(t)
    res["deterministic"] = {"all": det(R), **{ax: {lv: det([x for x in R if f(x) == lv]) for lv in sorted(set(f(x) for x in R))} for ax, f in axes.items()}}
    cells = collections.defaultdict(list)
    for x in R:
        if x["text"]: cells[(x["style"]["length"], x["style"]["format"])].append(x["text"])
    div = {}
    for c, ts in cells.items():
        sh = [shingles(t) for t in ts]; pairs = [(rnd.randrange(len(sh)), rnd.randrange(len(sh))) for _ in range(min(3000, len(sh) ** 2))]
        nd = [len(sh[a] & sh[b]) / max(1, len(sh[a] | sh[b])) >= 0.8 for a, b in pairs if a != b]
        div[f"{c[0]}|{c[1]}"] = {"n": len(ts), "distinct2": distinct(ts, 2), "distinct3": distinct(ts, 3), "near_dup_rate": float(np.mean(nd)) if nd else None}
    res["diversity"] = div
    # judge: stratified (mode, length) + overlap positions' first rendering
    strata = collections.defaultdict(list)
    for x in R:
        if x["text"] and x["src"] == "g1_fresh": strata[(x["style"]["mode"], x["style"]["length"])].append(x)
    items, meta = [], {}
    for c, xs in strata.items():
        for x in rnd.sample(xs, min(n_cell, len(xs))):
            k = f"g2|{x['ri']}|{x['j']}"; items.append((k, rows[x["ri"]]["text"], x["text"])); meta[k] = x
    for ri, r in enumerate(rows):
        if r["src"] == "opus_overlap" and r["renders"] and r["renders"][0]:
            k = f"ov|{ri}|0"; items.append((k, r["text"], r["renders"][0])); meta[k] = {"ri": ri, "j": 0, "style": json.loads(r["styles"][0]), "src": "opus_overlap"}
    J = asyncio.run(judge_all(items))
    def agg(keys):
        v = [J[k] for k in keys if J.get(k)]
        return jb.aggregate(v) if v else {"n": 0}
    res["judge"] = {"all_g2_fresh": agg([k for k in J if k.startswith("g2|")]), "overlap_first_render": agg([k for k in J if k.startswith("ov|")])}
    for ax, f in axes.items():
        res["judge"][ax] = {lv: agg([k for k in J if k.startswith("g2|") and f(meta[k]) == lv]) for lv in sorted(set(f(meta[k]) for k in J if k.startswith("g2|")))}
    res["judge"]["cells_mode_length"] = {f"{c[0]}|{c[1]}": agg([k for k in J if k.startswith("g2|") and (meta[k]["style"]["mode"], meta[k]["style"]["length"]) == c]) for c in strata}
    op = json.load(open("/home/celeste/shared/reports/nla-flow-prior/data/scale/pilot_judge.json"))
    res["judge"]["opus_same_positions"] = op["aggregate"].get("opus"); res["judge"]["gemma_v0_same_positions"] = op["aggregate"].get("gemma v0_opus")
    res["judge_rows"] = {k: v for k, v in J.items()}
    json.dump(res, open(out_p, "w"), indent=1)
    slim = lambda a: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in (a or {}).items() if k in ("n", "claims_per_expl", "claim_precision", "informativeness_1_10", "hallucination_1_10")}
    print(json.dumps({k: res[k] for k in ("n_positions", "fact_sheet_parse_rate", "fact_values_kept", "fact_values_dropped", "renders", "claims_per_position", "facts_per_position")}))
    print("deterministic all:", res["deterministic"]["all"])
    for ax in ("mode", "length", "format", "hedge", "spec_person", "spec_number", "window"):
        print(f"-- {ax}"); [print(f"   {lv:14s} det {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()})}  judge {slim(res['judge'].get(ax, {}).get(lv))}") for lv, d in res["deterministic"][ax].items()]
    print("judge all g2:", slim(res["judge"]["all_g2_fresh"]), "| overlap first:", slim(res["judge"]["overlap_first_render"]), "| opus same:", slim(res["judge"]["opus_same_positions"]))
    print("diversity:", {c: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()} for c, d in sorted(div.items())})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 40)
