"""Paired Sonnet-5 claim judge (scripts/judge_batch.py prompt) over the SAME (position, rendering-slot) samples for several bench variants.
  with-local-keys python scripts/gemma_bench/judge_pairs.py <out.json> <n_samples> <name=parquet> [<name=parquet> ...]
Samples n (position, j) pairs (rng 0) among positions whose rendering j parsed in EVERY variant; judges each variant's rendering with the full
prefix as ground truth; reports judge_batch.aggregate per variant + paired precision deltas vs the first variant. Resumable (rows cached)."""
import asyncio, collections, json, os, random, sys
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import judge_batch as jb  # noqa: E402

CONC = int(os.environ.get("CONC", 16))


async def main(out_p, n, specs):
    import anthropic
    hdr = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else {}
    cl = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=hdr, max_retries=8)
    V, texts = {}, None
    for spec in specs:
        name, path = spec.split("=", 1); T = pq.read_table(path); cols = ["doc_id", "n_raw_tokens", "renders", "styles"] + (["text"] if "text" in T.schema.names else [])
        V[name] = T.select(cols).to_pylist()
        if "text" in cols and texts is None: texts = [r["text"] for r in V[name]]          # bench variants carry no text: rows are aligned with the first file that does
    names = list(V); n_pos = min(len(v) for v in V.values()); assert texts is not None, "no input file carries the text column"
    for v in V.values():
        assert all(a["doc_id"] == b["doc_id"] and a["n_raw_tokens"] == b["n_raw_tokens"] for a, b in zip(v[:n_pos], V[names[0]][:n_pos])), "rows misaligned"
        for r, t in zip(v, texts): r["text"] = t
    rnd = random.Random(0); cand = [(i, j) for i in range(n_pos) for j in range(4) if all(len(V[nm][i]["renders"]) > j and V[nm][i]["renders"][j] for nm in names)]
    picks = rnd.sample(cand, min(n, len(cand)))
    res = json.load(open(out_p)).get("rows", {}) if os.path.exists(out_p) else {}
    sem = asyncio.Semaphore(CONC)

    async def one(k, text, z):
        if k in res: return
        async with sem:
            try:
                m = await cl.messages.create(model=jb.MODEL, max_tokens=2500, system=[{"type": "text", "text": jb.SYS, "cache_control": {"type": "ephemeral"}}],
                                             messages=[{"role": "user", "content": jb.user_msg(text, z, 8000)}])
                res[k] = jb.parse("".join(b.text for b in m.content if getattr(b, "type", None) == "text"))
            except Exception as e: print("ERR", k, str(e)[:120], flush=True)
    await asyncio.gather(*[one(f"{nm}|{i}|{j}", V[nm][i]["text"], V[nm][i]["renders"][j]) for i, j in picks for nm in names])
    prec = lambda r: (sum(c["verdict"] == "supported" for c in r["claims"]) / len(r["claims"])) if r and r.get("claims") else None
    agg = {nm: jb.aggregate([res.get(f"{nm}|{i}|{j}") for i, j in picks]) for nm in names}
    by_mode = {nm: {md: jb.aggregate([res.get(f"{nm}|{i}|{j}") for i, j in picks if json.loads(V[nm][i]["styles"][j])["mode"] == md]) for md in ("specific", "mixed", "vague")} for nm in names}
    paired = {}
    for nm in names[1:]:
        pp = [(prec(res.get(f"{names[0]}|{i}|{j}")), prec(res.get(f"{nm}|{i}|{j}"))) for i, j in picks]; pp = [(a, b) for a, b in pp if a is not None and b is not None]
        paired[nm] = {"n": len(pp), "base_precision": sum(a for a, _ in pp) / max(1, len(pp)), "precision": sum(b for _, b in pp) / max(1, len(pp)),
                      "frac_higher": sum(b > a for a, b in pp) / max(1, len(pp)), "frac_lower": sum(b < a for a, b in pp) / max(1, len(pp))}
    json.dump({"rows": res, "picks": picks, "aggregate": agg, "by_mode": by_mode, "paired_vs_first": paired}, open(out_p, "w"), indent=1)
    slim = lambda a: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in (a or {}).items() if k in ("n", "claims_per_expl", "supported_per_expl", "bad_per_expl", "claim_precision", "hallucination_1_10", "informativeness_1_10")}
    for nm in names: print(f"{nm:22s}", slim(agg[nm]), {md: round(by_mode[nm][md].get("claim_precision") or 0, 3) for md in by_mode[nm]})
    print("paired vs", names[0], json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in paired.items()}))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], int(sys.argv[2]), sys.argv[3:]))
