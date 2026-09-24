"""Claim-level judge (Sonnet 5, the judge_batch.py prompt) on the Gemma pilot vs the Opus-5 gold explanations for the SAME (doc, position)s.
Direct calls with bounded concurrency (small job; ~1.5k requests). Run on the box: with-local-keys python scripts/scale_pilot_judge.py <pilot.parquet> <out.json>
Output: per-row claims + per-(source, variant) aggregates (judge_batch.aggregate) + precision by claim type."""
import asyncio, json, os, re, sys, random, collections
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_batch as jb  # noqa: E402

N_OPUS, N_VAR, CONC = int(os.environ.get("N_OPUS", 400)), int(os.environ.get("N_VAR", 150)), int(os.environ.get("CONC", 16))


def opus_expl(resp):
    m = re.search(r"<explanation>(.*?)(?:</explanation>|<\s*$|$)", resp or "", re.S)
    return (m.group(1) if m else resp or "").strip()


async def main(src, out):
    import anthropic
    hdr = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else {}
    cl = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=hdr, max_retries=8)
    rows = pq.read_table(src).to_pylist(); rnd = random.Random(0)
    v0 = [r for r in rows if r["variant"] == "v0_opus"]; keep = set(r["row"] for r in rnd.sample(v0, min(N_OPUS, len(v0))))
    jobs = []
    for r in v0:
        if r["row"] in keep:
            jobs.append((f"opus|{r['row']}", r["text"], opus_expl(r["opus"])))
            if r["explanation"]: jobs.append((f"gemma|v0_opus|{r['row']}", r["text"], r["explanation"]))
    for v in sorted(set(r["variant"] for r in rows) - {"v0_opus"}):
        vr = [r for r in rows if r["variant"] == v and r["explanation"]]
        for r in rnd.sample(vr, min(N_VAR, len(vr))): jobs.append((f"gemma|{v}|{r['row']}", r["text"], r["explanation"]))
    res = json.load(open(out)).get("rows", {}) if os.path.exists(out) else {}
    sem = asyncio.Semaphore(CONC)

    async def one(k, text, z):
        if k in res: return
        async with sem:
            try:
                m = await cl.messages.create(model=jb.MODEL, max_tokens=2500, system=[{"type": "text", "text": jb.SYS, "cache_control": {"type": "ephemeral"}}],
                                             messages=[{"role": "user", "content": jb.user_msg(text, z, 8000)}])
                res[k] = jb.parse("".join(b.text for b in m.content if getattr(b, "type", None) == "text"))
            except Exception as e:
                print("ERR", k, str(e)[:120], flush=True)
    await asyncio.gather(*[one(*j) for j in jobs])
    groups = collections.defaultdict(list)
    for k, v in res.items():
        src_, *rest = k.split("|"); groups["opus" if src_ == "opus" else f"gemma {rest[0]}"].append(v)
    agg = {g: jb.aggregate(v) for g, v in groups.items()}
    bytype = {}
    for g, v in groups.items():
        c = collections.defaultdict(collections.Counter)
        for r in v:
            for x in (r or {}).get("claims", []): c[x["type"]][x["verdict"]] += 1
        bytype[g] = {t: {"n": sum(cc.values()), "precision": cc["supported"] / max(1, sum(cc.values()))} for t, cc in c.items()}
    # paired Opus vs Gemma-V0 on the same rows
    pairs = [(res.get(f"opus|{r}"), res.get(f"gemma|v0_opus|{r}")) for r in keep]
    pairs = [(a, b) for a, b in pairs if a and b]
    prec = lambda r: (sum(c["verdict"] == "supported" for c in r["claims"]) / len(r["claims"])) if r["claims"] else None
    pp = [(prec(a), prec(b)) for a, b in pairs if prec(a) is not None and prec(b) is not None]
    paired = {"n": len(pp), "opus_precision_mean": sum(a for a, _ in pp) / max(1, len(pp)), "gemma_precision_mean": sum(b for _, b in pp) / max(1, len(pp)),
              "frac_gemma_higher": sum(b > a for a, b in pp) / max(1, len(pp)), "frac_tie": sum(b == a for a, b in pp) / max(1, len(pp))}
    json.dump({"rows": res, "aggregate": agg, "by_type": bytype, "paired_v0": paired}, open(out, "w"), indent=1)
    print(json.dumps({"aggregate": {g: {k: (round(x, 3) if isinstance(x, float) else x) for k, x in a.items() if k in ("n", "claims_per_expl", "supported_per_expl", "bad_per_expl", "claim_precision", "hallucination_1_10", "informativeness_1_10")} for g, a in agg.items()}, "paired_v0": paired}, indent=1))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2]))
