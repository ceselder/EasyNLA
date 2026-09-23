"""Truth spot-check of synthetic claims with claude-sonnet-5 as judge (direct parallel calls).

  with-local-keys python scripts/claims_spotcheck.py --text <text_*.jsonl.gz> --claims <family parquet> [--claims ...] --n 200 --out <json>
Samples --n claims uniformly over the given claim files (stratified: equal share per file), shows the judge the anchor's prefix (last 3000
chars) and true continuation, and asks whether the claim is TRUE of the text. Model-internal claims (statements about the model's
predictions) are not checkable from the text and are not sampled here — they are exact by construction."""
import argparse, concurrent.futures as cf, gzip, json, os, random, re
import pyarrow.parquet as pq

SYS = """You check claims about a text. A reader has read the PREFIX and stops at its last character; the CONTINUATION is the true text that follows.
Claims about "the next word", "what comes next", "the next sentence", "coming up", "soon", "the text continues with" refer to the CONTINUATION.
All other claims describe the PREFIX as it stands at its last character (e.g. "the current sentence", "the last word so far", "the text mentions").
Judge only against the text shown. A claim is TRUE if the text supports it (paraphrase and reasonable interpretation are fine), FALSE if the text
contradicts it or it states something the text does not support, UNCLEAR only if it is too vague to judge.
Return ONLY JSON: {"verdict": "true" | "false" | "unclear", "reason": "<one short sentence>"}"""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--text", required=True); ap.add_argument("--claims", action="append", required=True)
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(); rng = random.Random(a.seed)
    rows = {}
    for l in gzip.open(a.text, "rt"):
        r = json.loads(l); rows[r["anchor_id"]] = r
    items = []
    for f in a.claims:
        fam = re.match(r"(internal|text|semantic)", os.path.basename(f)).group(1); pool = []
        for r in pq.read_table(f, columns=["anchor_id", "claims", "types"]).to_pylist():
            if r["anchor_id"] in rows: pool += [(r["anchor_id"], c, t) for c, t in zip(r["claims"], r["types"])]
        for aid, c, t in rng.sample(pool, min(len(pool), a.n // len(a.claims))): items.append({"family": fam, "type": t, "anchor_id": aid, "claim": c})
    import anthropic
    hdr = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else {}
    cl = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=hdr, max_retries=8)

    def judge(it):
        r = rows[it["anchor_id"]]; P = r["prefix_text"]; P = P if len(P) <= 3000 else "…" + P[-3000:]
        msg = f"PREFIX:\n<<<\n{P}\n>>>\n\nCONTINUATION:\n<<<\n{r['cont_text']}\n>>>\n\nCLAIM: {it['claim']}\n\nJSON:"
        for _ in range(3):
            try:
                m = cl.messages.create(model="claude-sonnet-5", max_tokens=300, system=[{"type": "text", "text": SYS, "cache_control": {"type": "ephemeral"}}],
                                       messages=[{"role": "user", "content": msg}])
                txt = "".join(b.text for b in m.content if b.type == "text"); j = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
                return dict(it, verdict=j.get("verdict"), reason=j.get("reason"), source=r["source"])
            except Exception as e: err = str(e)
        return dict(it, verdict="error", reason=err[:200], source=r["source"])
    with cf.ThreadPoolExecutor(a.workers) as ex: res = list(ex.map(judge, items))
    summ = {}
    for fam in sorted({x["family"] for x in res}):
        v = [x["verdict"] for x in res if x["family"] == fam]; n = len(v)
        summ[fam] = {"n": n, "true": v.count("true") / n, "false": v.count("false") / n, "unclear": v.count("unclear") / n, "error": v.count("error") / n}
    by_type = {}
    for x in res: by_type.setdefault(f"{x['family']}:{x['type'].split('/')[0]}", []).append(x["verdict"] == "true")
    out = {"summary": summ, "by_type_true_rate": {k: [sum(v) / len(v), len(v)] for k, v in sorted(by_type.items())}, "items": res}
    json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(summ, indent=1))
    for x in res:
        if x["verdict"] == "false": print(f"  FALSE [{x['family']}:{x['type']}] {x['claim']}  -- {x['reason']}")


if __name__ == "__main__":
    main()
