"""Quality guard for engine / quantisation / labeller changes: Sonnet-5 claim judge (judge_batch.SYS) on the fixed 400 judge prompts.

Run on the box only:  with-local-keys python scripts/gemma_bench/judge.py <name> [<name> ...]
Pulls /vol_glp/scale/bench/results/<name>_judge.parquet, judges every parseable explanation (cached per name in
~/nla-exp-logs/gemma_engine/judge/<name>.json), and prints claim precision, claims / names / numbers / quotes per explanation, next to the
pilot's Opus-5 and Gemma-bf16 numbers on the SAME rows (pilot_judge.json)."""
import asyncio, json, os, re, sys, subprocess, collections
import pyarrow.parquet as pq
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import judge_batch as jb  # noqa: E402
from nla.datagen.scale_templates import clean  # noqa: E402

OUT = os.path.expanduser("~/nla-exp-logs/gemma_engine/judge"); os.makedirs(OUT, exist_ok=True)
PILOT = "/home/celeste/shared/reports/nla-flow-prior/data/scale/pilot_judge.json"
CONC = int(os.environ.get("CONC", 12))


def stats(recs):
    """recs: judge dicts (claims list with type/verdict) -> precision + specificity"""
    n = len(recs); C = [c for r in recs for c in (r or {}).get("claims", [])]
    typ = collections.Counter(c["type"] for c in C)
    return {"n": n, "claims_per_expl": len(C) / max(1, n), "claim_precision": sum(c["verdict"] == "supported" for c in C) / max(1, len(C)),
            "contradicted_per_expl": sum(c["verdict"] == "contradicted" for c in C) / max(1, n),
            "names_per_expl": (typ["name"] + typ["place"] + typ["title"]) / max(1, n), "numbers_per_expl": (typ["number"] + typ["date"]) / max(1, n),
            "quotes_per_expl": typ["quote"] / max(1, n), "halluc_1_10": sum((r or {}).get("hallucination_1_10", 0) for r in recs) / max(1, n),
            "inform_1_10": sum((r or {}).get("informativeness_1_10", 0) for r in recs) / max(1, n)}


async def judge(name):
    import anthropic
    local = f"/tmp/{name}_judge.parquet"
    if not os.path.exists(local):
        subprocess.run(["modal", "volume", "get", "nla-glp", f"/scale/bench/results/{name}_judge.parquet", local, "--force"], check=True, capture_output=True)
    rows = pq.read_table(local).to_pylist(); cache_p = f"{OUT}/{name}.json"
    res = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    hdr = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else {}
    cl = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=hdr, max_retries=8); sem = asyncio.Semaphore(CONC)

    async def one(r):
        k = str(r["row"]); z = clean(r["raw"])
        if k in res or not z: return
        async with sem:
            try:
                m = await cl.messages.create(model=jb.MODEL, max_tokens=2500, system=[{"type": "text", "text": jb.SYS, "cache_control": {"type": "ephemeral"}}],
                                             messages=[{"role": "user", "content": jb.user_msg(r["text"], z, 8000)}])
                res[k] = jb.parse("".join(b.text for b in m.content if getattr(b, "type", None) == "text"))
            except Exception as e: print("ERR", k, str(e)[:120], flush=True)
    await asyncio.gather(*[one(r) for r in rows])
    json.dump(res, open(cache_p, "w"))
    return rows, res


def main(names):
    P = json.load(open(PILOT))["rows"]
    keys = sorted(int(k.split("|")[1]) for k in P if k.startswith("opus|"))
    ref = {"opus (pilot)": [P[f"opus|{r}"] for r in keys if P.get(f"opus|{r}")],
           "gemma bf16 V0 (pilot)": [P[f"gemma|v0_opus|{r}"] for r in keys if P.get(f"gemma|v0_opus|{r}")]}
    table = {k: stats(v) for k, v in ref.items()}
    for name in names:
        rows, res = asyncio.run(judge(name))
        table[name] = stats([res[str(r["row"])] for r in rows if str(r["row"]) in res and res[str(r["row"])]])
        table[name]["parse_fail"] = sum(clean(r["raw"]) is None for r in rows) / len(rows)
        table[name]["mean_out_tokens"] = sum(r["n_out"] for r in rows) / len(rows)
        # paired vs the pilot's bf16 Gemma on the same rows
        pairs = [(P[f"gemma|v0_opus|{r['row']}"], res[str(r["row"])]) for r in rows if P.get(f"gemma|v0_opus|{r['row']}") and res.get(str(r["row"]))]
        prec = lambda x: (sum(c["verdict"] == "supported" for c in x["claims"]) / len(x["claims"])) if x.get("claims") else None
        pp = [(prec(a), prec(b)) for a, b in pairs if prec(a) is not None and prec(b) is not None]
        table[name]["paired_n"] = len(pp); table[name]["paired_delta_precision_vs_bf16pilot"] = (sum(b - a for a, b in pp) / max(1, len(pp)))
    cols = ["n", "claim_precision", "claims_per_expl", "names_per_expl", "numbers_per_expl", "quotes_per_expl", "contradicted_per_expl", "halluc_1_10", "inform_1_10", "parse_fail", "mean_out_tokens", "paired_delta_precision_vs_bf16pilot"]
    print("%-32s" % "config" + "".join("%14s" % c[:13] for c in cols))
    for k, v in table.items():
        print("%-32s" % k[:32] + "".join("%14s" % (("%.3f" % v[c]) if isinstance(v.get(c), float) else str(v.get(c, ""))) for c in cols))
    json.dump(table, open(f"{OUT}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
