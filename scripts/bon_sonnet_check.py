"""Gemma-vs-Sonnet agreement on the best-of-N verification labels: ~N bullets sampled (stratified by Gemma's label) from verify_<tag>, re-labelled
by claude-sonnet-5 (synchronous Messages API, same TEXT / CONTINUATION / MODEL STATE / label definitions as Gemma's prompt, one bullet per call).
Run locally: with-local-keys python scripts/bon_sonnet_check.py --pool <pool parquet> --verify-dir <dir> --out <json>"""
import argparse, collections, concurrent.futures as cf, glob, json, os, random, re, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from bon_distill import VERIFY_SYS, verify_msgs

SYS1 = VERIFY_SYS.split("For every U or C claim")[0] + 'Answer with JSON only: {"label": "S"|"U"|"C"} for the ONE claim given.'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pool", required=True); ap.add_argument("--verify-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--model", default="claude-sonnet-5")
    a = ap.parse_args()
    import anthropic, pyarrow.parquet as pq
    hdr = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else {}
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=hdr)
    P = {r["anchor_id"]: r for r in pq.read_table(a.pool, columns=["anchor_id", "prefix_text", "cont_text", "top10_tokens", "top10_probs", "entropy", "jlens_tokens"]).to_pylist()}
    items = []
    for f in sorted(glob.glob(f"{a.verify_dir}/*.parquet")):
        for r in pq.read_table(f, columns=["anchor_id", "verdicts"]).to_pylist():
            for v in json.loads(r["verdicts"]):
                if v.get("label"): items.append((r["anchor_id"], v["bullet"], v["label"]))
    rng = random.Random(0); by = collections.defaultdict(list)
    for x in items: by[x[2]].append(x)
    per = a.n // 3; pick = []
    for lab in ("S", "U", "C"): pick += rng.sample(by[lab], min(per, len(by[lab])))
    if len(pick) < a.n: pick += rng.sample([x for x in items if x not in pick], min(a.n - len(pick), len(items) - len(pick)))
    def one(x):
        aid, b, g = x; msgs = verify_msgs(P[aid], [b]); user = msgs[1]["content"].replace("CLAIMS:\n0. ", "CLAIM:\n")
        for k in range(5):
            try:
                m = client.messages.create(model=a.model, max_tokens=50, system=[{"type": "text", "text": SYS1, "cache_control": {"type": "ephemeral"}}], messages=[{"role": "user", "content": user}])
                t = "".join(bl.text for bl in m.content if getattr(bl, "type", None) == "text"); mm = re.search(r'"label"\s*:\s*"([SUC])"', t)
                return {"anchor_id": aid, "bullet": b, "gemma": g, "sonnet": mm.group(1) if mm else None}
            except Exception as e:
                time.sleep(2 ** k + random.random())
        return {"anchor_id": aid, "bullet": b, "gemma": g, "sonnet": None}
    with cf.ThreadPoolExecutor(a.workers) as ex: R = list(ex.map(one, pick))
    R = [r for r in R if r["sonnet"]]; n = len(R)
    agree3 = sum(r["gemma"] == r["sonnet"] for r in R) / max(n, 1); bin_ = lambda l: "S" if l == "S" else "N"
    agree2 = sum(bin_(r["gemma"]) == bin_(r["sonnet"]) for r in R) / max(n, 1)
    cm = collections.Counter((r["gemma"], r["sonnet"]) for r in R)
    po = agree2; pg = sum(bin_(r["gemma"]) == "S" for r in R) / max(n, 1); ps = sum(bin_(r["sonnet"]) == "S" for r in R) / max(n, 1); pe = pg * ps + (1 - pg) * (1 - ps)
    res = {"n": n, "agree_3way": agree3, "agree_supported_vs_not": agree2, "kappa_supported_vs_not": (po - pe) / max(1 - pe, 1e-9),
           "confusion_gemma_x_sonnet": {f"{g}->{s}": c for (g, s), c in sorted(cm.items())},
           "gemma_S_precision_vs_sonnet": sum(1 for r in R if r["gemma"] == "S" and r["sonnet"] == "S") / max(sum(1 for r in R if r["gemma"] == "S"), 1), "rows": R}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[sonnet-check] n {n}: 3-way agreement {agree3:.3f}, supported-vs-not {agree2:.3f} (kappa {res['kappa_supported_vs_not']:.3f}), Gemma-S confirmed by Sonnet {res['gemma_S_precision_vs_sonnet']:.3f}; {res['confusion_gemma_x_sonnet']}", flush=True)


if __name__ == "__main__":
    main()
