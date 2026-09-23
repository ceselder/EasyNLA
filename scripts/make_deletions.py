"""Deletion test inputs ("are fabrications neutral?"): for 120 sampled explanations with >= 2 judge-flagged false claims and >= 2 supported claims,
Sonnet 5 writes (a) the explanation with ALL flagged false claims removed and nothing else changed, (b) the explanation with the SAME NUMBER of
supported claims removed (control). Run under with-local-keys.
  python scripts/make_deletions.py --dir <report>/data/flow_noise --n 120
"""
import argparse, concurrent.futures as cf, json, os, random, re

SYS = """You edit an explanation of a language model's hidden state. You get the explanation and a list of its specific claims, each marked TRUE or FALSE.
Produce two edited versions:
(a) "remove_false": delete EVERY claim marked FALSE — remove the words/clauses that state it, keep everything else verbatim (fix only grammar/punctuation broken by the deletion; do not add or reword anything else).
(b) "remove_true": delete exactly the TRUE claims listed in `remove_true_claims` in the same way, keep everything else (including the FALSE claims) verbatim.
Return ONLY JSON: {"remove_false": str, "remove_true": str}"""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--n", type=int, default=120); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    import anthropic
    gen = json.load(open(f"{a.dir}/gen.json")); J = json.load(open(f"{a.dir}/judge.json")); rng = random.Random(a.seed)
    pool = []
    for av, v in gen["avs"].items():
        for g, row in enumerate(gen["rows"]):
            for i, z in enumerate(v["explanations"][g]):
                r = J.get(f"{av}-{g}-{i}")
                if not r: continue
                F = [c["claim"] for c in r["claims"] if c["verdict"] != "supported"]; T = [c["claim"] for c in r["claims"] if c["verdict"] == "supported"]
                if len(F) >= 2 and len(T) >= 2: pool.append(dict(av=av, g=g, i=i, row=row, z=z, false=F, true=T))
    rng.shuffle(pool); pick = pool[: a.n]
    hdr = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else {}
    cl = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=hdr, max_retries=8)

    def work(it):
        k = min(len(it["false"]), len(it["true"])); rt = rng.sample(it["true"], k) if len(it["true"]) > k else it["true"]
        claims = "\n".join([f"- FALSE: {c}" for c in it["false"]] + [f"- TRUE: {c}" for c in it["true"]])
        msg = f"EXPLANATION:\n<<<\n{it['z']}\n>>>\n\nCLAIMS:\n{claims}\n\nremove_true_claims:\n" + "\n".join(f"- {c}" for c in rt)
        for _ in range(3):
            try:
                m = cl.messages.create(model="claude-sonnet-5", max_tokens=6000, system=SYS, messages=[{"role": "user", "content": msg}])
                txt = "".join(b.text for b in m.content if getattr(b, "type", None) == "text"); j = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
                return dict(it, remove_false=j["remove_false"], remove_true=j["remove_true"], n_removed=k, n_false=len(it["false"]))
            except Exception as e: err = str(e)[:100]
        return dict(it, error=err)
    with cf.ThreadPoolExecutor(16) as ex: res = list(ex.map(work, pick))
    ok = [r for r in res if "remove_false" in r]
    json.dump({"items": ok, "pool_size": len(pool)}, open(f"{a.dir}/deletions.json", "w"), indent=1)
    print(f"pool {len(pool)}; wrote {len(ok)} deletion items; mean words orig/rmF/rmT",
          round(sum(len(r['z'].split()) for r in ok) / len(ok), 1), round(sum(len(r['remove_false'].split()) for r in ok) / len(ok), 1), round(sum(len(r['remove_true'].split()) for r in ok) / len(ok), 1))


if __name__ == "__main__":
    main()
