"""Categorise bullets (cruxy vs not) by claim TYPE with Sonnet 5 (local box, `with-local-keys python3 -m nlt.bullets.categorize`).
Input: per_bullet.parquet from nlt.bullets.evaluate (columns bullet, cruxy, d_loo, ...). Output: <out>.json with per-bullet labels,
the type distribution for cruxy vs non-cruxy bullets, the cruxy rate per type, and 10 examples of each.

Types: next_token (a specific candidate continuation promoted / demoted), topic (topic, genre, register, domain, language),
syntax (grammatical role, clause / sentence position, punctuation, formatting), entity (entity, reference, binding, co-reference,
attribute of a named thing), route (mechanism talk: attention, retrieval, copying, induction, MLP, 'tracking'), magnitude
(confidence sharpened / broadened / hedged, uncertainty), other.
"""
from __future__ import annotations
import argparse, asyncio, json, os, random, re, sys, time
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nlt.proposers.teacher_sonnet import client_kwargs  # noqa: E402

TYPES = ["next_token", "topic", "syntax", "entity", "route", "magnitude", "other"]
SYSTEM = """You label short claims about what a language model worked out while reading a passage. For each claim pick exactly ONE type:
- next_token: the claim is about a specific candidate continuation or set of candidates being promoted, demoted, expected or ruled out (naming tokens or describing what word class comes next).
- topic: the claim is about the topic, genre, register, domain, language, style or overall structure of the text being recognised.
- syntax: the claim is about grammar, the grammatical role of the current word, clause or sentence position, punctuation, list / code / formatting structure.
- entity: the claim is about a specific entity, referent, binding, co-reference, an attribute or relation of a named thing.
- route: the claim is about the mechanism (attention, retrieval, copying, induction, tracking a variable, integrating context).
- magnitude: the claim is only about confidence sharpening, broadening, hedging, uncertainty or commitment strength, without new content.
- other: anything else.
Answer with JSON only: {"labels": ["<type for claim 0>", "<type for claim 1>", ...]} with exactly one label per claim, in order."""


async def _one(client, sem, key, prm, out, retries=12):
    async with sem:
        for a in range(retries):
            try:
                msg = await client.messages.create(**prm)
                txt = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
                if not txt.strip() and a < retries - 1: print(f"[cat] empty response for chunk {key} (stop {msg.stop_reason}), retrying", flush=True); continue
                out[key] = txt; return
            except Exception as e:
                if a == retries - 1: out[key] = None; print(f"[cat] giving up {key}: {str(e)[:100]}", flush=True); return
                await asyncio.sleep(min(60, 2 ** a) + random.random())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bullet", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1200, help="max bullets to label (balanced cruxy / non-cruxy)"); ap.add_argument("--chunk", type=int, default=25); ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); rng = np.random.default_rng(a.seed)
    df = pq.read_table(a.per_bullet).to_pandas()
    cx = df[df["cruxy"]]; nc = df[~df["cruxy"]]
    k = min(a.n // 2, len(cx), len(nc))
    sub = pd.concat([cx.sample(n=k, random_state=a.seed), nc.sample(n=k, random_state=a.seed)]).reset_index(drop=True)
    print(f"[cat] labelling {len(sub)} bullets ({k} cruxy + {k} non-cruxy of {len(cx)} / {len(nc)})", flush=True)
    import anthropic
    client = anthropic.AsyncAnthropic(**client_kwargs()); out = {}
    items = []
    for s in range(0, len(sub), a.chunk):
        claims = "\n".join(f"[{i}] {b}" for i, b in enumerate(sub["bullet"].values[s:s + a.chunk]))
        prm = dict(model="claude-sonnet-5", max_tokens=600, system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}], messages=[{"role": "user", "content": f"Claims:\n{claims}\n\nJSON only."}])
        items.append((s, prm))

    async def run():
        sem = asyncio.Semaphore(a.concurrency)
        await asyncio.gather(*[_one(client, sem, s, p, out) for s, p in items])
    t0 = time.time(); asyncio.run(run()); print(f"[cat] {sum(1 for s, _ in items if not out.get(s))} of {len(items)} chunks failed", flush=True)
    labels = np.array(["unlabelled"] * len(sub), dtype=object)
    for s, _ in items:
        raw = out.get(s)
        if not raw: continue
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            lab = json.loads(m.group(0))["labels"]
        except Exception:
            print(f"[cat] unparsable chunk {s}: {raw[:200]!r}", flush=True); continue
        for i, l in enumerate(lab[: a.chunk]):
            if s + i < len(sub): labels[s + i] = l if l in TYPES else "other"
    sub["type"] = labels; sub = sub[sub["type"] != "unlabelled"]
    dist = {t: {"cruxy": int(((sub["type"] == t) & sub["cruxy"]).sum()), "non_cruxy": int(((sub["type"] == t) & ~sub["cruxy"]).sum())} for t in TYPES}
    for t in TYPES:
        n = dist[t]["cruxy"] + dist[t]["non_cruxy"]; dist[t]["n"] = n; dist[t]["cruxy_rate_in_balanced_sample"] = dist[t]["cruxy"] / n if n else None
        dist[t]["d_loo_mean"] = float(sub.loc[sub["type"] == t, "d_loo"].mean()) if n else None
        dist[t]["share_of_cruxy"] = dist[t]["cruxy"] / max(1, int(sub["cruxy"].sum())); dist[t]["share_of_non_cruxy"] = dist[t]["non_cruxy"] / max(1, int((~sub["cruxy"]).sum()))
    ex = lambda m: [{"pair_id": r["pair_id"], "bullet": r["bullet"], "d_loo": float(r["d_loo"]), "d_swap_slot": float(r["d_swap_slot"]), "type": r["type"]} for r in sub[m].sort_values("d_loo", ascending=False).head(10).to_dict("records")]
    res = {"n_labelled": int(len(sub)), "n_cruxy_total": int(len(cx)), "n_non_cruxy_total": int(len(nc)), "types": dist,
           "examples_cruxy": ex(sub["cruxy"]), "examples_non_cruxy": [{"pair_id": r["pair_id"], "bullet": r["bullet"], "d_loo": float(r["d_loo"]), "d_swap_slot": float(r["d_swap_slot"]), "type": r["type"]} for r in sub[~sub["cruxy"]].sort_values("d_loo").head(10).to_dict("records")],
           "seconds": time.time() - t0}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
    sub.to_parquet(a.out.replace(".json", ".parquet"))
    print(json.dumps({t: (d["cruxy"], d["non_cruxy"]) for t, d in dist.items()}), flush=True)


if __name__ == "__main__":
    main()
