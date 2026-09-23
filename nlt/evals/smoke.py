"""Synthetic smoke test for the text-only evals: fabricates pairs/meta/docs in the infra #6 layout plus three z corpora
(clean / depth-leaking / copying+tagged) and checks each metric moves the right way. No GPU, no API.
  python -m nlt.evals.smoke [--out-dir /tmp/nlt_smoke]
"""
from __future__ import annotations
import os, json, argparse
import numpy as np, pandas as pd
from nlt.evals.common import encode, decode, PrefixStore, join_pairs, save_table
from nlt.evals import regex_tags, copy_rate, diversity, depth_clf, controls, run_text_evals

TOPICS = ["the capital city", "a chemical element", "a football club", "the author's mother", "a Python function", "the year of the treaty", "a musical key", "the recipe's next step"]
CLEAN = ["The model settles on {t} as the referent and drops the alternative reading.", "Attention to the earlier mention of {t} resolves the pronoun; the prediction now favours a proper noun.",
         "The representation of {t} sharpens while the generic continuation fades.", "A number is now expected; {t} is treated as the quantity being asked for.",
         "The clause is parsed as a question about {t}, and the answer candidate is promoted."]
LEAK_EARLY = ["Still parsing the syntax around {t}; nothing has been decided yet.", "The passage is only beginning to be read; {t} is barely registered."]
LEAK_LATE = ["The model is nearly ready to answer: {t} is committed to as the next token.", "Final commitment to {t}; the output is essentially fixed."]
TAGGED = ["Between layer {i} and layer {j} the model resolves {t}.", "At L{j} the residual stream encodes {t}; depth {j} commits."]


def fabricate(out_dir: str, n_docs: int = 120, n_pos: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed); os.makedirs(out_dir, exist_ok=True)
    docs = []
    for d in range(n_docs):
        words = rng.choice(["the", "council", "met", "in", "Paris", "and", "agreed", "that", "sodium", "reacts", "with", "water", "while", "Arsenal", "played", "on", "Sunday", "her", "mother", "said", "def", "parse", "(", "s", ")", "1848", "treaty", "signed"], size=120)
        text = " ".join(words); docs.append(dict(doc_id=d, source="synthetic", text=text, token_ids=encode(text)))
    docs = pd.DataFrame(docs)
    meta = []
    for p in range(n_pos):
        d = int(rng.integers(n_docs)); L = len(docs.token_ids[d]); pos = int(rng.integers(20, L - 2))
        meta.append(dict(pos_idx=p, doc_id=d, pos=pos, token_id=int(docs.token_ids[d][pos]), next_token_id=int(docs.token_ids[d][pos + 1]), source="synthetic"))
    meta = pd.DataFrame(meta)
    pairs = []
    for k in range(3 * n_pos):
        p = int(rng.integers(n_pos)); j = int(rng.integers(10, 35)); i = int(rng.integers(9, j))
        pairs.append(dict(pair_id=f"val:{p}:{i}:{j}", split="val", pos_idx=p, i=i, j=j, doc_id=int(meta.doc_id[p])))
    pairs = pd.DataFrame(pairs).drop_duplicates("pair_id")
    ps = PrefixStore.from_infra(meta, docs)
    z_clean, z_leak, z_bad = [], [], []
    for r in pairs.itertuples():
        t = TOPICS[int(rng.integers(len(TOPICS)))]
        z_clean.append(dict(pair_id=r.pair_id, text=CLEAN[int(rng.integers(len(CLEAN)))].format(t=t)))
        pool = LEAK_LATE if r.j >= 24 else LEAK_EARLY
        z_leak.append(dict(pair_id=r.pair_id, text=pool[int(rng.integers(len(pool)))].format(t=t)))
        pre = ps.text(r.pos_idx, last_n=12)
        z_bad.append(dict(pair_id=r.pair_id, text=(TAGGED[int(rng.integers(2))].format(t=t, i=r.i, j=r.j) + " " + pre) if rng.random() < 0.5 else pre))
    for name, z in (("clean", z_clean), ("leak", z_leak), ("bad", z_bad)): save_table(pd.DataFrame(z), f"{out_dir}/z_{name}.parquet")
    save_table(docs, f"{out_dir}/docs.parquet"); save_table(meta, f"{out_dir}/meta.parquet"); save_table(pairs, f"{out_dir}/pairs_val.parquet")
    return docs, meta, pairs, {"clean": pd.DataFrame(z_clean), "leak": pd.DataFrame(z_leak), "bad": pd.DataFrame(z_bad)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", default="/tmp/nlt_smoke"); a = ap.parse_args()
    docs, meta, pairs, zs = fabricate(a.out_dir)
    res = {k: run_text_evals.run(z, pairs, meta, docs, tag=k) for k, z in zs.items()}
    for k, r in res.items():
        print(f"[{k:5s}] verdicts {r['verdicts']} | hard/1000 {r['regex']['hard_hits_per_1000_z']:.1f} | copy4 {r['copy']['copy_rate_4gram_mean']:.3f} lcs95 {r['copy']['lcs_tokens_p95']:.0f} | "
              f"MI(z;j) {r['depth']['j']['mi_bits']:.2f} bits, gap ratio {r['depth']['gap']['ratio']:.2f} | median tok {r['diversity']['tokens_median']:.0f} distinct4 {r['diversity']['distinct_4gram_ratio']:.2f} selfBLEU {r['diversity']['self_bleu4']:.2f}")
    # expectations
    assert res["bad"]["regex"]["hard_hits_per_1000_z"] > 100 and res["clean"]["regex"]["hard_hits_per_1000_z"] <= 1, "regex tier"
    assert res["bad"]["copy"]["copy_rate_4gram_mean"] > 0.3 and res["clean"]["copy"]["copy_rate_4gram_mean"] < 0.05, "copy rate"
    assert res["leak"]["depth"]["j"]["mi_bits"] > res["clean"]["depth"]["j"]["mi_bits"] + 0.3, "depth clf should see the planted leak"
    # manifest
    m = controls.build_manifest(pairs, zs["clean"], PrefixStore.from_infra(meta, docs)); assert set(m.variant) == {"orig", "empty", "dm", "rp", "copy", "wrong_j", "wrong_i", "shuf_words"}, set(m.variant)
    dm = m[m.variant == "dm"].merge(pairs, on="pair_id"); src = pairs.set_index("pair_id")
    assert all(src.loc[s, "j"] == j for s, j in zip(dm.src_pair_id, dm.j)), "dm partner must share j"
    # fake scores: orig +20 bits, dm +5, rp 0, copy +1, wrong_j -10, wrong_i +8 -> verdict logic
    rng = np.random.default_rng(0); base = {"orig": 20, "dm": 5, "rp": 0, "copy": 1, "wrong_j": -10, "wrong_i": 8, "empty": 0, "shuf_words": 6}
    m["logp"] = [np.log(2) * (base[v] + rng.normal(0, 1)) for v in m.variant]
    s = controls.summarize_scores(m); print("manifest verdicts:", {k: v for k, v in s.items() if k.startswith("verdict")})
    assert s["verdict_3a"] == "PASS" and s["verdict_3c"] == "PASS" and s["verdict_3d"] == "PASS" and s["verdict_5c"] == "PASS", s
    json.dump({k: {kk: vv for kk, vv in r.items() if kk != "depth"} for k, r in res.items()}, open(f"{a.out_dir}/smoke_results.json", "w"), indent=1, default=str)
    print("SMOKE OK ->", a.out_dir)


if __name__ == "__main__":
    main()
