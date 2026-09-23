"""Synthetic claim data, last step: merge the three claim families per anchor, remove near-duplicates, cap over-frequent claims, write the
training files + stats.

  python scripts/claims_finalize.py --root /vol_glp/claims
reads  {root}/anchors/anchors_<name>.parquet (activation_vector, anchor_id, is_val, source, ...) and {root}/claims/{internal,text,semantic}_<name>.parquet
writes {root}/final/final_<name>.parquet: anchor_id, doc_id, source, is_val, n_raw_tokens, claims, families, types, activation_vector (anchors order)
       {root}/final/stats.json + {root}/final/examples.json (50 random anchors with all their claims)

Near-duplicates: sentence embeddings (all-MiniLM-L6-v2, mean-pooled, unit norm); within an anchor, a claim whose cosine to an earlier kept
claim exceeds --dup-cos (0.95) is dropped (paraphrases are exempt: they are meant to say the same thing in other words, but must still differ
from it textually). Across anchors: the near-duplicate rate is MEASURED on a random sample (share of sampled claims with a > 0.95 neighbour
from another anchor; templated claims repeat by design — the same claim true of many activations is exactly the label-sharing we want), and
exact repeats of SEMANTIC claims are CAPPED (generic labels such as "Language: English"): a normalised semantic claim string occurring in more
than --cap-frac of all anchors is kept with probability cap / count. Templated families are never capped (their buckets are the labels).
False twins: every kept claim gets a minimal false version (semantic: the writer's F field; templated: the quoted slot swapped for another
anchor's value of the same type, bucketed types flipped to a far bucket, else nla.flow.negatives.make_negative) -> column "twins"."""
import argparse, collections, glob, json, os, random, re, sys, time
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAMILIES = ("internal", "text", "semantic")


def _norm(s): return re.sub(r"\s+", " ", s.lower()).strip(" .")


# ---------------------------------------------------------------- false twins (minimal swaps that make a claim false for its anchor)
QSLOT = re.compile(r"(?:^|(?<=[\s(:]))(?:'([^'\n]{1,120})'|\"([^\"\n]{1,120})\")(?=$|[\s.,;:!?)])")   # a quoted slot, not an apostrophe
BUCKETS = {   # no-slot templated types: phrase -> bucket; the twin is a phrase of the same type from a DIFFERENT (preferably far) bucket
    "entropy": [("almost no uncertainty|very sure|highly predictable", 0), ("fairly sure|fairly predictable|only a few", 1),
                ("the model is unsure|several continuations|hard to predict", 2), ("very unsure|many different|highly unpredictable", 3)],
    "position": [("fewer than 64|few dozen", 0), ("near the start|early in", 1), ("few paragraphs|tokens of the document have been read", 2), ("deep into|long stretch", 3)],
    "sentence_end": [("ends within|about to end", 0), ("keeps going|does not end", 1)],
    "speaker": [("user", 0), ("assistant", 1)],
}


def _bucket(ctype, claim):
    for pat, b in BUCKETS.get(ctype, []):
        if re.search(pat, claim.lower()): return b
    return None


class Twins:
    """donor pools per family:type (slot values and whole claims from other anchors) -> twin(claim, family, type, own_claims)"""
    def __init__(self, rng, cap=4000):
        self.rng, self.cap = rng, cap; self.slots = collections.defaultdict(list); self.whole = collections.defaultdict(list); self.pool = []
        self.byb = collections.defaultdict(lambda: collections.defaultdict(list))   # family:type -> bucket -> donor claims (bucketed types)

    def add(self, claims, fams, types):
        for c, f, t in zip(claims, fams, types):
            k = f"{f}:{t.split('/')[0]}"; m = QSLOT.search(c)
            if m: self._push(self.slots[k], m.group(1) or m.group(2))
            self._push(self.whole[k], c)
            self._push(self.pool, c)
            b = _bucket(t.split("/")[0], c)
            if b is not None: self._push(self.byb[k][b], c)

    def _push(self, lst, x):
        if len(lst) < self.cap: lst.append(x)
        else: lst[self.rng.randrange(self.cap)] = x

    def twin(self, c, f, t, own):
        from nla.flow.negatives import make_negative
        base = t.split("/")[0]; k = f"{f}:{base}"; ownn = {_norm(x) for x in own}
        if f in ("internal", "text"):
            m = QSLOT.search(c)
            if m and self.slots[k]:
                val = m.group(1) or m.group(2)
                for _ in range(8):
                    d = self.rng.choice(self.slots[k])
                    if d.strip().lower() != val.strip().lower():
                        tw = c[: m.start()] + (f"'{d}'" if m.group(1) is not None else f'"{d}"') + c[m.end():]
                        if _norm(tw) not in ownn: return tw
            b = _bucket(base, c)
            if b is not None:
                far = [bb for bb in self.byb[k] if abs(bb - b) >= 2 and self.byb[k][bb]]; other = [bb for bb in self.byb[k] if bb != b and self.byb[k][bb]]
                if far or other: return self.rng.choice(self.byb[k][self.rng.choice(far or other)])
            if base in BUCKETS: return None                                            # bucketed type without a far bucket in the pool
            if self.whole[k]:                                                          # same claim type, another anchor (concept lists, next sentence, language, ...)
                for _ in range(8):
                    d = self.rng.choice(self.whole[k])
                    if _norm(d) not in ownn and _norm(d) != _norm(c): return d
        tw, _ = make_negative(c, self.rng, self.pool or [c])
        return tw if tw and _norm(tw) not in ownn else None


class Embedder:
    def __init__(self, name="sentence-transformers/all-MiniLM-L6-v2", dev="cuda"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.dev = torch, dev if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(name); self.m = AutoModel.from_pretrained(name).to(self.dev).eval().half() if self.dev == "cuda" else AutoModel.from_pretrained(name).eval()

    def __call__(self, texts, bs=2048):
        torch = self.torch; out = []
        for i in range(0, len(texts), bs):
            enc = self.tok(texts[i: i + bs], padding=True, truncation=True, max_length=64, return_tensors="pt").to(self.dev)
            with torch.no_grad():
                h = self.m(**enc).last_hidden_state.float(); m = enc["attention_mask"].unsqueeze(-1).float()
                e = (h * m).sum(1) / m.sum(1).clamp_min(1); out.append(torch.nn.functional.normalize(e, dim=-1).cpu())
        return torch.cat(out) if out else torch.zeros(0, 384)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="/vol_glp/claims"); ap.add_argument("--dup-cos", type=float, default=0.95)
    ap.add_argument("--cap-frac", type=float, default=0.005); ap.add_argument("--sample", type=int, default=50000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-claims", type=int, default=2)
    ap.add_argument("--names", default="*", help="glob over anchor-file names (e.g. 'v2_*'): finalize only these (streaming shards)")
    ap.add_argument("--skip-done", action="store_true", help="skip anchor files whose final_<name>.parquet exists")
    ap.add_argument("--stats-tag", default="", help="write stats_<tag>.json / examples_<tag>.json (stats.json is written only when absent or tag empty)")
    a = ap.parse_args(); rng = random.Random(a.seed); t0 = time.time()
    names = sorted(os.path.basename(f)[8:-8] for f in glob.glob(f"{a.root}/anchors/anchors_{a.names}.parquet"))
    if a.skip_done: names = [n for n in names if not os.path.exists(f"{a.root}/final/final_{n}.parquet")]
    if not names: print("[final] nothing to do", flush=True); return
    fam = {f: {} for f in FAMILIES}
    for f in FAMILIES:
        for p in [x for n in names for x in (f"{a.root}/claims/{f}_{n}.parquet", f"{a.root}/semantic/{f}_{n}.parquet", f"{a.root}/claims/{f}_sonnet_{n}.parquet",
                                             f"{a.root}/gemma/{f}_{n}.parquet") if os.path.exists(x)]:
            cols = ["anchor_id", "claims", "types"] + (["twins"] if "twins" in pq.read_schema(p).names else [])
            for r in pq.read_table(p, columns=cols).to_pylist():
                r["twins"] = r.get("twins") or [None] * len(r["claims"]); q = fam[f].get(r["anchor_id"])
                if q is None: fam[f][r["anchor_id"]] = r
                else: q["claims"] = q["claims"] + r["claims"]; q["types"] = q["types"] + r["types"]; q["twins"] = q["twins"] + r["twins"]   # several sources of one family
    print(f"[final] {len(names)} anchor files; claims for {[len(fam[f]) for f in FAMILIES]} anchors per family ({time.time() - t0:.0f}s)", flush=True)
    # pass 1: pool + exact-repeat counts (normalised strings, counted once per anchor)
    meta = {}
    cnt = collections.Counter()
    for n in names:
        ids = pq.read_table(f"{a.root}/anchors/anchors_{n}.parquet", columns=["anchor_id"]).column(0).to_pylist()
        for aid in ids:
            pool = []
            for f in FAMILIES:
                r = fam[f].get(aid)
                if r: pool += [(c, f, t, tw) for c, t, tw in zip(r["claims"], r["types"], r.get("twins") or [None] * len(r["claims"])) if c and len(c.split()) >= 2]
            meta[aid] = pool; cnt.update({_norm(c) for c, f, _, _ in pool if f == "semantic"})
    n_anchor = len(meta); cap = max(1, int(a.cap_frac * n_anchor))
    over = {k: v for k, v in cnt.items() if v > cap}
    print(f"[final] {n_anchor} anchors, {sum(len(v) for v in meta.values())} pooled claims; {len(over)} claim strings above the cap ({cap} anchors) e.g. "
          f"{sorted(over.items(), key=lambda x: -x[1])[:5]}", flush=True)
    emb = Embedder(); st = collections.Counter(); fam_ct = collections.Counter(); type_ct = collections.Counter(); per_anchor = []; tw_ct = collections.Counter()
    twins = Twins(random.Random(a.seed + 1)); src_fam = collections.defaultdict(lambda: collections.Counter()); src_n = collections.Counter()
    words = collections.defaultdict(lambda: collections.Counter())
    for aid in rng.sample(list(meta), min(len(meta), 20000)):                        # donor pools from a random sample of anchors
        twins.add([c for c, _, _, _ in meta[aid]], [f for _, f, _, _ in meta[aid]], [t for _, _, t, _ in meta[aid]])
    os.makedirs(f"{a.root}/final", exist_ok=True); examples = []; samp_txt, samp_aid = [], []
    for n in names:
        tbl = pq.read_table(f"{a.root}/anchors/anchors_{n}.parquet", columns=["anchor_id", "doc_id", "source", "is_val", "n_raw_tokens", "prefix_text", "cont_text", "activation_vector"])
        ids = tbl.column("anchor_id").to_pylist(); flat = [(i, c, f, t) for i, aid in enumerate(ids) for c, f, t, _ in meta[aid]]
        twin_given = {(i, c): tw for i, aid in enumerate(ids) for c, _, _, tw in meta[aid] if tw}
        E = emb([c for _, c, _, _ in flat]); keep_rows = [[] for _ in ids]; start = 0
        by = collections.defaultdict(list)
        for k, (i, c, f, t) in enumerate(flat): by[i].append(k)
        for i, ks in by.items():
            kept = []
            for k in ks:
                _, c, f, t = flat[k]; st["pooled"] += 1; nc = _norm(c)
                if f == "semantic" and nc in over and rng.random() > cap / over[nc]: st["capped"] += 1; continue
                if any(_norm(flat[j][1]) == nc for j in kept): st["exact_dup"] += 1; continue
                if not t.endswith("/paraphrase") and kept and float((E[kept] @ E[k]).max()) > a.dup_cos: st["near_dup"] += 1; continue
                kept.append(k)
            own = [flat[k][1] for k in kept]; kr = []
            for k in kept:
                _, c, f, t = flat[k]; tw = twin_given.get((i, c)) or twins.twin(c, f, t, own)
                kr.append((c, f, t, tw)); tw_ct[f"{f}:{'yes' if tw else 'no'}"] += 1
            keep_rows[i] = kr
        ok = [i for i in range(len(ids)) if len(keep_rows[i]) >= a.min_claims]; st["anchors_dropped_few_claims"] += len(ids) - len(ok)
        src_all = tbl.column("source").to_pylist()
        for i in ok:
            per_anchor.append(len(keep_rows[i]))
            src_n[src_all[i]] += 1
            for c, f, t, _ in keep_rows[i]:
                fam_ct[f] += 1; type_ct[f"{f}:{t.split('/')[0]}"] += 1; src_fam[src_all[i]][f] += 1; words[f][min(len(c.split()), 40)] += 1
            if rng.random() < 0.02: samp_txt += [c for c, _, _, _ in keep_rows[i]]; samp_aid += [ids[i]] * len(keep_rows[i])
        out = tbl.take(ok).drop_columns(["prefix_text", "cont_text"])
        out = out.append_column("claims", pa.array([[c for c, _, _, _ in keep_rows[i]] for i in ok], pa.list_(pa.string())))
        out = out.append_column("families", pa.array([[f for _, f, _, _ in keep_rows[i]] for i in ok], pa.list_(pa.string())))
        out = out.append_column("types", pa.array([[t for _, _, t, _ in keep_rows[i]] for i in ok], pa.list_(pa.string())))
        out = out.append_column("twins", pa.array([[tw for _, _, _, tw in keep_rows[i]] for i in ok], pa.list_(pa.string())))
        tmp = f"{a.root}/final/final_{n}.parquet.tmp"; pq.write_table(out, tmp, compression="zstd"); os.replace(tmp, f"{a.root}/final/final_{n}.parquet")
        pt, ct = tbl.column("prefix_text").to_pylist(), tbl.column("cont_text").to_pylist(); src = tbl.column("source").to_pylist()
        for i in rng.sample(ok, min(len(ok), 3)):
            examples.append({"anchor_id": ids[i], "source": src[i], "prefix_tail": pt[i][-600:], "continuation": ct[i], "claims": [{"family": f, "type": t, "claim": c, "false_twin": tw} for c, f, t, tw in keep_rows[i]]})
        print(f"[final] {n}: {len(ok)}/{len(ids)} anchors kept ({time.time() - t0:.0f}s)", flush=True)
    # cross-anchor near-duplicate rate on a sample
    import torch
    idx = rng.sample(range(len(samp_txt)), min(a.sample, len(samp_txt))); S = emb([samp_txt[i] for i in idx]).to(emb.dev); aids = [samp_aid[i] for i in idx]
    has = torch.zeros(len(idx), dtype=torch.bool); A = np.array([hash(x) for x in aids]); At = torch.tensor(A, device=emb.dev)
    for s in range(0, len(idx), 4096):
        sim = S[s: s + 4096] @ S.T; sim[At[s: s + 4096, None] == At[None, :]] = -1; has[s: s + 4096] = (sim > a.dup_cos).any(1).cpu()
    pa_ = np.array(per_anchor)
    stats = {"anchors_in": n_anchor, "anchors_kept": int(len(pa_)), "claims_kept": int(pa_.sum()), "claims_per_anchor_mean": float(pa_.mean()),
             "claims_per_anchor_pct10_50_90": np.percentile(pa_, [10, 50, 90]).tolist(), "filter_counts": dict(st), "cap_anchors": cap,
             "dedupe_rate_within_anchor": (st["near_dup"] + st["exact_dup"]) / max(st["pooled"], 1), "capped_rate": st["capped"] / max(st["pooled"], 1),
             "cross_anchor_near_dup_rate_sample": float(has.float().mean()), "cross_anchor_sample_n": len(idx),
             "claims_per_family": dict(fam_ct), "claims_per_type": dict(type_ct.most_common()), "twin_coverage": dict(tw_ct),
             "anchors_per_source": dict(src_n), "claims_per_anchor_by_source_family": {s_: {f: v / src_n[s_] for f, v in c_.items()} for s_, c_ in src_fam.items()},
             "words_per_claim_hist": {f: {str(k): v for k, v in sorted(c_.items())} for f, c_ in words.items()}, "top_repeated_claims": sorted(over.items(), key=lambda x: -x[1])[:30]}
    sfx = f"_{a.stats_tag}" if a.stats_tag else ""
    json.dump(stats, open(f"{a.root}/final/stats{sfx}.json", "w"), indent=1); json.dump(rng.sample(examples, min(50, len(examples))), open(f"{a.root}/final/examples{sfx}.json", "w"), indent=1)
    if sfx and not os.path.exists(f"{a.root}/final/stats.json"): json.dump(stats, open(f"{a.root}/final/stats.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k not in ("claims_per_type", "top_repeated_claims")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
