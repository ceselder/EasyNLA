"""Decodability test 2, data build (CPU) — (activation, candidate detail) pairs by detail type and token distance.

From the Qwen3.6-27B layer-42 extraction shards (each row: `text` = the context up to and including the extraction position, `activation_vector`
= h at the last token) we collect every NUMBER (>= 2 digits, the wrong-detail generator's definition), proper NAME (2-4 capitalised words) and
QUOTED span (12-120 chars) in the context, with its token distance k = (position index) − (index of the detail's last token): k = 0 means the
detail ends AT the read-out token, k = 1 one token earlier, ... Buckets: 0 | 1 | 2-4 | 5-16 | 17-64 | 65-256. For every positive we draw a
matched negative of the same type from ANOTHER document (not a substring of this context), and for numbers also a near-miss (10-40 % off,
years +-1..30) and a far (x3+7) perturbation — the controlled number test's edits. Documents are hash-split 80/20 (train / held-out test).
Output: one torch file with fp16 activations + a row table; scripts/decodability_probe_train.py fits the probes.
usage (Modal CPU): python scripts/decodability_probe_build.py --out /vol_glp/decodability/probe_data.pt
"""
from __future__ import annotations
import argparse, bisect, glob, hashlib, json, os, random, re, sys, time
import numpy as np, torch, pyarrow as pa, pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from nla.flow.negatives import NAME, QUOTE
from nla.flow.halluc_classify import NUM, perturb

BUCKETS = [(0, 0, "0"), (1, 1, "1"), (2, 4, "2-4"), (5, 16, "5-16"), (17, 64, "17-64"), (65, 256, "65-256")]


def bucket_of(k):
    for lo, hi, nm in BUCKETS:
        if lo <= k <= hi: return nm
    return None


def candidates(text):
    """-> list of (type, value, char_end) for the LAST occurrence of each distinct value (the nearest one to the read-out position)."""
    out = {}
    for m in NUM.finditer(text):
        if len(m.group(1).replace(",", "")) >= 2 or m.group(2): out[("number", m.group(0))] = m.end()
    for m in NAME.finditer(text): out[("name", m.group(1))] = m.end(1)
    for m in QUOTE.finditer(text): out[("quote", m.group(1))] = m.end(1)
    return [(t, v, e) for (t, v), e in out.items()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True); p.add_argument("--base", default="Qwen/Qwen3.6-27B")
    p.add_argument("--g1-glob", default="/vol_glp/scale/g1/shards/shard_*.parquet"); p.add_argument("--acts-glob", default="/vol_q36/data/acts_qwen36_L42/shard_*.parquet")
    p.add_argument("--g1-shards", type=int, default=14); p.add_argument("--acts-shards", type=int, default=6)
    p.add_argument("--per-cell", type=int, default=6000, help="max examples kept per (type, bucket) before the doc split (~80 %% train / 20 %% test)")
    p.add_argument("--max-per-row", type=int, default=2, help="max candidates kept per (row, type) so no single context dominates a cell")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(); rng = random.Random(a.seed); t0 = time.time()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base, token=os.environ.get("HF_TOKEN"))
    shards = sorted(glob.glob(a.g1_glob))[: a.g1_shards] + sorted(glob.glob(a.acts_glob))[: a.acts_shards]
    print(f"[build] {len(shards)} shards", flush=True)

    cells = {}          # (type, bucket) -> reservoir of candidate dicts
    seen = {}           # (type, bucket) -> count seen (reservoir sampling)
    pool = {"number": [], "name": [], "quote": []}   # (value, doc) for matched negatives
    n_rows = 0; n_tok_mismatch = 0; ctx_len = []
    for sp in shards:
        pf = pq.ParquetFile(sp); t = pf.read(columns=["text", "doc_id", "n_raw_tokens"]); texts = t.column("text").to_pylist(); docs = t.column("doc_id").to_pylist(); nrt = t.column("n_raw_tokens").to_pylist()
        for s in range(0, len(texts), 2048):
            enc = tok(texts[s: s + 2048], add_special_tokens=False, return_offsets_mapping=True)
            for j in range(len(enc["input_ids"])):
                i = s + j; text = texts[i]; offs = enc["offset_mapping"][j]; n = len(offs)
                if n < 2: continue
                n_rows += 1; ctx_len.append(n)
                if abs(n - nrt[i]) > 3: n_tok_mismatch += 1
                starts = [o[0] for o in offs]
                cands = candidates(text); per_type = {}
                rng.shuffle(cands)
                for typ, val, end in cands:
                    if per_type.get(typ, 0) >= a.max_per_row: continue
                    last_tok = bisect.bisect_left(starts, end) - 1          # last token whose start is before the detail's char end
                    if last_tok < 0: continue
                    k = (n - 1) - last_tok; b = bucket_of(k)
                    if b is None: continue
                    per_type[typ] = per_type.get(typ, 0) + 1
                    key = (typ, b); seen[key] = seen.get(key, 0) + 1; rec = {"shard": sp, "row": i, "doc": docs[i], "type": typ, "value": val, "k": k, "bucket": b, "n_ctx": n}
                    res = cells.setdefault(key, [])
                    if len(res) < a.per_cell: res.append(rec)
                    else:
                        r_ = rng.randrange(seen[key])
                        if r_ < a.per_cell: res[r_] = rec
                    if len(pool[typ]) < 200000: pool[typ].append((val, docs[i]))
        print(f"[build] {os.path.basename(sp)}: rows {n_rows}, cells " + " ".join(f"{t_}/{b_}:{len(v)}" for (t_, b_), v in sorted(cells.items())) + f" ({time.time() - t0:.0f}s)", flush=True)
    print(f"[build] context tokens pcts {np.percentile(ctx_len, [5, 25, 50, 75, 95])}; tokenizer/n_raw_tokens mismatches (>3) {n_tok_mismatch}/{n_rows}", flush=True)

    # ---- doc split + matched negatives (+ near / far for numbers)
    def split_of(doc): return "test" if int(hashlib.sha256(doc.encode()).hexdigest()[:8], 16) % 5 == 0 else "train"
    rows = [r for v in cells.values() for r in v]
    by_shard = {}
    for r in rows: by_shard.setdefault(r["shard"], []).append(r)
    texts_by = {}
    for sp, rs in by_shard.items():
        idx = sorted({r["row"] for r in rs}); tt = pq.read_table(sp, columns=["text"]).take(pa.array(idx)).column(0).to_pylist(); texts_by[sp] = dict(zip(idx, tt))
    for r in rows:
        r["split"] = split_of(r["doc"]); text = texts_by[r["shard"]][r["row"]]; tl = text.lower().replace(",", "")
        for _ in range(50):
            v, d_ = pool[r["type"]][rng.randrange(len(pool[r["type"]]))]
            if d_ != r["doc"] and v != r["value"] and v.lower().replace(",", "") not in tl: r["neg"] = v; break
        if r["type"] == "number":
            try: r["near"] = perturb(r["value"], rng, "near"); r["far"] = perturb(r["value"], rng, "far")
            except Exception: r["near"] = None; r["far"] = None
            if r.get("near") and r["near"].lower().replace(",", "") in tl: r["near_in_ctx"] = True
    rows = [r for r in rows if r.get("neg")]
    # ---- activations of the selected rows
    H = torch.zeros(len(rows), 5120, dtype=torch.float16); pos = {}
    for i, r in enumerate(rows): pos.setdefault(r["shard"], []).append(i)
    for sp, ii in pos.items():
        idx = [rows[i]["row"] for i in ii]; order = np.argsort(idx); idx_sorted = [idx[o] for o in order]
        col = pq.read_table(sp, columns=["activation_vector"]).take(pa.array(idx_sorted)).column(0)
        arr = np.asarray(col.combine_chunks().flatten(), dtype=np.float32).reshape(len(idx_sorted), -1)
        for rank, o in enumerate(order): H[ii[o]] = torch.tensor(arr[rank]).half()
        print(f"[build] acts {os.path.basename(sp)}: {len(ii)} rows ({time.time() - t0:.0f}s)", flush=True)
    for r in rows: r.pop("shard", None)
    counts = {}
    for r in rows: counts[f"{r['type']}/{r['bucket']}/{r['split']}"] = counts.get(f"{r['type']}/{r['bucket']}/{r['split']}", 0) + 1
    torch.save({"h": H, "rows": rows, "buckets": [b for _, _, b in BUCKETS], "counts": counts, "n_scanned_rows": n_rows, "ctx_len_pcts": [float(x) for x in np.percentile(ctx_len, [5, 25, 50, 75, 95])]}, a.out)
    json.dump({"counts": counts, "n_scanned_rows": n_rows, "shards": shards}, open(a.out.replace(".pt", "_counts.json"), "w"), indent=1)
    print(f"[build] wrote {a.out}: {len(rows)} rows; counts {json.dumps(dict(sorted(counts.items())))} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
