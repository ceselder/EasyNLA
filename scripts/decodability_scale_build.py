"""Decodability data-scaling study, data build (CPU, streaming) — (activation, detail, token distance) rows from ~10M Qwen3.6-27B L42 positions.

Sources: the scale fork's labelled shards (/vol_glp/scale/{g1,g2}/shards/shard_*.parquet: `text` = context up to and including the read-out
position, `activation_vector` = h there) and the Opus extraction shards (/vol_q36/data/acts_qwen36_L42). Same detail definitions and distance
buckets as scripts/decodability_probe_build.py (numbers >= 2 digits, 2-4-word capitalised names, 12-120-char quoted spans; k = tokens after the
detail's last token; buckets 0 | 1 | 2-4 | 5-16 | 17-64 | 65-256), but every (type, bucket) cell keeps up to --per-cell rows (reservoir), so
the probes can be trained at 10k … 3M rows. Pass 1 tokenises every context (offsets) and fills the reservoirs with row metadata only; pass 2
re-reads each shard once, writes the selected activations as fp16 chunk files (row order = metadata order), draws the matched negative
(the true value of another selected row of the same type and split — other document, not a substring of this context; identical marginals),
the near / far number edits, and finally tokenises the value table (token ids, hashed character 2-3-grams, digit-position one-hots).
Documents are hash-split: 5 % test, 5 % val, 90 % train.
Output (--out-dir): h_<type>_<chunk>.npy (fp16 [n, 5120]), meta_<type>.parquet, values_<type>_{tokens,hash,digits,strings}.npy|json, counts.json
usage: python scripts/decodability_scale_build.py --out-dir /vol_glp/decodability/scale [--max-shards N] [--per-cell 500000]
"""
from __future__ import annotations
import argparse, bisect, glob, hashlib, json, os, random, re, sys, time
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from nla.flow.negatives import NAME, QUOTE
from nla.flow.halluc_classify import NUM, perturb

TYPES = ("number", "name", "quote")
BUCKETS = [(0, 0, "0"), (1, 1, "1"), (2, 4, "2-4"), (5, 16, "5-16"), (17, 64, "17-64"), (65, 256, "65-256")]
BNAME = [b for _, _, b in BUCKETS]
NHASH, NTOK, NHG = 4096, 32, 64


def bucket_of(k):
    for i, (lo, hi, _) in enumerate(BUCKETS):
        if lo <= k <= hi: return i
    return -1


def candidates(text):
    out = {}
    for m in NUM.finditer(text):
        if len(m.group(1).replace(",", "")) >= 2 or m.group(2): out[(0, m.group(0))] = m.end()
    for m in NAME.finditer(text): out[(1, m.group(1))] = m.end(1)
    for m in QUOTE.finditer(text): out[(2, m.group(1))] = m.end(1)
    return [(t, v, e) for (t, v), e in out.items()]


def doc_hash(doc): return int(hashlib.sha256(doc.encode()).hexdigest()[:15], 16)
def split_of(dh): r = dh % 20; return 0 if r == 0 else (1 if r == 1 else 2)      # 0 test, 1 val, 2 train


def digit_extras(v):
    x = np.zeros(131, dtype=np.float16); digits = re.sub(r"[^\d]", "", v.split(".")[0] if "." in v else v)
    if digits:
        for i, ch in enumerate(digits[:6]): x[i * 10 + int(ch)] = 1
        for i, ch in enumerate(digits[::-1][:6]): x[60 + i * 10 + int(ch)] = 1
        x[120 + min(len(digits), 8) - 1] = 1; x[128] = min(len(digits) - 1, 30) / 10   # ~log10 magnitude (digit strings can exceed float range)
    x[129] = float("," in v); x[130] = float("." in v)
    return x


def hash_ids(v):
    s = f"^{v.lower()}$"; ids = [int(hashlib.md5(s[i: i + n].encode()).hexdigest()[:8], 16) % NHASH for n in (2, 3) for i in range(len(s) - n + 1)]
    ids = ids[:NHG]; return ids + [-1] * (NHG - len(ids))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True); p.add_argument("--base", default="Qwen/Qwen3.6-27B")
    p.add_argument("--globs", default="/vol_glp/scale/g1/shards/shard_*.parquet,/vol_glp/scale/g2/shards/shard_*.parquet,/vol_q36/data/acts_qwen36_L42/shard_*.parquet")
    p.add_argument("--max-shards", type=int, default=0); p.add_argument("--per-cell", type=int, default=500000); p.add_argument("--max-per-row", type=int, default=2)
    p.add_argument("--chunk-rows", type=int, default=300000); p.add_argument("--resume-pass1", action="store_true", help="load pass1_state.pkl from --out-dir instead of re-scanning the shards"); p.add_argument("--seed", type=int, default=0); p.add_argument("--tok-batch", type=int, default=2048)
    a = p.parse_args(); rng = random.Random(a.seed); t0 = time.time(); os.makedirs(a.out_dir, exist_ok=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base, token=os.environ.get("HF_TOKEN"))
    shards = [s for g in a.globs.split(",") for s in sorted(glob.glob(g))]
    if a.max_shards: shards = shards[: a.max_shards]
    print(f"[scale-build] {len(shards)} shards", flush=True)

    # ---------------- pass 1: metadata reservoirs
    vals = [{} for _ in TYPES]; vstr = [[] for _ in TYPES]                      # per type: value string -> id
    def vid(t, v):
        d = vals[t]; i = d.get(v)
        if i is None: i = len(vstr[t]); d[v] = i; vstr[t].append(v)
        return i
    res = {(t, b): [] for t in range(3) for b in range(6)}; seen = {k: 0 for k in res}   # (shard_idx, row, k, value_id, doc_hash, n_ctx)
    seen_pos = set(); n_rows = 0; n_dup = 0; ctx_len = []
    import pickle; state_path = os.path.join(a.out_dir, "pass1_state.pkl")
    if a.resume_pass1:
        st = pickle.load(open(state_path, "rb")); res, seen, vstr, n_rows, n_dup, ctx_len, shards = st["res"], st["seen"], st["vstr"], st["n_rows"], st["n_dup"], st["ctx_len"], st["shards"]
        vals = [{v: i for i, v in enumerate(vs)} for vs in vstr]; print(f"[scale-build] resumed pass-1 state: {n_rows} positions", flush=True)
    for si, sp in enumerate(shards if not a.resume_pass1 else []):
        t1 = time.time(); t = pq.read_table(sp, columns=["text", "doc_id"]); texts = t.column("text").to_pylist(); docs = t.column("doc_id").to_pylist()
        for s in range(0, len(texts), a.tok_batch):
            enc = tok(texts[s: s + a.tok_batch], add_special_tokens=False, return_offsets_mapping=True)
            for j, offs in enumerate(enc["offset_mapping"]):
                i = s + j; n = len(offs)
                if n < 2: continue
                key = (docs[i], n)
                if key in seen_pos: n_dup += 1; continue
                seen_pos.add(key); n_rows += 1
                if n_rows % 50 == 0: ctx_len.append(n)
                dh = doc_hash(docs[i]); starts = [o[0] for o in offs]; cands = candidates(texts[i]); rng.shuffle(cands); per_type = [0, 0, 0]
                for typ, val, end in cands:
                    if per_type[typ] >= a.max_per_row: continue
                    last = bisect.bisect_left(starts, end) - 1
                    if last < 0: continue
                    k = (n - 1) - last; b = bucket_of(k)
                    if b < 0: continue
                    per_type[typ] += 1; key2 = (typ, b); seen[key2] += 1; rec = (si, i, k, vid(typ, val), dh, n)
                    r_ = res[key2]
                    if len(r_) < a.per_cell: r_.append(rec)
                    else:
                        q = rng.randrange(seen[key2])
                        if q < a.per_cell: r_[q] = rec
        fill = " ".join(f"{TYPES[t][0]}{BNAME[b]}:{len(res[(t, b)]) // 1000}k" for t in range(3) for b in range(6))
        print(f"[scale-build] pass1 {si + 1}/{len(shards)} {os.path.basename(sp)}: {len(texts)} rows ({time.time() - t1:.0f}s) | total {n_rows} (dup {n_dup}) | {fill} | {(time.time() - t0) / 60:.0f} min", flush=True)
    counts_seen = {f"{TYPES[t]}/{BNAME[b]}": seen[(t, b)] for t in range(3) for b in range(6)}
    if not a.resume_pass1: pickle.dump({"res": res, "seen": seen, "vstr": vstr, "n_rows": n_rows, "n_dup": n_dup, "ctx_len": ctx_len, "shards": shards}, open(state_path, "wb")); print(f"[scale-build] pass-1 state saved to {state_path}", flush=True)
    print(f"[scale-build] pass1 done: {n_rows} positions; candidates seen {json.dumps(counts_seen)}; context tokens pcts {np.percentile(ctx_len, [5, 25, 50, 75, 95]).tolist()}", flush=True)

    # ---------------- pass 2: activations + negatives, per shard
    # rows per type in a fixed order (by shard, then row) so each shard is read once; negatives drawn from the whole selected set of the same type+split
    rows = {t: [] for t in range(3)}
    for (t, b), lst in res.items():
        for (si, i, k, v, dh, n) in lst: rows[t].append({"si": si, "row": i, "k": k, "b": b, "v": v, "dh": dh, "n": n, "split": split_of(dh)})
    for t in range(3): rows[t].sort(key=lambda r: (r["si"], r["row"]))
    by_split = {t: {s: [r["v"] for r in rows[t] if r["split"] == s] for s in range(3)} for t in range(3)}
    by_split_doc = {t: {s: [r["dh"] for r in rows[t] if r["split"] == s] for s in range(3)} for t in range(3)}
    per_shard = {}
    for t in range(3):
        for idx, r in enumerate(rows[t]): per_shard.setdefault(r["si"], []).append((t, idx))
    buf = {t: np.zeros((a.chunk_rows, 5120), dtype=np.float16) for t in range(3)}; nbuf = {t: 0 for t in range(3)}; chunk_id = {t: 0 for t in range(3)}; chunk_sizes = {t: [] for t in range(3)}
    def flush(t):
        if nbuf[t] == 0: return
        np.save(os.path.join(a.out_dir, f"h_{TYPES[t]}_{chunk_id[t]:04d}.npy"), buf[t][: nbuf[t]]); chunk_sizes[t].append(nbuf[t]); chunk_id[t] += 1; nbuf[t] = 0
    n_written = {t: 0 for t in range(3)}; n_noneg = 0
    for si, sp in enumerate(shards):
        if si not in per_shard: continue
        t1 = time.time(); t = pq.read_table(sp, columns=["activation_vector", "text", "doc_id"]); texts = t.column("text").to_pylist()
        acts = np.asarray(t.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(t.num_rows, -1)
        items = sorted(per_shard[si], key=lambda x: rows[x[0]][x[1]]["row"])
        for (typ, idx) in items:
            r = rows[typ][idx]; text = texts[r["row"]]; tl = text.lower().replace(",", ""); pool_v = by_split[typ][r["split"]]; pool_d = by_split_doc[typ][r["split"]]
            r["neg"] = -1
            for _ in range(60):
                j = rng.randrange(len(pool_v)); v = vstr[typ][pool_v[j]]
                if pool_d[j] != r["dh"] and pool_v[j] != r["v"] and v.lower().replace(",", "") not in tl: r["neg"] = pool_v[j]; break
            if r["neg"] < 0: n_noneg += 1
            if typ == 0:
                try: r["near"] = vid(0, perturb(vstr[0][r["v"]], rng, "near")); r["far"] = vid(0, perturb(vstr[0][r["v"]], rng, "far"))
                except Exception: r["near"] = -1; r["far"] = -1
                r["near_in_ctx"] = int(r["near"] >= 0 and vstr[0][r["near"]].lower().replace(",", "") in tl)
            buf[typ][nbuf[typ]] = acts[r["row"]].astype(np.float16); nbuf[typ] += 1; n_written[typ] += 1
            if nbuf[typ] == a.chunk_rows: flush(typ)
        print(f"[scale-build] pass2 {si + 1}/{len(shards)}: {len(items)} rows ({time.time() - t1:.0f}s) | written {n_written} | {(time.time() - t0) / 60:.0f} min", flush=True)
    for t in range(3): flush(t)
    # ---------------- metadata + value tables
    counts = {}
    for t in range(3):   # metadata first (so a value-table failure cannot lose the row tables)
        R = rows[t]
        cols = {"bucket": pa.array([BNAME[r["b"]] for r in R]), "k": pa.array([r["k"] for r in R], pa.int32()), "split": pa.array([r["split"] for r in R], pa.int8()),
                "value_id": pa.array([r["v"] for r in R], pa.int32()), "neg_id": pa.array([r["neg"] for r in R], pa.int32()), "doc_hash": pa.array([r["dh"] for r in R], pa.int64()),
                "n_ctx": pa.array([r["n"] for r in R], pa.int32()), "shard": pa.array([r["si"] for r in R], pa.int16()), "row": pa.array([r["row"] for r in R], pa.int32())}
        if t == 0: cols.update({"near_id": pa.array([r.get("near", -1) for r in R], pa.int32()), "far_id": pa.array([r.get("far", -1) for r in R], pa.int32()), "near_in_ctx": pa.array([r.get("near_in_ctx", 0) for r in R], pa.int8())})
        pq.write_table(pa.table(cols), os.path.join(a.out_dir, f"meta_{TYPES[t]}.parquet")); json.dump(vstr[t], open(os.path.join(a.out_dir, f"values_{TYPES[t]}_strings.json"), "w"))
        print(f"[scale-build] meta_{TYPES[t]}.parquet written ({len(R)} rows)", flush=True)
    for t in range(3):
        R = rows[t]; keep = [r for r in R if r["neg"] >= 0]
        for r in R: counts[f"{TYPES[t]}/{BNAME[r['b']]}/{['test', 'val', 'train'][r['split']]}"] = counts.get(f"{TYPES[t]}/{BNAME[r['b']]}/{['test', 'val', 'train'][r['split']]}", 0) + 1
        V = vstr[t]; print(f"[scale-build] {TYPES[t]}: {len(R)} rows ({len(keep)} with a negative), {len(V)} unique values; tokenising values …", flush=True)
        try:
            toks = np.full((len(V), NTOK), -1, dtype=np.int32)
            for s in range(0, len(V), 8192):
                enc = tok([" " + v for v in V[s: s + 8192]], add_special_tokens=False)["input_ids"]
                for j, ids in enumerate(enc): ids = (ids or [0])[:NTOK]; toks[s + j, : len(ids)] = ids
            np.save(os.path.join(a.out_dir, f"values_{TYPES[t]}_tokens.npy"), toks)
            np.save(os.path.join(a.out_dir, f"values_{TYPES[t]}_hash.npy"), np.array([hash_ids(v) for v in V], dtype=np.int16))
            np.save(os.path.join(a.out_dir, f"values_{TYPES[t]}_digits.npy"), np.stack([digit_extras(v) for v in V]) if t == 0 else np.zeros((len(V), 131), dtype=np.float16))
        except Exception as e: print(f"[scale-build] value tables for {TYPES[t]} FAILED: {e!r} (strings + meta are saved; rerun the tables offline)", flush=True)
    json.dump({"counts": counts, "candidates_seen": counts_seen, "n_positions": n_rows, "n_dup_positions": n_dup, "n_no_negative": n_noneg, "chunk_sizes": {TYPES[t]: chunk_sizes[t] for t in range(3)},
               "shards": shards, "per_cell": a.per_cell, "ctx_len_pcts": np.percentile(ctx_len, [5, 25, 50, 75, 95]).tolist(), "elapsed_min": (time.time() - t0) / 60},
              open(os.path.join(a.out_dir, "counts.json"), "w"), indent=1)
    print(f"[scale-build] DONE {json.dumps(counts)} ({(time.time() - t0) / 60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
