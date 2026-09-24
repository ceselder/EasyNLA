"""Verbalizer warm-start data for the compositional NLA: bullet lists of claims the critic can read from the activation (critic-filtered SFT).

  split  (Gemma container) the 728k gold Opus explanations (/vol_q36/data/acts_qwen36_L42/shard_*.parquet) split into atomic bullets
         -> /vol_glp/claims/warmstart/split/split_<shard>.parquet (row, doc_id, is_val, bullets)
  multi  (Gemma container) held-out synthetic anchors (is_val; no critic phase trains on them): 3 random aspects x 2 claims each, quote-checked
         -> /vol_glp/claims/warmstart/multi/multi_<name>.parquet (anchor_id, claims, types)
  score  (nla-glp GPU) single-claim PMI of every candidate bullet under a critic (FM proxy, D noise draws x 5 t, the same noise for all of an
         activation's bullets) -> /vol_glp/claims/warmstart/scores_<critic>/<part>.parquet
  build  (CPU) keep bullets with PMI > lambda, best-first, cap 6 -> SFT parquets in the verbalizer's schema (prompt, activation_vector, doc_id,
         response = "<explanation>\\n• c1\\n• c2 ...\\n</explanation>") + held-out split -> /vol_glp/claims/warmstart/sft_<critic>/{train,val}.parquet
"""
import argparse, glob, gzip, json, os, random, re, sys, time, zlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
WS = "/vol_glp/claims/warmstart"
GOLD = "/vol_q36/data/acts_qwen36_L42/shard_*.parquet"

SPLIT_SYS = """You split an explanation of a language model's internal state into ATOMIC CLAIMS.
The explanation describes what a language model is representing at one token of a document (topic, genre, entities, numbers, the current sentence, what comes next, ...).
Rewrite it as a list of short, self-contained claims:
- one fact per claim; split compound sentences; drop filler and hedging words but keep every specific (names, numbers, quoted words);
- each claim must stand alone (name the thing, never "it" or "this" without a referent);
- use only content that is in the explanation; do not add, infer or correct anything;
- keep section labels only when they carry meaning ("Genre: product review");
- at most 12 claims.
Output one claim per line, each starting with "• ", and nothing else."""


def split_items(files, rng):
    import pyarrow.parquet as pq
    out = []
    for f in files:
        name = os.path.basename(f)[:-8]; t = pq.read_table(f, columns=["explanation", "doc_id", "is_val"]).to_pydict()
        for r, (z, d, v) in enumerate(zip(t["explanation"], t["doc_id"], t["is_val"])):
            if z and z.strip(): out.append(({"shard": name, "row": r, "doc_id": d, "is_val": bool(v)}, [{"role": "system", "content": SPLIT_SYS}, {"role": "user", "content": z.strip()}]))
    return out


def run_split(root_out, files, commit=None):
    import pyarrow as pa, pyarrow.parquet as pq
    import claims_gemma as cg
    os.makedirs(f"{root_out}/split", exist_ok=True)
    todo = [f for f in files if not os.path.exists(f"{root_out}/split/split_{os.path.basename(f)[:-8]}.parquet")]
    if not todo: return 0
    proc, t_start = cg.start_server([], 512, 8000, open("/tmp/server_split.log", "w"))
    if t_start is None: raise SystemExit("gemma server did not start: " + open("/tmp/server_split.log").read()[-2000:])
    try:
        for f in todo:
            items = split_items([f], random.Random(0)); t0 = time.time()
            texts, tok = cg.run_items([(None, None, m) for _, m in items], 8000, 1024, max_tokens=400, temperature=0.3)
            recs = []
            for (meta, _), txt in zip(items, texts):
                b = [re.sub(r"^\s*[•\-*]\s*", "", l).strip() for l in (txt or "").splitlines() if re.match(r"^\s*[•\-*]\s+\S", l)]
                b = [x for x in b if len(x.split()) >= 3][:12]
                if b: recs.append(dict(meta, bullets=b))
            name = os.path.basename(f)[:-8]; tmp = f"{root_out}/split/split_{name}.parquet.tmp"
            pq.write_table(pa.Table.from_pylist(recs), tmp, compression="zstd"); os.replace(tmp, f"{root_out}/split/split_{name}.parquet")
            print(f"[ws-split] {name}: {len(items)} explanations -> {len(recs)} with bullets ({np.mean([len(r['bullets']) for r in recs]):.1f} each) in {time.time() - t0:.0f}s ({len(items) / (time.time() - t0):.0f}/s)", flush=True)
            if commit: commit()
    finally:
        cg.stop_server(proc)
    return len(todo)


def run_multi(root, root_out, names, commit=None):
    """3 random aspects x 2 claims for every held-out anchor of the given text shards (quote-checked, false twins kept for reference)"""
    import pyarrow as pa, pyarrow.parquet as pq
    import claims_gemma as cg
    from claims_semantic import SYSTEM_WIDE, sample_request_wide, user_msg
    os.makedirs(f"{root_out}/multi", exist_ok=True)
    todo = [n for n in names if not os.path.exists(f"{root_out}/multi/multi_{n}.parquet")]
    if not todo: return 0
    proc, t_start = cg.start_server([], 512, 8000, open("/tmp/server_multi.log", "w"))
    if t_start is None: raise SystemExit("gemma server did not start")
    try:
        for n in todo:
            rng = random.Random(zlib.crc32(n.encode())); items = []
            for l in gzip.open(f"{root}/text/text_{n}.jsonl.gz", "rt"):
                r = json.loads(l)
                if not r.get("is_val"): continue
                asp = []
                while len(asp) < 3:
                    q = sample_request_wide(r, rng)
                    if q["aspects"][0] not in asp: asp.append(q["aspects"][0])
                q = {"aspects": asp, "gran": q["gran"], "style": q["style"], "n": 6, "short": True}; r["_shard"] = n
                um = user_msg(r, q).replace("Aspect: ", "Aspects (write 2 claims for EACH of these three aspects): ", 1)
                items.append((r, q, [{"role": "system", "content": SYSTEM_WIDE}, {"role": "user", "content": um}]))
            t0 = time.time(); texts, tok = cg.run_items(items, 8000, 1024, max_tokens=500)
            recs, st = cg.verify(items, texts)
            tbl = pa.Table.from_pylist([{k: v for k, v in r.items() if k != "_shard"} for r in recs])
            tmp = f"{root_out}/multi/multi_{n}.parquet.tmp"; pq.write_table(tbl, tmp, compression="zstd"); os.replace(tmp, f"{root_out}/multi/multi_{n}.parquet")
            print(f"[ws-multi] {n}: {len(items)} held-out anchors -> {st['quote_ok']} claims (quote pass {st['quote_pass_rate']:.3f}) in {time.time() - t0:.0f}s", flush=True)
            if commit: commit()
    finally:
        cg.stop_server(proc)
    return len(todo)
