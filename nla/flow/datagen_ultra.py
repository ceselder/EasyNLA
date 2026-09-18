"""More Opus-style warm-start data: UltraFineWeb prefixes -> Sonnet-5 explanations (the original NLA instruction prompt) -> Qwen3.6-27B
layer-42 activations, written in the raw extraction-shard schema that nla.flow.train_cond --train-shards-glob reads.

  rows     : stream UltraFineWeb (split en), cut each doc at k random token positions in [min_tokens, max_tokens], detokenise the prefix
  explain  : Sonnet-5 (concurrent, cached instruction as system prompt), strict <analysis>...</analysis> extraction, resumable chunks, cost log
  extract  : (GPU) layer-42 output at the last token of every prefix -> shard_XXXX.parquet (doc_id/text/explanation/is_val/n_raw_tokens/
             activation_layer/activation_vector), is_val by the standard doc-level rule (permille 20)
"""
from __future__ import annotations
import argparse, asyncio, glob, json, math, os, random, re, sys, time
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

BASE = "Qwen/Qwen3.6-27B"


def cmd_rows(a):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base, token=os.environ.get("HF_TOKEN"))
    ds = load_dataset("openbmb/Ultra-FineWeb", split=a.split, streaming=True, token=os.environ.get("HF_TOKEN"))
    rng = random.Random(a.seed); rows = []; n_docs = 0; t0 = time.time()
    it = iter(ds)
    for _ in range(a.start): next(it)
    for ex in it:
        text = ex.get("content") or ex.get("text") or ""
        if len(text) < 200: continue
        ids = tok(text[: a.max_chars], add_special_tokens=False)["input_ids"]
        if len(ids) < a.min_tokens: continue
        hi = min(len(ids), a.max_tokens); k = min(a.positions_per_doc, max(1, (hi - a.min_tokens) // 40))
        doc_id = f"ultrafineweb:{a.split}:{a.start + n_docs}"
        for pos in sorted(rng.sample(range(a.min_tokens, hi + 1), k)):
            rows.append({"doc_id": doc_id, "text": tok.decode(ids[:pos]), "n_raw_tokens": pos})
        n_docs += 1
        if n_docs % 5000 == 0: print(f"[rows] {n_docs} docs -> {len(rows)} rows ({time.time()-t0:.0f}s)", flush=True)
        if n_docs >= a.n_docs: break
    pq.write_table(pa.Table.from_pylist(rows), a.out, compression="zstd"); print(f"[rows] wrote {a.out}: {len(rows)} rows from {n_docs} docs", flush=True)


def cmd_explain(a):
    import anthropic
    from nla.datagen.stage2_api_explain import _DEFAULT_INSTRUCTION, _DEFAULT_RESPONSE_PATTERN, _extract_and_clean
    instr_prefix, text_part = _DEFAULT_INSTRUCTION.split("Text to analyze:")
    system = [{"type": "text", "text": instr_prefix.strip(), "cache_control": {"type": "ephemeral"}}]
    hdr = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else {}
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=hdr, max_retries=8)
    t = pq.read_table(a.rows); n = t.num_rows
    if a.limit: t = t.slice(0, a.limit); n = t.num_rows
    os.makedirs(a.out_dir, exist_ok=True); done = {int(re.search(r"chunk_(\d+)", f).group(1)) for f in glob.glob(f"{a.out_dir}/chunk_*.parquet")}
    usage = {"in": 0, "cache_read": 0, "cache_write": 0, "out": 0, "ok": 0, "fail": 0, "dropped": 0}
    sem = asyncio.Semaphore(a.concurrency)

    async def one(text):
        prompt = "Text to analyze:" + text_part.replace("{text}", text)
        for attempt in range(6):
            async with sem:
                try:
                    r = await client.messages.create(model=a.model, max_tokens=a.max_tokens, system=system, messages=[{"role": "user", "content": prompt}])
                    u = r.usage; usage["in"] += u.input_tokens; usage["out"] += u.output_tokens
                    usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0; usage["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0
                    raw = "".join(b.text for b in r.content if getattr(b, "type", None) == "text"); ex = _extract_and_clean(raw, _DEFAULT_RESPONSE_PATTERN)
                    if ex is None:
                        usage["dropped"] += 1; usage["drop_truncated" if r.stop_reason == "max_tokens" else "drop_notags"] = usage.get("drop_truncated" if r.stop_reason == "max_tokens" else "drop_notags", 0) + 1
                        if a.debug_drops and usage["dropped"] <= 20: open(f"{a.out_dir}/dropped_raw.txt", "a").write(f"--- stop={r.stop_reason}\n{raw}\n")
                        return None
                    if ex.count("\n\n") < 1:
                        usage["dropped"] += 1; usage["drop_onefeature"] = usage.get("drop_onefeature", 0) + 1
                        if a.debug_drops and usage["dropped"] <= 20: open(f"{a.out_dir}/dropped_raw.txt", "a").write(f"--- onefeature stop={r.stop_reason}\n{raw}\n")
                        return None
                    usage["ok"] += 1; return ex
                except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as e:
                    await asyncio.sleep(min(60, 2 ** attempt + random.random()))
                except anthropic.APIStatusError as e:
                    if attempt >= 2: usage["fail"] += 1; return None
                    await asyncio.sleep(2 ** attempt)
        usage["fail"] += 1; return None

    async def run_chunk(ci, tbl):
        texts = tbl.column("text").to_pylist(); ex = await asyncio.gather(*[one(x) for x in texts])
        keep = [i for i, e in enumerate(ex) if e]
        out = tbl.take(keep).append_column("explanation", pa.array([ex[i] for i in keep]))
        tmp = f"{a.out_dir}/chunk_{ci:04d}.parquet.tmp"; pq.write_table(out, tmp, compression="zstd"); os.replace(tmp, f"{a.out_dir}/chunk_{ci:04d}.parquet")
        return len(keep), len(texts)

    async def main_async():
        t0 = time.time(); total_ok = 0
        for ci, cs in enumerate(range(0, n, a.chunk)):
            if ci in done: continue
            k, m = await run_chunk(ci, t.slice(cs, min(a.chunk, n - cs))); total_ok += k
            el = time.time() - t0; rate = (usage["ok"] + usage["fail"] + usage["dropped"]) / max(el, 1)
            cost = (usage["in"] - usage["cache_read"]) * a.price_in / 1e6 + usage["cache_read"] * a.price_in * 0.1 / 1e6 + usage["cache_write"] * a.price_in * 0.25 / 1e6 + usage["out"] * a.price_out / 1e6
            print(f"[explain] chunk {ci}: {k}/{m} kept | total ok {usage['ok']} dropped {usage['dropped']} (trunc {usage.get('drop_truncated', 0)}, notags {usage.get('drop_notags', 0)}, 1feat {usage.get('drop_onefeature', 0)}) fail {usage['fail']} | {rate:.1f} req/s | "
                  f"tokens in {usage['in']/1e6:.1f}M (cache read {usage['cache_read']/1e6:.1f}M) out {usage['out']/1e6:.2f}M | est cost ${cost:.0f} | {el/60:.0f} min", flush=True)
            json.dump(usage | {"elapsed_s": el, "est_cost_usd": cost, "model": a.model}, open(f"{a.out_dir}/usage.json", "w"), indent=1)
    asyncio.run(main_async())


def cmd_extract(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from nla.val_split import is_val_doc
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.base); tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval(); model.requires_grad_(False)
    inner = model.model; layers = inner.layers if hasattr(inner, "layers") else inner.language_model.layers
    del layers[a.layer + 1:]
    cap = {}
    class _Stop(Exception): pass
    def hook(_m, _i, out): cap["h"] = out[0] if isinstance(out, tuple) else out; raise _Stop()
    layers[a.layer].register_forward_hook(hook)
    files = sorted(glob.glob(a.explained_glob)); files = files[a.shard::a.nshards]
    os.makedirs(a.out_dir, exist_ok=True); d = None; t0 = time.time(); n_done = 0
    for f in files:
        tbl = pq.read_table(f); texts = tbl.column("text").to_pylist(); order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        vecs = [None] * len(texts)
        for cs in range(0, len(order), a.batch):
            idx = order[cs: cs + a.batch]; enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=a.max_len, add_special_tokens=False)
            ids, am = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
            with torch.no_grad():
                try: model(input_ids=ids, attention_mask=am, use_cache=False)
                except _Stop: pass
            h = cap.pop("h"); last = am.sum(1) - 1; v = h[torch.arange(len(idx), device=dev), last].float().cpu().numpy()
            for j, i in enumerate(idx): vecs[i] = v[j]
        d = vecs[0].shape[0]; doc_ids = tbl.column("doc_id").to_pylist()
        out = pa.table({"doc_id": pa.array(doc_ids), "text": tbl.column("text"), "explanation": tbl.column("explanation"),
                        "is_val": pa.array([is_val_doc(x, 20) for x in doc_ids]), "n_raw_tokens": tbl.column("n_raw_tokens"),
                        "activation_layer": pa.array([a.layer] * len(texts)), "activation_vector": pa.FixedSizeListArray.from_arrays(pa.array(np.stack(vecs).reshape(-1)), d)})
        name = os.path.basename(f).replace("chunk_", f"shard_{a.shard:02d}_").replace(".parquet", "") + ".parquet"
        pq.write_table(out, f"{a.out_dir}/{name}", compression="zstd"); n_done += len(texts)
        print(f"[extract] {name}: {len(texts)} rows | {n_done} total | {n_done/(time.time()-t0):.1f} rows/s", flush=True)
    print(f"[extract] done shard {a.shard}/{a.nshards}: {n_done} rows", flush=True)


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rows"); r.add_argument("--out", required=True); r.add_argument("--n-docs", type=int, default=60000); r.add_argument("--start", type=int, default=0); r.add_argument("--split", default="en")
    r.add_argument("--positions-per-doc", type=int, default=4); r.add_argument("--min-tokens", type=int, default=30); r.add_argument("--max-tokens", type=int, default=1024); r.add_argument("--max-chars", type=int, default=12000); r.add_argument("--seed", type=int, default=0); r.add_argument("--base", default=BASE)
    e = sub.add_parser("explain"); e.add_argument("--rows", required=True); e.add_argument("--out-dir", required=True); e.add_argument("--model", default="claude-sonnet-5"); e.add_argument("--max-tokens", type=int, default=400); e.add_argument("--debug-drops", action="store_true")
    e.add_argument("--concurrency", type=int, default=64); e.add_argument("--chunk", type=int, default=2000); e.add_argument("--limit", type=int, default=0); e.add_argument("--price-in", type=float, default=3.0); e.add_argument("--price-out", type=float, default=15.0)
    x = sub.add_parser("extract"); x.add_argument("--explained-glob", required=True); x.add_argument("--out-dir", required=True); x.add_argument("--base", default=BASE); x.add_argument("--layer", type=int, default=42)
    x.add_argument("--batch", type=int, default=32); x.add_argument("--max-len", type=int, default=1100); x.add_argument("--shard", type=int, default=0); x.add_argument("--nshards", type=int, default=1)
    a = p.parse_args(); {"rows": cmd_rows, "explain": cmd_explain, "extract": cmd_extract}[a.cmd](a)


if __name__ == "__main__":
    main()
