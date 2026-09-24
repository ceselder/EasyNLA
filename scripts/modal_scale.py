"""Scaling the conditional (activation, explanation) data with a cheap open labeller: Gemma 4 26B-A4B (MoE, 4B active) on B200s.

Pipeline (all idempotent per shard; outputs under /vol_glp/scale on the nla-glp volume):
  docs      (CPU)   fresh FineFineWeb documents, uniform over its 66 domains (scripts/claims_extract.py docs --source ffw)
  positions (CPU)   Qwen3.6-27B tokenizer, 10 positions per doc exactly as nla/datagen/stage0_extract.py (per-doc sha256 RNG, token index >= 50,
                    non-special; token index < 4096), prefix decoded -> detokenized text; documents whose opening
                    matches any Opus-pool prefix are dropped (leak guard); variant assigned per row (nla/datagen/scale_templates.MIX)
  label     (B200)  Gemma 4 26B-A4B-it in vLLM, thinking off, temperature 1.0 / top-p 0.95, the gold-data instruction (V0 verbatim) + variants
  extract   (B200)  Qwen3.6-27B HF forward ONCE per document up to its last sampled position, output of decoder block 42 at every position
  join      (CPU)   -> /vol_glp/scale/shards/shard_XXXX.parquet in the raw-extraction schema nla/flow/train_cond.py reads
                    (doc_id, text, explanation, is_val, n_raw_tokens, activation_layer, activation_vector) + variant

  modal run scripts/modal_scale.py --task pilot --n 3000 --nv 400
  modal run --detach scripts/modal_scale.py --task run --n-docs 1000000 --docs-per-shard 5000
"""
import os, sys, json, time
import modal
for _p in (os.path.dirname(os.path.abspath(__file__)), os.path.join(os.environ.get("PYTHONPATH", "/root/easyNLA").split(":")[0], "scripts")):
    if os.path.isdir(_p) and _p not in sys.path: sys.path.insert(0, _p)
from modal_nla_exp import image_base, SECRETS, REPO_LOCAL, REPO_REMOTE, REPO_IGNORE  # noqa: E402

LABELLER = "google/gemma-4-26B-A4B-it"
BASE = "Qwen/Qwen3.6-27B"
ROOT = "/vol_glp/scale"
LAYER = 42
MAX_POS = 4096            # positions are token indices < 4096 (the Opus pool's n_raw_tokens reach 3929; docs are capped at 12k chars anyway)
MIN_POS = 50              # stage-0 _MIN_POSITION
N_POS = 10                # stage-0 positions_per_doc
SEED = 42                 # stage-0 default seed

vol_glp = modal.Volume.from_name("nla-glp", create_if_missing=True)
vol_q36 = modal.Volume.from_name("nla-qwen36-ema")
VOLS = {"/vol_glp": vol_glp, "/vol_q36": vol_q36}
image_q = image_base.add_local_dir(REPO_LOCAL, REPO_REMOTE, copy=False, ignore=REPO_IGNORE)
image_g = (modal.Image.from_registry("vllm/vllm-openai:v0.29.0", setup_dockerfile_commands=["RUN ln -sf $(which python3) /usr/local/bin/python"]).entrypoint([])
           .run_commands("pip install --no-cache-dir pyarrow wandb 'huggingface_hub[hf_xet]'")
           .env({"HF_HOME": "/vol_glp/hf", "HF_XET_HIGH_PERFORMANCE": "1", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
                 "PYTHONPATH": REPO_REMOTE, "VLLM_ALLOW_INSECURE_SERIALIZATION": "1"})
           .add_local_dir(REPO_LOCAL, REPO_REMOTE, copy=False, ignore=REPO_IGNORE))
app = modal.App("nla-scale")


# ------------------------------------------------------------------------------------------------------------ labeller (Gemma, vLLM)
# production labeller config, from the 6-way bench (20k real prompts, 1 B200 each): bigger batches +8 %, one variant per document +3 % prompts/s at
# 10 % longer outputs; fp8 weights+KV -4 % and 8x parse failures (decode-bound on the MoE/attention kernels, not weight bandwidth); n>1 gives no
# extra explanations per second (127 at n=2 vs 126) -> bf16, 1024 seqs, 32k batched tokens, n=1
LABEL_CFG = dict(fp8=False, fp8_kv=False, max_num_seqs=1024, max_num_batched_tokens=32768, n=1)


def _llm(cfg=None):
    from vllm import LLM
    c = {**LABEL_CFG, **(cfg or {})}
    kw = dict(model=LABELLER, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.90, limit_mm_per_prompt={"image": 0},
              max_num_seqs=c["max_num_seqs"], enable_prefix_caching=True, seed=0)
    if c.get("max_num_batched_tokens"): kw["max_num_batched_tokens"] = c["max_num_batched_tokens"]
    if c.get("fp8"): kw["quantization"] = "fp8"
    if c.get("fp8_kv"): kw["kv_cache_dtype"] = "fp8"
    return LLM(**kw)


def _label(llm, texts, variants, max_prefix_chars=24000, n=1):
    """n Gemma completions per (text, variant) -> list of (explanations[n] (None = unparseable), raws[n], n_prompt_tokens, n_out_tokens_total).
    Prompts are issued per variant in the given row order (rows of one document are adjacent -> the document prefix is shared in the cache)."""
    from vllm import SamplingParams
    from nla.datagen.scale_templates import VARIANTS, MAX_TOKENS, clean
    by_v = {}
    for i, v in enumerate(variants): by_v.setdefault(v, []).append(i)
    out = [None] * len(texts)
    for v, idx in by_v.items():
        msgs = [[{"role": "user", "content": VARIANTS[v].format(text=texts[i][-max_prefix_chars:])}] for i in idx]
        sp = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=MAX_TOKENS[v], seed=None, n=n)
        res = llm.chat(msgs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
        for i, r in zip(idx, res):
            raws = [o.text for o in r.outputs]
            out[i] = ([clean(x) for x in raws], raws, len(r.prompt_token_ids), sum(len(o.token_ids) for o in r.outputs))
    return out


@app.function(image=image_g, gpu="B200", timeout=4 * 3600, volumes=VOLS, secrets=SECRETS, cpu=16, memory=128 * 1024)
def pilot(n: int = 3000, nv: int = 400):
    """Gemma on (doc, position) rows that already have Opus-5 explanations (held-out av_sft_val): V0 on n rows, each other variant on the first
    nv of them. Throughput measured on the V0 pass. -> ROOT/pilot/gemma_pilot.parquet + throughput.json"""
    import numpy as np, pyarrow as pa, pyarrow.parquet as pq
    from nla.datagen.scale_templates import VARIANTS
    t = pq.read_table("/vol_q36/data/sft/av_sft_val.parquet", columns=["doc_id", "detokenized_text_truncated", "response", "n_raw_tokens"])
    rng = np.random.default_rng(0); pick = sorted(rng.choice(t.num_rows, size=min(n, t.num_rows), replace=False).tolist())
    rows = t.take(pick).to_pydict(); texts = rows["detokenized_text_truncated"]
    t0 = time.time(); llm = _llm(); t_load = time.time() - t0
    t0 = time.time(); r0 = _label(llm, texts, ["v0_opus"] * len(texts)); dt = time.time() - t0
    r0 = [(x[0][0], x[1][0], x[2], x[3]) for x in r0]
    thr = dict(n=len(texts), seconds=dt, load_seconds=t_load, expl_per_s=len(texts) / dt, prompt_tok_per_s=sum(x[2] for x in r0) / dt,
               out_tok_per_s=sum(x[3] for x in r0) / dt, mean_prompt_tokens=float(np.mean([x[2] for x in r0])), mean_out_tokens=float(np.mean([x[3] for x in r0])),
               parse_fail=float(np.mean([x[0] is None for x in r0])))
    print("[pilot] V0 throughput", json.dumps(thr), flush=True)
    recs = [dict(row=pick[i], doc_id=rows["doc_id"][i], n_raw_tokens=int(rows["n_raw_tokens"][i]), text=texts[i], opus=rows["response"][i], variant="v0_opus",
                 explanation=r0[i][0], raw=r0[i][1], n_prompt=r0[i][2], n_out=r0[i][3]) for i in range(len(texts))]
    for v in VARIANTS:
        if v == "v0_opus": continue
        t0 = time.time(); rv = [(x[0][0], x[1][0], x[2], x[3]) for x in _label(llm, texts[:nv], [v] * nv)]; thr[f"{v}_seconds"] = time.time() - t0
        thr[f"{v}_parse_fail"] = float(np.mean([x[0] is None for x in rv])); thr[f"{v}_mean_out_tokens"] = float(np.mean([x[3] for x in rv]))
        recs += [dict(row=pick[i], doc_id=rows["doc_id"][i], n_raw_tokens=int(rows["n_raw_tokens"][i]), text=texts[i], opus=rows["response"][i], variant=v,
                      explanation=rv[i][0], raw=rv[i][1], n_prompt=rv[i][2], n_out=rv[i][3]) for i in range(nv)]
    os.makedirs(f"{ROOT}/pilot", exist_ok=True)
    pq.write_table(pa.Table.from_pylist(recs), f"{ROOT}/pilot/gemma_pilot.parquet", compression="zstd")
    json.dump(thr, open(f"{ROOT}/pilot/throughput.json", "w"), indent=1); vol_glp.commit()
    print("[pilot] done", json.dumps(thr), flush=True)
    return thr


@app.function(image=image_g, gpu="B200", timeout=3 * 3600, volumes=VOLS, secrets=SECRETS, cpu=16, memory=128 * 1024)
def bench(name: str, cfg: dict, per_doc_variants: bool = False, n_rows: int = 20000, pos_file: str = f"{ROOT}/pos/pos_0000.parquet"):
    """labeller throughput for one config on n_rows real prompts (a positions shard: 10 positions per doc, adjacent). per_doc_variants: one
    variant per document (all its positions share instruction + document prefix in the cache) instead of one per row. fp8 configs also
    label the pilot's judged rows (V0) -> ROOT/pilot/bench_<name>.parquet for a quality check against the bf16 pilot."""
    import numpy as np, pyarrow as pa, pyarrow.parquet as pq
    from nla.datagen.scale_templates import assign
    d = pq.read_table(pos_file).slice(0, n_rows).to_pydict()
    var = [assign(doc) for doc in d["doc_id"]] if per_doc_variants else d["variant"]
    t0 = time.time()
    try: llm = _llm(cfg)
    except Exception as e:
        return {"name": name, "error": f"engine: {str(e)[:300]}"}
    t_load = time.time() - t0; n = cfg.get("n", 1)
    _label(llm, d["text"][:500], var[:500], n=n)                                   # warm-up (compile / graph capture outside the timing)
    t0 = time.time(); r = _label(llm, d["text"], var, n=n); dt = time.time() - t0
    st = {"name": name, "cfg": cfg, "per_doc_variants": per_doc_variants, "rows": len(r), "seconds": dt, "load_seconds": t_load,
          "prompts_per_s": len(r) / dt, "expl_per_s": sum(sum(e is not None for e in x[0]) for x in r) / dt, "prompt_tok_per_s": sum(x[2] for x in r) / dt,
          "decode_tok_per_s": sum(x[3] for x in r) / dt, "mean_out_tokens_per_sample": float(np.mean([x[3] / n for x in r])),
          "parse_fail": float(np.mean([e is None for x in r for e in x[0]]))}
    if cfg.get("fp8") or n > 1:
        P = pq.read_table(f"{ROOT}/pilot/gemma_pilot.parquet").to_pydict(); idx = [i for i, v in enumerate(P["variant"]) if v == "v0_opus"]
        rp = _label(llm, [P["text"][i] for i in idx], ["v0_opus"] * len(idx), n=1)
        pq.write_table(pa.Table.from_pylist([{"row": P["row"][i], "text": P["text"][i], "opus": P["opus"][i], "variant": "v0_opus", "explanation": x[0][0], "n_out": x[3]}
                                             for i, x in zip(idx, rp)]), f"{ROOT}/pilot/bench_{name}.parquet", compression="zstd"); vol_glp.commit()
    print("[bench]", json.dumps(st), flush=True); return st


# ------------------------------------------------------------------------------------------------------------ pipeline stages
def _exists(p):
    return os.path.exists(p) and os.path.getsize(p) > 0


@app.function(image=image_q, timeout=8 * 3600, volumes=VOLS, secrets=SECRETS, cpu=4, memory=16 * 1024)
def docs(slice_: str, n_docs: int, tag: str = "g1"):
    """fresh FineFineWeb docs (66 domains, this worker's slice of them) -> ROOT/docs/docs_ffw_<tag>_sXX_YYY.parquet, 5000 docs per part.
    claims_extract counts --n-docs over its whole source mix and gives FineFineWeb 50 %, hence 2 * n_docs."""
    import subprocess
    vol_glp.reload()
    cmd = [sys.executable, f"{REPO_REMOTE}/scripts/claims_extract.py", "docs", "--source", "ffw", "--n-docs", str(2 * n_docs), "--root", ROOT,
           "--tag", tag, "--slice", slice_, "--part-size", "5000", "--seed", "7"]
    print("[modal] " + " ".join(cmd), flush=True); rc = subprocess.call(cmd, cwd=REPO_REMOTE); vol_glp.commit(); return rc


@app.function(image=image_q, timeout=12 * 3600, volumes=VOLS, secrets=SECRETS, cpu=2, memory=4 * 1024)
def docs_all(n_docs: int, slices: int, tag: str):
    """all docs workers for one run; markers ROOT/<tag>/docs.started / docs.done let the orchestrator attach instead of re-spawning"""
    R = f"{ROOT}/{tag}"; os.makedirs(R, exist_ok=True); open(f"{R}/docs.started", "w").write(str(time.time())); vol_glp.commit()
    rcs = []
    for c in [docs.spawn(f"{i}/{slices}", n_docs, tag) for i in range(slices)]:
        try: rcs.append(c.get())
        except Exception as e: rcs.append(str(e)[:200])
    vol_glp.reload(); open(f"{R}/docs.done", "w").write(json.dumps(rcs)); vol_glp.commit(); return rcs


_POOL_HEADS = None


def _pool_heads():
    """first 160 characters of every Opus-pool prefix (all splits): a fresh document whose opening matches one of them is dropped"""
    global _POOL_HEADS
    if _POOL_HEADS is None:
        import pyarrow.parquet as pq
        _POOL_HEADS = set(t[:160] for t in pq.read_table("/vol_q36/data/shared_pool.parquet", columns=["text"]).column(0).to_pylist())
    return _POOL_HEADS


@app.function(image=image_q, timeout=2 * 3600, volumes=VOLS, secrets=SECRETS, cpu=8, memory=32 * 1024, max_containers=24)
def positions(part: str, sid: int, root: str = ROOT):
    """one docs part -> ROOT/pos/pos_XXXX.parquet (row level: doc_id, n_raw_tokens, text, variant) + ROOT/pos/posdocs_XXXX.parquet (doc_id, ids up
    to the last sampled position). Positions follow nla/datagen/stage0_extract._sample_positions exactly (seed 42, index >= 50, non-special)."""
    import hashlib, random
    import pyarrow as pa, pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from nla.datagen.scale_templates import assign
    out, outd = f"{root}/pos/pos_{sid:04d}.parquet", f"{root}/pos/posdocs_{sid:04d}.parquet"
    vol_glp.reload()
    if _exists(out) and _exists(outd): return "exists"
    tok = AutoTokenizer.from_pretrained(BASE); special = set(tok.all_special_ids); heads = _pool_heads()
    d = pq.read_table(part, columns=["doc_id", "text"]).to_pydict()
    rows, drows, n_leak, n_short = [], [], 0, 0
    enc = tok(d["text"], add_special_tokens=False, truncation=True, max_length=MAX_POS)["input_ids"]
    for doc_id, text, ids in zip(d["doc_id"], d["text"], enc):
        if text[:160] in heads: n_leak += 1; continue
        cand = [i for i, t in enumerate(ids) if i >= MIN_POS and t not in special]
        if not cand: n_short += 1; continue
        rng = random.Random(hashlib.sha256(f"{SEED}|{doc_id}".encode()).digest())
        pos = sorted(rng.sample(cand, k=min(N_POS, len(cand))))
        drows.append({"doc_id": doc_id, "ids": ids[:pos[-1] + 1]})
        for p in pos:
            rows.append({"doc_id": doc_id, "n_raw_tokens": p + 1, "text": tok.decode(ids[:p + 1], skip_special_tokens=True), "variant": assign(doc_id)})   # one variant per doc: its positions share the cached prefix
    os.makedirs(f"{root}/pos", exist_ok=True)
    pq.write_table(pa.Table.from_pylist(drows, schema=pa.schema([("doc_id", pa.string()), ("ids", pa.list_(pa.int32()))])), outd, compression="zstd")
    pq.write_table(pa.Table.from_pylist(rows), out, compression="zstd"); vol_glp.commit()
    msg = f"pos {sid}: {len(drows)} docs, {len(rows)} rows, {n_leak} leak-dropped, {n_short} short"; print(msg, flush=True); return msg


@app.function(image=image_g, gpu="B200", timeout=6 * 3600, volumes=VOLS, secrets=SECRETS, cpu=16, memory=128 * 1024, max_containers=8)
def label(sid: int, root: str = ROOT):
    """ROOT/pos/pos_XXXX -> ROOT/lab/lab_XXXX.parquet (doc_id, n_raw_tokens, variant, explanation, n_prompt, n_out); explanation None = unparseable"""
    import pyarrow as pa, pyarrow.parquet as pq
    out = f"{root}/lab/lab_{sid:04d}.parquet"
    vol_glp.reload()
    if _exists(out): return "exists"
    d = pq.read_table(f"{root}/pos/pos_{sid:04d}.parquet").to_pydict()
    global _LLM
    if "_LLM" not in globals(): _LLM = _llm()
    t0 = time.time(); r = _label(_LLM, d["text"], d["variant"], n=LABEL_CFG["n"]); dt = time.time() - t0
    rows = [{"doc_id": d["doc_id"][i], "n_raw_tokens": d["n_raw_tokens"][i], "variant": d["variant"][i], "explanation": r[i][0][0],
             "explanations_extra": r[i][0][1:], "n_prompt": r[i][2], "n_out": r[i][3]} for i in range(len(r))]
    os.makedirs(f"{root}/lab", exist_ok=True); pq.write_table(pa.Table.from_pylist(rows), out, compression="zstd"); vol_glp.commit()
    st = {"sid": sid, "n": len(rows), "seconds": dt, "expl_per_s": len(rows) / dt, "out_tok_per_s": sum(x["n_out"] for x in rows) / dt,
          "prompt_tok_per_s": sum(x["n_prompt"] for x in rows) / dt, "parse_fail": sum(x["explanation"] is None for x in rows) / max(1, len(rows))}
    print("[label]", json.dumps(st), flush=True); return st


def _extract_impl(sid: int, root: str = ROOT):
    import numpy as np, torch, pyarrow as pa, pyarrow.parquet as pq
    from transformers import AutoModelForCausalLM
    from nla.utils.arch_adapters import resolve_decoder_layers
    out = f"{root}/acts/acts_{sid:04d}.parquet"
    vol_glp.reload()
    if _exists(out): return "exists"
    rows = pq.read_table(f"{root}/pos/pos_{sid:04d}.parquet", columns=["doc_id", "n_raw_tokens"]).to_pydict()
    dd = pq.read_table(f"{root}/pos/posdocs_{sid:04d}.parquet").to_pydict(); ids_of = dict(zip(dd["doc_id"], dd["ids"]))
    want = {}
    for i, (doc, n) in enumerate(zip(rows["doc_id"], rows["n_raw_tokens"])): want.setdefault(doc, []).append((n - 1, i))
    global _QM
    if "_QM" not in globals():
        from huggingface_hub import snapshot_download          # raw snapshot on local disk (the volume HF cache served partial snapshots before)
        snap = snapshot_download(BASE, token=os.environ.get("HF_TOKEN"), local_dir="/root/base_snap", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.py"])
        m = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16, device_map="cuda:0", low_cpu_mem_usage=True).eval(); m.requires_grad_(False)
        _QM = (m, resolve_decoder_layers(m))
    model, layers = _QM
    grab = {}

    class _Stop(Exception): pass

    def hook(_m, _i, o):
        grab["h"] = o[0] if isinstance(o, tuple) else o
        raise _Stop
    hd = layers[LAYER].register_forward_hook(hook)
    docs_ = sorted(want, key=lambda k: len(ids_of[k])); acts = np.zeros((len(rows["doc_id"]), 5120), dtype=np.float32)
    budget, i0, t0, ntok = 131072, 0, time.time(), 0          # batch*seq <= 131072 keeps the linear-attention conv1d under 32-bit index math
    try:
        while i0 < len(docs_):
            n = 1
            while i0 + n < len(docs_) and (n + 1) * len(ids_of[docs_[i0 + n]]) <= budget and n < 256: n += 1
            b = docs_[i0:i0 + n]; L = len(ids_of[b[-1]])
            x = torch.zeros(n, L, dtype=torch.long); am = torch.zeros(n, L, dtype=torch.long)
            for j, k in enumerate(b):
                s = ids_of[k]; x[j, :len(s)] = torch.tensor(s); am[j, :len(s)] = 1
            with torch.no_grad():
                try: model(input_ids=x.cuda(), attention_mask=am.cuda(), use_cache=False)
                except _Stop: pass
            h = grab["h"].float()
            for j, k in enumerate(b):
                for p, r in want[k]: acts[r] = h[j, p].cpu().numpy()
            ntok += int(am.sum()); i0 += n
    finally:
        hd.remove()
    dt = time.time() - t0
    os.makedirs(f"{root}/acts", exist_ok=True)
    t = pa.table({"doc_id": pa.array(rows["doc_id"]), "n_raw_tokens": pa.array(rows["n_raw_tokens"]),
                  "activation_vector": pa.FixedSizeListArray.from_arrays(pa.array(acts.reshape(-1)), 5120)})
    pq.write_table(t, out, compression="zstd"); vol_glp.commit()
    st = {"sid": sid, "rows": len(rows["doc_id"]), "docs": len(docs_), "tokens": ntok, "seconds": dt, "tok_per_s": ntok / dt, "rows_per_s": len(rows["doc_id"]) / dt,
          "norm_mean": float(np.linalg.norm(acts, axis=1).mean())}
    print("[extract]", json.dumps(st), flush=True); return st


@app.function(image=image_q, gpu="B200", timeout=6 * 3600, volumes=VOLS, secrets=SECRETS, cpu=16, memory=128 * 1024, max_containers=2)
def extract(sid: int, root: str = ROOT):
    """ROOT/pos -> ROOT/acts/acts_XXXX.parquet: Qwen3.6-27B, one forward per document, output of decoder block 42 at each sampled position"""
    return _extract_impl(sid, root)


@app.function(image=image_q, gpu="B200", timeout=6 * 3600, volumes=VOLS, secrets=SECRETS, cpu=16, memory=128 * 1024, max_containers=6)
def extract_burst(sid: int, root: str = ROOT):
    """same as extract; extra containers once the labeller GPUs are free"""
    return _extract_impl(sid, root)


@app.function(image=image_q, timeout=3600, volumes=VOLS, secrets=SECRETS, cpu=4, memory=48 * 1024, max_containers=16)
def join(sid: int, root: str = ROOT):
    """pos + lab + acts -> ROOT/shards/shard_XXXX.parquet in the raw-extraction schema (+ variant, labeller); unparseable labels dropped"""
    import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
    out = f"{root}/shards/shard_{sid:04d}.parquet"
    vol_glp.reload()
    if _exists(out): return "exists"
    P = pq.read_table(f"{root}/pos/pos_{sid:04d}.parquet"); Lb = pq.read_table(f"{root}/lab/lab_{sid:04d}.parquet", columns=["doc_id", "n_raw_tokens", "explanation"])
    A = pq.read_table(f"{root}/acts/acts_{sid:04d}.parquet")
    for X, nm in ((Lb, "lab"), (A, "acts")):          # every stage writes rows in the positions file's order (arrow joins can't carry fixed-size lists)
        assert X.num_rows == P.num_rows and X.column("doc_id").equals(P.column("doc_id")) and X.column("n_raw_tokens").equals(P.column("n_raw_tokens")), f"{nm} rows misaligned"
    keep = pc.invert(pc.is_null(Lb.column("explanation")))
    n = int(pc.sum(keep).as_py() or 0); f = lambda c: pc.filter(c, keep)
    T = pa.table({"doc_id": f(P.column("doc_id")), "text": f(P.column("text")), "explanation": f(Lb.column("explanation")), "is_val": pa.array([False] * n),
                  "n_raw_tokens": f(P.column("n_raw_tokens")), "activation_layer": pa.array([LAYER] * n), "activation_vector": f(A.column("activation_vector")),
                  "variant": f(P.column("variant")), "labeller": pa.array([LABELLER] * n)})
    os.makedirs(f"{root}/shards", exist_ok=True); pq.write_table(T, out, compression="zstd"); vol_glp.commit()
    return {"sid": sid, "rows": n, "pos_rows": P.num_rows}


@app.function(image=image_q, timeout=24 * 3600, volumes=VOLS, secrets=SECRETS, cpu=2, memory=8 * 1024)
def run(n_docs: int = 1000000, slices: int = 16, tag: str = "g1", max_shards: int = 0):
    """orchestrator: docs workers -> per part positions -> label (8 B200) + extract (2 B200, +6 once labelling is done) -> join. Idempotent: re-running
    resumes from the files on the volume. Progress -> ROOT/status.json + wandb (nla-glp, run scale_gen_<tag>)."""
    import glob, re, wandb
    vol_glp.reload()
    run_ = wandb.init(project="nla-glp", entity="octahedral-systems", name=f"scale_gen_{tag}", id=f"scale_gen_{tag}", resume="allow", config=dict(n_docs=n_docs, slices=slices, labeller=LABELLER))
    R = f"{ROOT}/{tag}"; os.makedirs(R, exist_ok=True)
    if not os.path.exists(f"{R}/docs.started"): docs_all.spawn(n_docs, slices, tag)
    inflight, fails, t0 = {}, {}, time.time()          # (stage, sid) -> FunctionCall
    sid_of = {}                                         # docs part path -> shard id (stable: sorted order of first sight)
    idmap_p = f"{ROOT}/sid_map_{tag}.json"
    if os.path.exists(idmap_p): sid_of = json.load(open(idmap_p))
    while True:
        vol_glp.reload()
        for p in sorted(glob.glob(f"{ROOT}/docs/docs_ffw_{tag}_*.parquet")):
            if p not in sid_of and (not max_shards or len(sid_of) < max_shards): sid_of[p] = len(sid_of)
        json.dump(sid_of, open(idmap_p, "w")); vol_glp.commit()
        have = lambda st, s: _exists(f"{R}/{st}/{st if st != 'shards' else 'shard'}_{s:04d}.parquet")
        n_lab_open = 0
        for p, s in sid_of.items():
            for st, fn, ready in (("pos", positions, True), ("lab", label, have("pos", s)), ("acts", extract, have("pos", s)),
                                  ("shards", join, have("lab", s) and have("acts", s))):
                key = f"{st}|{s}"
                if have(st, s): inflight.pop(key, None); continue
                if not ready: continue
                if st == "lab": n_lab_open += 1
                c = inflight.get(key)
                if c is not None:
                    try: c.get(timeout=0); inflight.pop(key)
                    except TimeoutError: continue
                    except Exception as e:
                        inflight.pop(key); fails[key] = fails.get(key, 0) + 1; print(f"[run] {key} failed #{fails[key]}: {str(e)[:200]}", flush=True)
                if fails.get(key, 0) >= 3: continue
                if key not in inflight and not have(st, s):
                    if st == "acts" and n_lab_open == 0 and sum(1 for k in inflight if k.startswith("acts|")) >= 2: fn = extract_burst
                    inflight[key] = fn.spawn(p, s, R) if st == "pos" else fn.spawn(s, R)
        cnt = {st: len(glob.glob(f"{R}/{st}/{st if st != 'shards' else 'shard'}_*.parquet")) for st in ("pos", "lab", "acts", "shards")}
        docs_done = os.path.exists(f"{R}/docs.done")
        st = dict(t=time.time() - t0, parts=len(sid_of), **cnt, inflight=len(inflight), failed=[k for k, v in fails.items() if v >= 3], docs_done=docs_done)
        json.dump(st, open(f"{ROOT}/status_{tag}.json", "w")); vol_glp.commit(); run_.log({f"scale/{k}": v for k, v in st.items() if isinstance(v, (int, float))})
        print("[run]", json.dumps(st), flush=True)
        if docs_done and cnt["shards"] + len(st["failed"]) >= len(sid_of) and not inflight: break
        time.sleep(60)
    run_.finish(); return st


def _done(c):
    try: c.get(timeout=0); return True
    except TimeoutError: return False
    except Exception: return True


@app.local_entrypoint()
def main(task: str = "pilot", n: int = 3000, nv: int = 400, n_docs: int = 1000000, slices: int = 16, tag: str = "g1", max_shards: int = 0):
    if task == "pilot": print(pilot.remote(n, nv))
    elif task == "run": print(run.remote(n_docs, slices, tag, max_shards))
    elif task == "docs": print(docs_all.remote(n_docs, slices, tag))
    elif task == "bench":
        cfgs = [("A_bf16_mns512", dict(fp8=False, max_num_seqs=512), False),
                ("B_bf16_mns1024_mbt32k", dict(fp8=False, max_num_seqs=1024, max_num_batched_tokens=32768), False),
                ("C_B_perdoc", dict(fp8=False, max_num_seqs=1024, max_num_batched_tokens=32768), True),
                ("D_C_fp8", dict(fp8=True, fp8_kv=True, max_num_seqs=1024, max_num_batched_tokens=32768), True),
                ("E_D_n2", dict(fp8=True, fp8_kv=True, max_num_seqs=1024, max_num_batched_tokens=32768, n=2), True),
                ("F_D_n4", dict(fp8=True, fp8_kv=True, max_num_seqs=1024, max_num_batched_tokens=32768, n=4), True)]
        calls = [bench.spawn(nm, c, pd) for nm, c, pd in cfgs]
        res = []
        for c in calls:
            try: res.append(c.get())
            except Exception as e: res.append({"error": str(e)[:300]})
        print("BENCH_RESULTS " + json.dumps(res))
    elif task == "smoke_qwen":                     # docs -> positions -> extract on one 5k-doc part (sid 9000), no labeller
        import glob as _g
        print("docs rc", docs.remote("0/1", 5000, "smoke"))
        print(positions.remote(f"{ROOT}/docs/docs_ffw_smoke_s00_000.parquet", 9000))
        print(extract.remote(9000))
