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
image_g = (modal.Image.from_registry("vllm/vllm-openai:v0.30.0", setup_dockerfile_commands=["RUN ln -sf $(which python3) /usr/local/bin/python"]).entrypoint([])   # 0.30: FlashInfer attention for Gemma-4 (engine optimiser, +65%)
           .run_commands("pip install --no-cache-dir pyarrow wandb 'huggingface_hub[hf_xet]'")
           .env({"HF_HOME": "/vol_glp/hf", "HF_XET_HIGH_PERFORMANCE": "1", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
                 "PYTHONPATH": REPO_REMOTE, "VLLM_ALLOW_INSECURE_SERIALIZATION": "1", "VLLM_CACHE_ROOT": "/vol_glp/scale/vllm_cache/v30"})
           .add_local_dir(REPO_LOCAL, REPO_REMOTE, copy=False, ignore=REPO_IGNORE))
app = modal.App("nla-scale")


# ------------------------------------------------------------------------------------------------------------ labeller (Gemma, vLLM)
# production labeller config, from the 6-way bench (20k real prompts, 1 B200 each): bigger batches +8 %, one variant per document +3 % prompts/s at
# 10 % longer outputs; fp8 weights+KV -4 % and 8x parse failures (decode-bound on the MoE/attention kernels, not weight bandwidth); n>1 gives no
# extra explanations per second (127 at n=2 vs 126) -> bf16, 1024 seqs, 32k batched tokens, n=1
LABEL_CFG = dict(fp8=False, fp8_kv=False, max_num_seqs=1024, max_num_batched_tokens=32768, n=1, attention_backend="TRITON_FLASHINFER")   # backend: [gemma-engine] 03:40 win, quality-guarded


def _engine_kwargs(cfg=None):
    c = {**LABEL_CFG, **(cfg or {})}
    kw = dict(model=LABELLER, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.90, limit_mm_per_prompt={"image": 0},
              max_num_seqs=c["max_num_seqs"], enable_prefix_caching=True, seed=0)
    if c.get("max_num_batched_tokens"): kw["max_num_batched_tokens"] = c["max_num_batched_tokens"]
    if c.get("attention_backend"): kw["attention_backend"] = c["attention_backend"]
    if c.get("fp8"): kw["quantization"] = "fp8"
    if c.get("fp8_kv"): kw["kv_cache_dtype"] = "fp8"
    return kw


def _llm(cfg=None):
    from vllm import LLM
    return LLM(**_engine_kwargs(cfg))


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


# ------------------------------------------------------------------------------------------------------------ g2: fact sheet -> factorised renderings
G2_K = 4          # renderings per position


def _g2_lib():
    p = f"{REPO_REMOTE}/nla/datagen/g2_library.json"
    return json.load(open(p)) if os.path.exists(p) else None


class _AsyncGen:
    """AsyncLLM on a persistent per-container event loop, pre-tokenised requests: the chat template + tokenisation run in a worker thread in
    500-prompt batches, so the engine gets work while later prompts are still being encoded ([gemma-pipeline] g2_async, pipeline=False:
    +5.6% positions/s with identical outputs vs llm.chat, which tokenises ~100k prompts single-threaded first)."""
    def __init__(self, cfg=None):
        import asyncio, threading
        from transformers import AutoTokenizer
        from vllm.engine.arg_utils import AsyncEngineArgs
        try: from vllm.v1.engine.async_llm import AsyncLLM
        except ImportError: from vllm import AsyncLLMEngine as AsyncLLM
        self.tok = AutoTokenizer.from_pretrained(LABELLER, token=os.environ.get("HF_TOKEN")); self.n = 0
        self.loop = asyncio.new_event_loop(); threading.Thread(target=self.loop.run_forever, daemon=True).start()
        kw = _engine_kwargs(cfg)
        async def mk(): return AsyncLLM.from_engine_args(AsyncEngineArgs(**kw))
        self.engine = asyncio.run_coroutine_threadsafe(mk(), self.loop).result()

    def _encode(self, contents):
        return self.tok([self.tok.apply_chat_template([{"role": "user", "content": c}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                         for c in contents], add_special_tokens=False)["input_ids"]

    def generate(self, contents, sps):
        """contents: user messages; sps: one SamplingParams or a list -> [(text, n_prompt_tokens, n_out_tokens)] in order"""
        import asyncio
        from vllm.sampling_params import RequestOutputKind
        sps = sps if isinstance(sps, list) else [sps] * len(contents)
        async def one(ids, sp, rid):
            final = None
            async for o in self.engine.generate({"prompt_token_ids": ids}, sp, rid): final = o
            return final
        async def run():
            loop = asyncio.get_running_loop(); tasks = []
            for b0 in range(0, len(contents), 500):
                ids = await loop.run_in_executor(None, self._encode, contents[b0:b0 + 500])
                for j, pid in enumerate(ids):
                    sp = sps[b0 + j].clone(); sp.output_kind = RequestOutputKind.FINAL_ONLY; self.n += 1
                    tasks.append(asyncio.ensure_future(one(pid, sp, f"r{self.n}")))
            outs = await asyncio.gather(*tasks)
            return [(o.outputs[0].text, len(o.prompt_token_ids), len(o.outputs[0].token_ids)) for o in outs]
        return asyncio.run_coroutine_threadsafe(run(), self.loop).result()


def _sync_gen(llm):
    """llm.chat backend with the same interface as _AsyncGen.generate"""
    def generate(contents, sps):
        res = llm.chat([[{"role": "user", "content": c}] for c in contents], sps, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
        return [(r.outputs[0].text, len(r.prompt_token_ids), len(r.outputs[0].token_ids)) for r in res]
    return generate


def _g2_rows(llm, texts, keys, docs, k=G2_K):
    """stage A fact sheets (sampled context window) + validation, programmatic ladders/twins, stage B k renderings at sampled style points + QC.
    -> (records aligned with texts, timing stats)"""
    import random
    from vllm import SamplingParams
    from nla.datagen import g2_spec as G
    lib = _g2_lib(); gen = llm.generate if isinstance(llm, _AsyncGen) else _sync_gen(llm)
    wins = [G._pick(G.WINDOWS, random.Random(f"w|{key}")) for key in keys]
    ctxs = [G.window(t, w) for t, w in zip(texts, wins)]
    t0 = time.time()
    resA = gen([G.FACT_PROMPT.format(text=c) for c in ctxs], SamplingParams(temperature=0.3, top_p=0.95, max_tokens=900))
    tA = time.time() - t0; nA_in = sum(r[1] for r in resA); nA_out = sum(r[2] for r in resA)
    facts, vstats = [], []
    n_bad = 0
    for r, c in zip(resA, ctxs):
        f = G.parse_facts(r[0])
        if f is None: facts.append(None); vstats.append(None); continue
        try: v, st = G.validate(f, c); G.fact_list(v)
        except Exception: n_bad += 1; facts.append(None); vstats.append(None); continue          # one malformed fact sheet must never fail the shard
        facts.append(v); vstats.append(st)
    pool = {}
    for d, v in zip(docs, facts):
        if v is None: continue
        for x in G.fact_list(v): pool.setdefault(x.get("etype") or x["type"], []).append((d, x["value"]))      # twins of the same subtype
    jobs, plans = [], []
    for i, (key, d, v) in enumerate(zip(keys, docs, facts)):
        if v is None: continue
        rng = random.Random(f"s|{key}"); fl = G.fact_list(v); tw = []
        for x in fl:
            cands = pool.get(x.get("etype") or x["type"], []); tv = None
            for _ in range(8):
                if not cands: break
                dd, vv = rng.choice(cands)
                if dd != d and vv != x["value"]: tv = vv; break
            tw.append(tv)
        for j in range(k):
            try: st_ = G.sample_style(rng); pr, plan = G.build_render(v, fl, tw, st_, lib, rng)
            except Exception: n_bad += 1; continue
            jobs.append((i, j, pr, G.RENDER_MAX_TOKENS[st_["length"]])); plans.append((st_, plan, fl, tw))
    t0 = time.time()
    resB = gen([pr for _, _, pr, _ in jobs], [SamplingParams(temperature=0.9, top_p=0.95, max_tokens=mt) for *_, mt in jobs])
    tB = time.time() - t0; nB_in = sum(r[1] for r in resB); nB_out = sum(r[2] for r in resB)
    recs = [{"window": wins[i], "facts": None if facts[i] is None else json.dumps(facts[i], ensure_ascii=False), "claims": G.claims(facts[i]) if facts[i] else [],
             "fact_ladders": None, "renders": [], "styles": [], "qc": [], "validate": json.dumps(vstats[i]) if vstats[i] else None} for i in range(len(texts))]
    for (i, j, _, _), r, (st_, plan, fl, tw) in zip(jobs, resB, plans):
        txt = G.parse_render(r[0])
        recs[i]["renders"].append(txt); recs[i]["styles"].append(json.dumps(st_))
        recs[i]["qc"].append(json.dumps(G.qc_render(txt, fl, plan, ctxs[i], tw) if txt else {"parse_fail": 1}))
        if recs[i]["fact_ladders"] is None: recs[i]["fact_ladders"] = json.dumps([{**x, "twin": t} for x, t in zip(fl, tw)], ensure_ascii=False)
    for rc in recs:                             # canonical explanation: first rendering passing every deterministic check
        good = [t for t, q in zip(rc["renders"], rc["qc"]) if t and not any(json.loads(q).get(kk, 0) for kk in ("exact_missing", "leaked", "unsupported_numbers", "parse_fail"))]
        rc["explanation"] = good[0] if good else None; rc["n_pass"] = len(good)          # no passing rendering -> the position is dropped at join
    stats = dict(n=len(texts), facts_ok=sum(f is not None for f in facts), malformed_skipped=n_bad, stageA_s=tA, stageA_prefill_tok_s=nA_in / tA, stageA_decode_tok_s=nA_out / tA,
                 stageA_mean_out=nA_out / len(texts), renders=len(jobs), stageB_s=tB, stageB_prefill_tok_s=nB_in / tB, stageB_decode_tok_s=nB_out / tB,
                 stageB_mean_out=nB_out / max(1, len(jobs)), positions_per_s=len(texts) / (tA + tB), renders_per_s=len(jobs) / (tA + tB))
    return recs, stats


@app.function(image=image_g, gpu="B200", timeout=4 * 3600, volumes=VOLS, secrets=SECRETS, cpu=16, memory=128 * 1024)
def g2_pilot(overlap_rows: list, n_new: int = 4600, src: str = f"{ROOT}/g1", name: str = "g2_pilot"):
    """g2 on the pilot's Opus-overlap rows (av_sft_val) + n_new fresh g1 positions -> ROOT/g2pilot/g2_pilot.parquet + throughput.json"""
    import glob, pyarrow as pa, pyarrow.parquet as pq
    vol_glp.reload()
    t = pq.read_table("/vol_q36/data/sft/av_sft_val.parquet", columns=["doc_id", "detokenized_text_truncated", "response", "n_raw_tokens"]).take(overlap_rows).to_pydict()
    rows = [{"src": "opus_overlap", "row": r, "doc_id": d, "n_raw_tokens": int(n), "text": x, "opus": o} for r, d, x, o, n in
            zip(overlap_rows, t["doc_id"], t["detokenized_text_truncated"], t["response"], t["n_raw_tokens"])]
    for f in sorted(glob.glob(f"{src}/pos/pos_*.parquet")):
        d = pq.read_table(f).to_pydict()
        rows += [{"src": "g1_fresh", "row": -1, "doc_id": a, "n_raw_tokens": int(b), "text": c, "opus": None} for a, b, c in zip(d["doc_id"], d["n_raw_tokens"], d["text"])][: n_new - (len(rows) - len(overlap_rows))]
        if len(rows) - len(overlap_rows) >= n_new: break
    t0 = time.time(); llm = _llm(); t_load = time.time() - t0
    recs, st = _g2_rows(llm, [r["text"] for r in rows], [f"{r['doc_id']}|{r['n_raw_tokens']}" for r in rows], [r["doc_id"] for r in rows])
    st["load_s"] = t_load; st["library"] = _g2_lib() is not None
    os.makedirs(f"{ROOT}/g2pilot", exist_ok=True)
    pq.write_table(pa.Table.from_pylist([{**r, **x} for r, x in zip(rows, recs)]), f"{ROOT}/g2pilot/{name}.parquet", compression="zstd")
    json.dump(st, open(f"{ROOT}/g2pilot/{name}_throughput.json", "w"), indent=1); vol_glp.commit()
    print("[g2pilot]", json.dumps(st), flush=True); return st


@app.function(image=image_g, gpu="B200", timeout=8 * 3600, volumes=VOLS, secrets=SECRETS, cpu=16, memory=128 * 1024, max_containers=8)
def g2_label(sid: int, src: str, out_root: str):
    """g2 labels for the positions of <src>/pos/pos_XXXX (activations are shared with the source run) -> <out_root>/lab/lab_XXXX.parquet"""
    import pyarrow as pa, pyarrow.parquet as pq
    out = f"{out_root}/lab/lab_{sid:04d}.parquet"
    vol_glp.reload()
    if _exists(out): return "exists"
    d = pq.read_table(f"{src}/pos/pos_{sid:04d}.parquet", columns=["doc_id", "n_raw_tokens", "text"]).to_pydict()
    global _AGEN
    if "_AGEN" not in globals(): _AGEN = _AsyncGen()
    recs, st = _g2_rows(_AGEN, d["text"], [f"{a}|{b}" for a, b in zip(d["doc_id"], d["n_raw_tokens"])], d["doc_id"])
    import vllm as _v; eng = f"vllm-{_v.__version__}-{LABEL_CFG.get('attention_backend') or 'auto'}-asyncpretok"
    rows = [{"doc_id": a, "n_raw_tokens": b, **r, "engine": eng} for a, b, r in zip(d["doc_id"], d["n_raw_tokens"], recs)]
    st["engine"] = eng
    os.makedirs(f"{out_root}/lab", exist_ok=True); pq.write_table(pa.Table.from_pylist(rows), out, compression="zstd"); vol_glp.commit()
    st["sid"] = sid; print("[g2label]", json.dumps(st), flush=True); return st


@app.function(image=image_g, gpu="B200", timeout=3600, volumes=VOLS, secrets=SECRETS, cpu=16, memory=128 * 1024)
def g2_async_smoke(n: int = 2000):
    """the async pre-tokenised labeller on n real g1 positions (two back-to-back calls = the per-container engine reuse across shards)"""
    import pyarrow.parquet as pq
    d = pq.read_table(f"{ROOT}/g1/pos/pos_0000.parquet", columns=["doc_id", "n_raw_tokens", "text"]).slice(0, n).to_pydict()
    keys = [f"{a}|{b}" for a, b in zip(d["doc_id"], d["n_raw_tokens"])]
    t0 = time.time(); g = _AsyncGen(); load = time.time() - t0
    h = n // 2; r1, s1 = _g2_rows(g, d["text"][:h], keys[:h], d["doc_id"][:h]); r2, s2 = _g2_rows(g, d["text"][h:], keys[h:], d["doc_id"][h:])
    recs = r1 + r2; ren = [t for r in recs for t in r["renders"]]
    out = {"load_s": load, "call1": s1, "call2": s2, "render_parse": sum(t is not None for t in ren) / max(1, len(ren)), "facts_ok": sum(r["facts"] is not None for r in recs) / len(recs),
           "positions_passing": sum(r["n_pass"] > 0 for r in recs) / len(recs), "example": next((t for t in ren if t), None)}
    print("[g2smoke]", json.dumps(out)[:3000], flush=True); return out


@app.function(image=image_q, timeout=3600, volumes=VOLS, secrets=SECRETS, cpu=8, memory=64 * 1024)
def g2_qc_stats(root: str = f"{ROOT}/g2"):
    """per-engine QC of the g2 label shards: rendering parse rate, fact-sheet parse rate, deterministic pass rates, positions passing, mean words"""
    import glob, json as _j, pyarrow.parquet as pq, collections
    vol_glp.reload(); agg = collections.defaultdict(lambda: collections.Counter()); shards = collections.defaultdict(list)
    for f in sorted(glob.glob(f"{root}/lab/lab_*.parquet")):
        cols = pq.ParquetFile(f).schema_arrow.names
        t = pq.read_table(f, columns=[c for c in ("renders", "qc", "facts", "n_pass", "engine") if c in cols]).to_pydict()
        eng = (t.get("engine") or ["vllm-0.29.0-auto"])[0]; shards[eng].append(os.path.basename(f)); a = agg[eng]
        for rs, qs, fa, npass in zip(t["renders"], t["qc"], t["facts"], t["n_pass"]):
            a["positions"] += 1; a["facts_ok"] += fa is not None; a["positions_passing"] += (npass or 0) > 0; a["passing_renders"] += npass or 0
            for r, q in zip(rs or [], qs or []):
                a["renders"] += 1; a["parsed"] += r is not None
                if r is None: continue
                q = _j.loads(q); a["exact_missing"] += q.get("exact_missing", 0) > 0; a["leaked"] += q.get("leaked", 0) > 0; a["unsupported_numbers"] += q.get("unsupported_numbers", 0) > 0; a["words"] += q.get("words", 0)
    out = {}
    for eng, a in agg.items():
        P, Rn, Pa = max(1, a["positions"]), max(1, a["renders"]), max(1, a["parsed"])
        out[eng] = {"shards": len(shards[eng]), "positions": a["positions"], "fact_sheet_parse": a["facts_ok"] / P, "render_parse": a["parsed"] / Rn,
                    "exact_missing": a["exact_missing"] / Pa, "leaked": a["leaked"] / Pa, "unsupported_numbers": a["unsupported_numbers"] / Pa,
                    "positions_with_passing_render": a["positions_passing"] / P, "passing_renders_per_position": a["passing_renders"] / P, "mean_words": a["words"] / Pa}
    return out


@app.function(image=image_q, timeout=3600, volumes=VOLS, secrets=SECRETS, cpu=4, memory=48 * 1024, max_containers=16)
def g2_join(sid: int, src: str, out_root: str):
    """<src> pos + acts + <out_root> g2 labels -> <out_root>/shards/shard_XXXX.parquet: trainer schema (explanation = first rendering passing every
    deterministic check) + explanations / styles / qc (k renderings per activation) + window, facts, claims (compositional one-claim-per-fact column),
    fact_ladders (specificity ladder + wrong-exact twin per fact)"""
    import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
    out = f"{out_root}/shards/shard_{sid:04d}.parquet"
    vol_glp.reload()
    if _exists(out): return "exists"
    P = pq.read_table(f"{src}/pos/pos_{sid:04d}.parquet"); Lb = pq.read_table(f"{out_root}/lab/lab_{sid:04d}.parquet"); A = pq.read_table(f"{src}/acts/acts_{sid:04d}.parquet")
    for X, nm in ((Lb, "lab"), (A, "acts")):
        assert X.num_rows == P.num_rows and X.column("doc_id").equals(P.column("doc_id")) and X.column("n_raw_tokens").equals(P.column("n_raw_tokens")), f"{nm} rows misaligned"
    keep = pc.invert(pc.is_null(Lb.column("explanation"))); n = int(pc.sum(keep).as_py() or 0); f = lambda c: pc.filter(c, keep)
    cols = {"doc_id": f(P.column("doc_id")), "text": f(P.column("text")), "explanation": f(Lb.column("explanation")), "is_val": pa.array([False] * n),
            "n_raw_tokens": f(P.column("n_raw_tokens")), "activation_layer": pa.array([LAYER] * n), "activation_vector": f(A.column("activation_vector")),
            "variant": pa.array(["g2"] * n), "labeller": pa.array([LABELLER] * n)}
    for c in ("renders", "styles", "qc", "window", "facts", "claims", "fact_ladders", "n_pass"): cols[{"renders": "explanations"}.get(c, c)] = f(Lb.column(c))
    os.makedirs(f"{out_root}/shards", exist_ok=True); pq.write_table(pa.table(cols), out, compression="zstd"); vol_glp.commit()
    return {"sid": sid, "rows": n, "pos_rows": P.num_rows}


@app.function(image=image_q, timeout=24 * 3600, volumes=VOLS, secrets=SECRETS, cpu=2, memory=8 * 1024)
def run_g2(src_tag: str = "g1", tag: str = "g2", max_shards: int = 0):
    """g2 orchestrator over the source run's positions: g2 labels (8 B200) + extraction of any missing source activations (2 B200, +6 when no
    g2 labelling is pending) + g2 joins + joins of the source run's own labelled shards (so the source slice completes after its labeller stops)"""
    import glob, wandb
    S, R = f"{ROOT}/{src_tag}", f"{ROOT}/{tag}"; os.makedirs(R, exist_ok=True)
    run_ = wandb.init(project="nla-glp", entity="octahedral-systems", name=f"scale_gen_{tag}", id=f"scale_gen_{tag}", resume="allow", config=dict(src=src_tag, labeller=LABELLER, k=G2_K))
    inflight, fails, t0 = {}, {}, time.time()
    have = lambda root, st, s: _exists(f"{root}/{st}/{st if st != 'shards' else 'shard'}_{s:04d}.parquet")
    while True:
        vol_glp.reload()
        sids = sorted(int(os.path.basename(p)[4:8]) for p in glob.glob(f"{S}/pos/pos_*.parquet"))
        if max_shards: sids = sids[:max_shards]
        n_lab_open = sum(1 for s in sids if not have(R, "lab", s))
        for s in sids:
            for key, done, ready, fn, args in ((f"g2lab|{s}", have(R, "lab", s), True, g2_label, (s, S, R)),
                                               (f"acts|{s}", have(S, "acts", s), True, extract, (s, S)),
                                               (f"g2join|{s}", have(R, "shards", s), have(R, "lab", s) and have(S, "acts", s), g2_join, (s, S, R)),
                                               (f"srcjoin|{s}", have(S, "shards", s), have(S, "lab", s) and have(S, "acts", s), join, (s, S))):
                if done: inflight.pop(key, None); continue
                if not ready: continue
                c = inflight.get(key)
                if c is not None:
                    try: c.get(timeout=0); inflight.pop(key)
                    except TimeoutError: continue
                    except Exception as e:
                        inflight.pop(key); fails[key] = fails.get(key, 0) + 1; print(f"[run_g2] {key} failed #{fails[key]}: {str(e)[:200]}", flush=True)
                if fails.get(key, 0) >= 3 or key in inflight: continue
                if key.startswith("g2lab|") and sum(1 for k in inflight if k.startswith("g2lab|")) >= 8: continue   # never queue beyond the 8 labellers:
                # queued inputs stay bound to the deploy version they were spawned under, so a capped queue lets redeploys take effect at shard boundaries
                if key.startswith("acts|") and n_lab_open == 0 and sum(1 for k in inflight if k.startswith("acts|")) >= 2: fn = extract_burst
                inflight[key] = fn.spawn(*args)
        cnt = dict(pos=len(sids), g2_lab=sum(have(R, "lab", s) for s in sids), acts=sum(have(S, "acts", s) for s in sids), g2_shards=sum(have(R, "shards", s) for s in sids),
                   src_lab=sum(have(S, "lab", s) for s in sids), src_shards=sum(have(S, "shards", s) for s in sids))
        st = dict(t=time.time() - t0, **cnt, inflight=len(inflight), failed=[k for k, v in fails.items() if v >= 3])
        json.dump(st, open(f"{ROOT}/status_{tag}.json", "w")); vol_glp.commit(); run_.log({f"scale/{k}": v for k, v in st.items() if isinstance(v, (int, float))})
        print("[run_g2]", json.dumps(st), flush=True)
        if cnt["g2_shards"] + sum(1 for k in st["failed"] if k.startswith("g2")) >= len(sids) and not any(k.startswith(("g2", "acts")) for k in inflight): break
        time.sleep(60)
    run_.finish(); return st


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
            while i0 + n < len(docs_) and (n + 1) * (-(-len(ids_of[docs_[i0 + n]]) // 256) * 256) <= budget and n < 256: n += 1
            b = docs_[i0:i0 + n]; L = -(-len(ids_of[b[-1]]) // 256) * 256              # round up to x256: bounded set of autotuned (batch, length) shapes
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
    elif task == "g2asyncsmoke": print(json.dumps(g2_async_smoke.remote(n))[:3000])
    elif task == "g2pilot":
        rows = sorted(int(k.split("|")[1]) for k in json.load(open("/home/celeste/shared/reports/nla-flow-prior/data/scale/pilot_judge.json"))["rows"] if k.startswith("opus|"))
        print(g2_pilot.remote(rows, n, f"{ROOT}/g1", tag))
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
