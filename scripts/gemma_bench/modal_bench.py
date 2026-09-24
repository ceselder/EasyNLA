"""Inference-engine hill-climb for the Gemma-4-26B-A4B labeller on Modal B200s (own app `gemma-bench-engine`; production app nla-scale untouched).

Fixed benchmark set (built once, reused by every config; on the nla-glp volume under /vol_glp/scale/bench/set_v1):
  plain  20k real labeller prompts (g1 positions shard, one variant per document as in production, V0/v2/v3/v4 instructions, max_tokens per variant)
  g2a    4k stage-A fact-sheet prompts (sampled context windows, temperature 0.3, max_tokens 900)
  g2b    16k stage-B rendering prompts (4 per position, built from the g2 pilot's validated fact sheets with the real style sampler + library)
  judge  the 400 pilot V0 rows that have Opus-5 gold + a Sonnet-5 claim judgement (quality guard: judged locally with scripts/gemma_bench/judge.py)

  modal run scripts/gemma_bench/modal_bench.py --task build
  modal run scripts/gemma_bench/modal_bench.py --task probe --engine v30
  modal run --detach scripts/gemma_bench/modal_bench.py --task run --engine v29 --name B0_prod --cfg @scripts/gemma_bench/cfgs/B0_prod.json
Results: /vol_glp/scale/bench/results/<name>.json (+ <name>_judge.parquet), logs: /vol_glp/scale/bench/logs/<name>.log
"""
import os, sys, json, time, threading, shutil
import modal

REPO_LOCAL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_REMOTE = "/root/easyNLA"
REPO_IGNORE = [".git", ".venv", "__pycache__", "*.pyc", "*.parquet"]
LABELLER = "google/gemma-4-26B-A4B-it"
BENCH = "/vol_glp/scale/bench"
SET = f"{BENCH}/set_v1"
PROD_HF = "/vol_glp/hf/hub"                      # production's HF cache (read-only for us: the base Gemma snapshot lives here)
BENCH_HF = f"{BENCH}/hf/hub"                     # extra checkpoints we download (NVFP4 / FP8 / drafters / DeepSeek)
ENV = {"HF_HOME": "/root/hf", "HF_HUB_CACHE": "/root/hf/hub", "HF_XET_HIGH_PERFORMANCE": "1", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
       "PYTHONPATH": REPO_REMOTE, "REPO_REMOTE": REPO_REMOTE, "VLLM_ALLOW_INSECURE_SERIALIZATION": "1"}
PIP = "pip install --no-cache-dir pyarrow aiohttp 'huggingface_hub[hf_xet]'"


def _img(base):
    return (modal.Image.from_registry(base, setup_dockerfile_commands=["RUN ln -sf $(which python3) /usr/local/bin/python"]).entrypoint([])
            .run_commands(PIP).env(ENV).add_local_dir(REPO_LOCAL, REPO_REMOTE, copy=False, ignore=REPO_IGNORE))


IMAGES = {"v29": "vllm/vllm-openai:v0.29.0", "v30": "vllm/vllm-openai:v0.30.0", "sgl": "lmsysorg/sglang:v0.5.20-cu130"}
img_v29, img_v30, img_sgl = (_img(IMAGES[k]) for k in ("v29", "v30", "sgl"))
vol_glp = modal.Volume.from_name("nla-glp")
VOLS = {"/vol_glp": vol_glp}
SECRETS = [modal.Secret.from_name("nla-exp-secrets")]
app = modal.App("gemma-bench-engine")


# ------------------------------------------------------------------------------------------------------------ helpers (in-container)
def _prep_hf(models):
    """HF cache = local dir of symlinks: base Gemma from production's cache, everything else downloaded once into BENCH_HF"""
    from huggingface_hub import snapshot_download
    os.makedirs("/root/hf/hub", exist_ok=True); os.makedirs(BENCH_HF, exist_ok=True)
    for m in models:
        d = "models--" + m.replace("/", "--")
        for src in (f"{PROD_HF}/{d}", f"{BENCH_HF}/{d}"):
            if os.path.isdir(src) and os.path.isdir(f"{src}/snapshots"): break
        else:
            t0 = time.time(); print(f"[hf] downloading {m} -> {BENCH_HF}", flush=True)
            snapshot_download(m, cache_dir=BENCH_HF, token=os.environ.get("HF_TOKEN"), allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.py", "*.model", "*.tiktoken"])
            vol_glp.commit(); print(f"[hf] {m} downloaded in {time.time() - t0:.0f}s", flush=True); src = f"{BENCH_HF}/{d}"
        dst = f"/root/hf/hub/{d}"
        if not os.path.lexists(dst): os.symlink(src, dst)


def _tee(proc_log, vol_log, stop):
    while not stop.is_set():
        stop.wait(90)
        try: shutil.copy(proc_log, vol_log); vol_glp.commit()
        except Exception as e: print("[tee]", e, flush=True)


def _run(name, cfg):
    """cfg["variants"] = [[suffix, overrides], ...] runs several engine configs in ONE container (shared checkpoint download / disk)"""
    if cfg.get("variants"):
        out = {}
        for suf, ov in cfg["variants"]:
            c = {k: v for k, v in cfg.items() if k != "variants"}
            for k, v in ov.items():
                c[k] = {**c[k], **v} if isinstance(v, dict) and isinstance(c.get(k), dict) else v
            try: out[f"{name}_{suf}"] = _run_one(f"{name}_{suf}", c)
            except Exception as e: out[f"{name}_{suf}"] = {"error": str(e)[:300]}
        return out
    return _run_one(name, cfg)


def _run_one(name, cfg):
    import subprocess
    vol_glp.reload(); _prep_hf(cfg.get("hf_models", [LABELLER]))
    os.makedirs(f"{BENCH}/results", exist_ok=True); os.makedirs(f"{BENCH}/logs", exist_ok=True)
    out, vlog, plog = f"{BENCH}/results/{name}.json", f"{BENCH}/logs/{name}.log", f"/tmp/{name}.log"
    env = {**os.environ, **cfg.get("env", {})}
    if cfg.get("profile"):                                                              # vLLM >= 0.29: profiler is a config, not an env var
        env["VLLM_TORCH_PROFILER_DIR"] = "/tmp/prof"
        cfg = {**cfg, "engine_kwargs": {**cfg["engine_kwargs"], "profiler_config": {"profiler": "torch", "torch_profiler_dir": "/tmp/prof"}}}
    cfg = {**cfg, "server_log": plog}
    cmd = [sys.executable, f"{REPO_REMOTE}/scripts/gemma_bench/worker.py", "--set-dir", SET, "--out", out, "--cfg", json.dumps(cfg)]
    print("[run]", name, json.dumps({k: v for k, v in cfg.items() if k != "hf_models"})[:1500], flush=True)
    stop = threading.Event(); th = threading.Thread(target=_tee, args=(plog, vlog, stop), daemon=True); th.start()
    t0 = time.time()
    with open(plog, "ab") as lf:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=REPO_REMOTE)
        for line in p.stdout:
            lf.write(line)
            s = line.decode(errors="replace")
            if "[worker]" in s or "Error" in s or "error" in s[:40] or "Traceback" in s or "Using" in s[:80]: print(s.rstrip()[:400], flush=True)
        rc = p.wait()
    stop.set(); shutil.copy(plog, vlog)
    res = json.load(open(out)) if os.path.exists(out) else {"error": f"worker rc {rc}"}
    res.update({"name": name, "rc": rc, "wall_seconds": time.time() - t0}); json.dump(res, open(out, "w"), indent=1); vol_glp.commit()
    print("[run] done", name, "rc", rc, json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items() if kk in ("prompts_per_s", "decode_tok_per_s", "parse_fail")}
                                                    for k, v in res.get("sets", {}).items()}), flush=True)
    return res


FN = dict(image=img_v29, timeout=3 * 3600, volumes=VOLS, secrets=SECRETS, cpu=16, memory=128 * 1024)


@app.function(gpu="B200", **FN)
def run_v29(name: str, cfg: dict): return _run(name, cfg)


@app.function(gpu="B200", **{**FN, "image": img_v30})
def run_v30(name: str, cfg: dict): return _run(name, cfg)


@app.function(gpu="B200", **{**FN, "image": img_sgl})
def run_sgl(name: str, cfg: dict): return _run(name, cfg)


@app.function(gpu="B200:8", **{**FN, "image": img_v30, "cpu": 64, "memory": 640 * 1024, "timeout": 4 * 3600})
def run_v30_x8(name: str, cfg: dict): return _run(name, cfg)


@app.function(gpu="B200:8", **{**FN, "image": img_sgl, "cpu": 64, "memory": 640 * 1024, "timeout": 4 * 3600})
def run_sgl_x8(name: str, cfg: dict): return _run(name, cfg)


RUNNERS = {"v29": run_v29, "v30": run_v30, "sgl": run_sgl, "v30x8": run_v30_x8, "sglx8": run_sgl_x8}


@app.function(**{**FN, "cpu": 8, "memory": 32 * 1024, "timeout": 4 * 3600})
def prefetch(model: str):
    """download one checkpoint into BENCH_HF once (shared by every later run)"""
    vol_glp.reload(); t0 = time.time(); _prep_hf([model]); return {"model": model, "seconds": time.time() - t0}


# ------------------------------------------------------------------------------------------------------------ probe: what does this engine support?
def _probe():
    import subprocess, glob, importlib
    out = {}
    try:
        import vllm; out["vllm"] = vllm.__version__; root = os.path.dirname(vllm.__file__)
        out["moe_configs_E128_N704"] = sorted(os.path.basename(p) for p in glob.glob(f"{root}/model_executor/layers/fused_moe/configs/E=128,N=704,*"))
        out["moe_configs_E128_N1408"] = sorted(os.path.basename(p) for p in glob.glob(f"{root}/model_executor/layers/fused_moe/configs/E=128,N=1408,*"))
        g = subprocess.run(["grep", "-rl", "Gemma4Assistant\\|gemma4_assistant", root], capture_output=True, text=True).stdout.split()
        out["gemma4_mtp_files"] = [p.replace(root, "") for p in g][:10]
        g = subprocess.run(["grep", "-rho", "VLLM_USE_FLASHINFER_MOE[A-Z0-9_]*\\|VLLM_FLASHINFER_MOE_BACKEND\\|VLLM_USE_TRTLLM_ATTENTION\\|VLLM_USE_FLASHINFER_SAMPLER\\|VLLM_ATTENTION_BACKEND", f"{root}/envs.py"], capture_output=True, text=True).stdout.split()
        out["envs"] = sorted(set(g))
        g = subprocess.run(["grep", "-rn", "moe_backend", f"{root}/config/kernel.py"], capture_output=True, text=True).stdout
        out["kernel_config_moe_backend"] = g[:1500]
        g = subprocess.run(["grep", "-rn", "class MoEBackend\\|= \"", f"{root}/model_executor/layers/fused_moe/config.py"], capture_output=True, text=True).stdout
        out["moe_backend_enum"] = g[:1500]
        g = subprocess.run(["grep", "-rn", "head_size=512\\|TMEM\\|heterogeneous head", f"{root}/attention/utils/fa_utils.py", f"{root}/model_executor/models/gemma4.py", f"{root}/config/model.py"], capture_output=True, text=True).stdout
        out["fa_512"] = g[:1500]
        g = subprocess.run(["grep", "-rn", "dflash\\|dspark", f"{root}/config/speculative.py"], capture_output=True, text=True).stdout
        out["spec_methods"] = g[:1200]
        g = subprocess.run(["bash", "-c", f"ls {root}/v1/attention/backends/ | head -50"], capture_output=True, text=True).stdout
        out["attn_backends"] = g.split()
        for mod in ("flashinfer", "flash_attn", "deep_gemm", "transformers"):
            try: out[f"ver_{mod}"] = importlib.import_module(mod).__version__
            except Exception as e: out[f"ver_{mod}"] = f"ERR {str(e)[:60]}"
    except Exception as e: out["vllm_err"] = str(e)[:300]
    try:
        import sglang; out["sglang"] = sglang.__version__
    except Exception as e: out["sglang_err"] = str(e)[:100]
    return out


@app.function(**FN)
def probe_v29(): return _probe()


@app.function(**{**FN, "image": img_v30})
def probe_v30(): return _probe()


@app.function(**{**FN, "image": img_sgl})
def probe_sgl(): return _probe()


# ------------------------------------------------------------------------------------------------------------ fixed benchmark set
@app.function(**{**FN, "cpu": 8, "memory": 32 * 1024})
def build_set(judge_rows: list, n_plain: int = 20000, n_g2a: int = 4000, n_g2pos: int = 4000):
    import random, pyarrow as pa, pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download
    from nla.datagen.scale_templates import VARIANTS, MAX_TOKENS, assign
    from nla.datagen import g2_spec as G
    vol_glp.reload(); _prep_hf([LABELLER])
    tok = AutoTokenizer.from_pretrained(LABELLER)
    chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ds = None
    try:
        p = hf_hub_download("deepseek-ai/DeepSeek-V4-Flash-0731", "encoding/encoding_dsv4.py", token=os.environ.get("HF_TOKEN"), cache_dir=BENCH_HF)
        sys.path.insert(0, os.path.dirname(p)); import encoding_dsv4 as E                                   # noqa
        ds = lambda u: E.encode_messages([{"role": "user", "content": u}], thinking_mode="chat")
    except Exception as e: print("[build] no DeepSeek encoding:", str(e)[:200], flush=True)
    rows = {"plain": [], "g2a": [], "g2b": [], "judge": []}
    add = lambda k, u, mt, T, P, meta: rows[k].append({"user": u, "prompt_text": chat(u), "prompt_ds": ds(u) if ds else None, "max_tokens": mt, "temperature": T, "top_p": P, "meta": json.dumps(meta, ensure_ascii=False)})
    # plain: production `label` on a g1 positions shard, one variant per document (LABEL_CFG C)
    d = pq.read_table("/vol_glp/scale/g1/pos/pos_0000.parquet").slice(0, n_plain).to_pydict()
    for doc, n, text in zip(d["doc_id"], d["n_raw_tokens"], d["text"]):
        v = assign(doc); add("plain", VARIANTS[v].format(text=text[-24000:]), MAX_TOKENS[v], 1.0, 0.95, {"variant": v, "doc_id": doc, "n_raw_tokens": int(n)})
    # g2 stage A: fact sheets on sampled windows (another shard so it does not overlap plain)
    d = pq.read_table("/vol_glp/scale/g1/pos/pos_0001.parquet").slice(0, n_g2a).to_pydict()
    for doc, n, text in zip(d["doc_id"], d["n_raw_tokens"], d["text"]):
        key = f"{doc}|{n}"; w = G._pick(G.WINDOWS, random.Random(f"w|{key}")); ctx = G.window(text, w)
        add("g2a", G.FACT_PROMPT.format(text=ctx), 900, 0.3, 0.95, {"window": w, "ctx": ctx, "doc_id": doc})
    # g2 stage B: renderings from the g2 pilot's validated fact sheets, exactly _g2_rows' stage-B construction
    P = pq.read_table("/vol_glp/scale/g2pilot/g2_pilot_v3.parquet", columns=["doc_id", "n_raw_tokens", "text", "window", "facts"]).to_pydict()
    lib = json.load(open(f"{REPO_REMOTE}/nla/datagen/g2_library.json"))
    idx = [i for i, f in enumerate(P["facts"]) if f][:n_g2pos]
    facts = [json.loads(P["facts"][i]) for i in idx]; docs = [P["doc_id"][i] for i in idx]
    pool = {}
    for dd, v in zip(docs, facts):
        for x in G.fact_list(v): pool.setdefault(x.get("etype") or x["type"], []).append((dd, x["value"]))
    for i, v, dd in zip(idx, facts, docs):
        key = f"{dd}|{P['n_raw_tokens'][i]}"; rng = random.Random(f"s|{key}"); fl = G.fact_list(v); tw = []
        ctx = G.window(P["text"][i], P["window"][i])
        for x in fl:
            cands = pool.get(x.get("etype") or x["type"], []); tv = None
            for _ in range(8):
                if not cands: break
                d2, vv = rng.choice(cands)
                if d2 != dd and vv != x["value"]: tv = vv; break
            tw.append(tv)
        for j in range(4):
            st_ = G.sample_style(rng); pr, plan = G.build_render(v, fl, tw, st_, lib, rng)
            add("g2b", pr, G.RENDER_MAX_TOKENS[st_["length"]], 0.9, 0.95, {"fl": fl, "tw": tw, "plan": plan, "ctx": ctx, "style": st_["length"]})
    # judge: the pilot V0 rows with Opus gold + Sonnet judgement (same prompt as production V0)
    Pj = pq.read_table("/vol_glp/scale/pilot/gemma_pilot.parquet").to_pydict(); keep = set(judge_rows)
    for i in range(len(Pj["row"])):
        if Pj["variant"][i] == "v0_opus" and Pj["row"][i] in keep:
            add("judge", VARIANTS["v0_opus"].format(text=Pj["text"][i][-24000:]), MAX_TOKENS["v0_opus"], 1.0, 0.95, {"row": int(Pj["row"][i]), "text": Pj["text"][i], "opus": Pj["opus"][i]})
    os.makedirs(SET, exist_ok=True); man = {}
    for k, r in rows.items():
        pq.write_table(pa.Table.from_pylist(r), f"{SET}/{k}.parquet", compression="zstd")
        man[k] = {"n": len(r), "mean_prompt_tokens": float(sum(len(tok(x["prompt_text"]).input_ids) for x in r[:500]) / max(1, min(500, len(r)))), "sum_max_tokens": sum(x["max_tokens"] for x in r)}
    man["example_prompt_head"] = rows["plain"][0]["prompt_text"][:300]; man["example_ds_head"] = (rows["plain"][0]["prompt_ds"] or "")[:300]
    json.dump(man, open(f"{SET}/manifest.json", "w"), indent=1); vol_glp.commit()
    print("[build]", json.dumps(man, indent=1), flush=True); return man


@app.local_entrypoint()
def main(task: str = "probe", engine: str = "v29", name: str = "", cfg: str = "", n_plain: int = 20000, n_g2a: int = 4000, n_g2pos: int = 4000):
    if task == "build":
        d = json.load(open("/home/celeste/shared/reports/nla-flow-prior/data/scale/pilot_judge.json"))
        rows = sorted(int(k.split("|")[1]) for k in d["rows"] if k.startswith("opus|"))
        print(json.dumps(build_set.remote(rows, n_plain, n_g2a, n_g2pos), indent=1))
    elif task == "probe":
        print(json.dumps({"v29": probe_v29, "v30": probe_v30, "sgl": probe_sgl}[engine].remote(), indent=1))
    elif task == "run":
        c = json.load(open(cfg[1:])) if cfg.startswith("@") else json.loads(cfg)
        r = RUNNERS[engine].remote(name or c.get("name", "run"), c)
        if c.get("variants"): print("RESULT", json.dumps({n: {k: v for k, v in x.items() if k in ("rc", "load_seconds", "sets", "error")} for n, x in r.items()}, indent=1))
        else: print("RESULT", json.dumps({k: v for k, v in r.items() if k in ("name", "rc", "load_seconds", "sets", "profile")}, indent=1))
    elif task == "prefetch":                     # --name "a/b,c/d"
        calls = [prefetch.spawn(m) for m in name.split(",")]
        for c_ in calls:
            try: print(c_.get())
            except Exception as e: print("prefetch failed:", str(e)[:300])
