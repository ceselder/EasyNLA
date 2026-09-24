"""Engine benchmark worker (runs INSIDE the Modal container as a subprocess, so env vars / crashes are isolated).

    python scripts/gemma_bench/worker.py --set-dir /vol_glp/scale/bench/set_v1 --out /vol_glp/scale/bench/results/<name>.json --cfg '<json>'

cfg (JSON):
  mode          "offline" (in-process vllm.LLM, exactly what production uses) | "server" (any OpenAI-compatible server: `cmd` is launched,
                /health polled, then the sets are streamed through /v1/completions with `concurrency` in-flight requests)
  engine_kwargs offline: kwargs for vllm.LLM
  cmd, port     server: argv list + port (health = GET /health, completions = POST /v1/completions)
  model         served model name (server mode)
  sets          subset of ["plain", "g2a", "g2b", "judge"] (default all); "judge" outputs are written as parquet next to the json
  warmup        number of plain prompts used to warm up (compile / graph capture / autotune out of the timing), default 300
  profile       optional {"n": 1024, "max_tokens": 48}: torch-profiler window over one decode-heavy batch (offline mode); kernel table in the json
  prompt_col    "prompt_text" (chat template pre-applied, default) | "prompt_ds" (DeepSeek-V4 encoding) | any column of the set parquet
  skip_tokens   list of extra stop token ids (server mode: passed as stop_token_ids)
Metrics per set: seconds, prompts/s, prefill tok/s (all prompt tokens, as production reports), uncached prefill tok/s (prompt tokens minus
cached), decode tok/s, output-length distribution, frac capped at max_tokens, parse-fail rate with the production parsers.
"""
import argparse, json, os, sys, time, gzip, subprocess, asyncio, math, re, collections
import numpy as np

REPO = os.environ.get("REPO_REMOTE", "/root/easyNLA")
if REPO not in sys.path: sys.path.insert(0, REPO)


# ------------------------------------------------------------------------------------------------------------ production parsers
def parse_ok(kind, meta, text):
    from nla.datagen.scale_templates import clean
    from nla.datagen import g2_spec as G
    if kind in ("plain", "judge"): return clean(text) is not None
    if kind == "g2a":
        f = G.parse_facts(text)
        if f is None: return False
        try: G.validate(f, meta["ctx"]); return True
        except Exception: return False
    if kind == "g2b":
        t = G.parse_render(text)
        if not t: return False
        q = G.qc_render(t, meta["fl"], meta["plan"], meta["ctx"], meta["tw"])
        return not (q["exact_missing"] or q["leaked"] or q["unsupported_numbers"])
    return True


def summarize(kind, rows, dt):
    """rows: list of dict(prompt_tokens, cached_tokens, output_tokens, finished_by_length, ok)"""
    n = len(rows); out = np.array([r["output_tokens"] for r in rows], dtype=np.float64)
    pt = sum(r["prompt_tokens"] for r in rows); ct = sum(r.get("cached_tokens", 0) or 0 for r in rows)
    return {"kind": kind, "n": n, "seconds": dt, "prompts_per_s": n / dt, "prefill_tok_per_s": pt / dt, "uncached_prefill_tok_per_s": (pt - ct) / dt,
            "cached_frac": ct / max(1, pt), "decode_tok_per_s": float(out.sum()) / dt, "mean_prompt_tokens": pt / n,
            "out_mean": float(out.mean()), "out_p50": float(np.percentile(out, 50)), "out_p90": float(np.percentile(out, 90)),
            "out_p99": float(np.percentile(out, 99)), "out_max": float(out.max()), "frac_capped": float(np.mean([r["finished_by_length"] for r in rows])),
            "parse_fail": float(np.mean([not r["ok"] for r in rows])), "ok_per_s": sum(r["ok"] for r in rows) / dt}


def load_sets(set_dir, names):
    import pyarrow.parquet as pq
    S = {}
    for k in names:
        t = pq.read_table(f"{set_dir}/{k}.parquet").to_pydict()
        S[k] = [{"prompt_text": t["prompt_text"][i], "prompt_ds": t.get("prompt_ds", [None] * len(t["user"]))[i], "user": t["user"][i],
                 "max_tokens": t["max_tokens"][i], "temperature": t["temperature"][i], "top_p": t["top_p"][i], "meta": json.loads(t["meta"][i])}
                for i in range(len(t["user"]))]
    return S


# ------------------------------------------------------------------------------------------------------------ offline (vllm.LLM)
def run_offline(cfg, S, out_path):
    from vllm import LLM, SamplingParams
    t0 = time.time(); llm = LLM(**cfg["engine_kwargs"]); t_load = time.time() - t0
    print(f"[worker] engine up in {t_load:.0f}s", flush=True)
    col = cfg.get("prompt_col", "prompt_text")
    sp = lambda r, mt=None: SamplingParams(temperature=r["temperature"], top_p=r["top_p"], max_tokens=mt or r["max_tokens"], seed=None)

    def gen(rows, mt=None):
        return llm.generate([r[col] for r in rows], [sp(r, mt) for r in rows], use_tqdm=False)

    def collect(kind, rows, res):
        recs = []
        for r, o in zip(rows, res):
            oo = o.outputs[0]; txt = oo.text
            recs.append({"prompt_tokens": len(o.prompt_token_ids), "cached_tokens": getattr(o, "num_cached_tokens", 0) or 0, "output_tokens": len(oo.token_ids),
                         "finished_by_length": oo.finish_reason == "length", "ok": parse_ok(kind, r["meta"], txt), "text": txt})
        return recs

    warm = S.get("plain") or next(iter(S.values()))
    nw = min(cfg.get("warmup", 300), len(warm))
    t0 = time.time(); gen(warm[:nw]); t_warm = time.time() - t0
    print(f"[worker] warm-up {nw} prompts in {t_warm:.0f}s", flush=True)
    result = {"mode": "offline", "engine_kwargs": cfg["engine_kwargs"], "env": cfg.get("env", {}), "load_seconds": t_load, "warmup_seconds": t_warm, "sets": {}}
    for kind in cfg.get("sets", ["plain", "g2a", "g2b", "judge"]):
        rows = S[kind]
        t0 = time.time(); res = gen(rows); dt = time.time() - t0
        recs = collect(kind, rows, res)
        result["sets"][kind] = summarize(kind, recs, dt)
        print(f"[worker] {kind}: {json.dumps(result['sets'][kind])}", flush=True)
        if kind == "judge":
            import pyarrow as pa, pyarrow.parquet as pq
            pq.write_table(pa.Table.from_pylist([{"row": r["meta"]["row"], "text": r["meta"]["text"], "opus": r["meta"].get("opus"), "raw": x["text"],
                                                  "n_out": x["output_tokens"], "ok": x["ok"]} for r, x in zip(rows, recs)]),
                           out_path.replace(".json", "_judge.parquet"), compression="zstd")
        if kind == "g2a":                                     # sample of raw outputs for eyeballing the JSON quality
            result["g2a_samples"] = [x["text"][:600] for x in recs[:3]]
        json.dump(result, open(out_path, "w"), indent=1)
    if cfg.get("profile"):
        result["profile"] = profile_offline(llm, S, cfg["profile"], col, sp)
        json.dump(result, open(out_path, "w"), indent=1)
    return result


def profile_offline(llm, S, pcfg, col, sp):
    """torch profiler over one batch of n prompts x max_tokens tokens (decode-heavy steady state at the real batch size) -> kernel table"""
    import glob
    pdir = pcfg.get("dir") or os.environ.get("VLLM_TORCH_PROFILER_DIR") or "/tmp/prof"
    os.makedirs(pdir, exist_ok=True)
    rows = (S.get("g2a") or S["plain"])[: pcfg.get("n", 1024)]
    llm.generate([r[col] for r in rows], [sp(r, 4) for r in rows], use_tqdm=False)      # prefill everything once so the prefix cache is warm
    llm.start_profile()
    t0 = time.time(); res = llm.generate([r[col] for r in rows], [sp(r, pcfg.get("max_tokens", 48)) for r in rows], use_tqdm=False); dt = time.time() - t0
    llm.stop_profile(); time.sleep(5)
    ntok = sum(len(o.outputs[0].token_ids) for o in res)
    files = sorted(glob.glob(f"{pdir}/**/*.json*", recursive=True), key=os.path.getmtime)
    if not files: return {"error": "no trace", "seconds": dt, "tokens": ntok}
    agg = collections.Counter(); cnt = collections.Counter(); total = 0.0
    for f in files[-2:]:                                                                    # engine-core trace (+ frontend trace)
        op = gzip.open if f.endswith(".gz") else open
        try: ev = json.load(op(f, "rt")).get("traceEvents", [])
        except Exception as e: return {"error": f"trace parse: {e}", "file": f}
        for e in ev:
            if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"): continue
            nm = re.sub(r"<.*", "", e.get("name", ""))[:110]; d = float(e.get("dur", 0)); agg[nm] += d; cnt[nm] += 1; total += d
    top = [{"kernel": k, "ms": v / 1e3, "frac": v / total, "calls": cnt[k]} for k, v in agg.most_common(40)]
    for f in files: os.remove(f)
    return {"seconds": dt, "tokens": ntok, "tok_per_s": ntok / dt, "gpu_busy_ms": total / 1e3, "gpu_busy_frac": total / 1e6 / dt, "top": top}


# ------------------------------------------------------------------------------------------------------------ server (OpenAI-compatible)
async def _client_run(cfg, kind, rows, col):
    import aiohttp
    port = cfg["port"]; url = f"http://127.0.0.1:{port}/v1/completions"; conc = cfg.get("concurrency", 2048)
    sem = asyncio.Semaphore(conc); recs = [None] * len(rows)

    async def one(i, r, sess):
        body = {"model": cfg["model"], "prompt": r[col], "max_tokens": r["max_tokens"], "temperature": r["temperature"], "top_p": r["top_p"]}
        if cfg.get("stop_token_ids"): body["stop_token_ids"] = cfg["stop_token_ids"]
        if cfg.get("extra_body"): body.update(cfg["extra_body"])
        async with sem:
            for attempt in range(3):
                try:
                    async with sess.post(url, json=body, timeout=aiohttp.ClientTimeout(total=3600)) as resp:
                        d = await resp.json()
                    if "choices" not in d: raise RuntimeError(str(d)[:200])
                    ch = d["choices"][0]; u = d.get("usage", {}); det = u.get("prompt_tokens_details") or {}
                    recs[i] = {"prompt_tokens": u.get("prompt_tokens", 0), "cached_tokens": det.get("cached_tokens", 0) or 0, "output_tokens": u.get("completion_tokens", 0),
                               "finished_by_length": ch.get("finish_reason") == "length", "ok": parse_ok(kind, r["meta"], ch.get("text", "")), "text": ch.get("text", "")}
                    return
                except Exception as e:
                    err = str(e)[:200]
            recs[i] = {"prompt_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "finished_by_length": False, "ok": False, "text": "", "error": err}

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=conc + 64)) as sess:
        await asyncio.gather(*[one(i, r, sess) for i, r in enumerate(rows)])
    return recs


def run_server(cfg, S, out_path):
    import urllib.request
    port = cfg["port"]; log = open(cfg.get("server_log", "/tmp/server.log"), "ab")
    t0 = time.time(); proc = subprocess.Popen(cfg["cmd"], stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **cfg.get("env", {})})
    ok = False
    while time.time() - t0 < cfg.get("startup_timeout", 3600):
        if proc.poll() is not None: raise RuntimeError(f"server exited with {proc.returncode}; see log")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}{cfg.get('health_path', '/health')}", timeout=5); ok = True; break
        except Exception: time.sleep(5)
    if not ok: proc.kill(); raise RuntimeError("server did not become healthy")
    t_load = time.time() - t0; print(f"[worker] server up in {t_load:.0f}s", flush=True)
    col = cfg.get("prompt_col", "prompt_text")
    try:
        warm = S.get("plain") or next(iter(S.values())); nw = min(cfg.get("warmup", 300), len(warm))
        t0 = time.time(); asyncio.run(_client_run(cfg, "plain", warm[:nw], col)); t_warm = time.time() - t0
        print(f"[worker] warm-up {nw} prompts in {t_warm:.0f}s", flush=True)
        result = {"mode": "server", "cmd": cfg["cmd"], "env": cfg.get("env", {}), "load_seconds": t_load, "warmup_seconds": t_warm, "sets": {}}
        for kind in cfg.get("sets", ["plain", "g2a", "g2b", "judge"]):
            rows = S[kind]; t0 = time.time(); recs = asyncio.run(_client_run(cfg, kind, rows, col)); dt = time.time() - t0
            result["sets"][kind] = summarize(kind, recs, dt); result["sets"][kind]["errors"] = sum("error" in r for r in recs)
            print(f"[worker] {kind}: {json.dumps(result['sets'][kind])}", flush=True)
            if kind == "judge":
                import pyarrow as pa, pyarrow.parquet as pq
                pq.write_table(pa.Table.from_pylist([{"row": r["meta"]["row"], "text": r["meta"]["text"], "opus": r["meta"].get("opus"), "raw": x["text"],
                                                      "n_out": x["output_tokens"], "ok": x["ok"]} for r, x in zip(rows, recs)]),
                               out_path.replace(".json", "_judge.parquet"), compression="zstd")
            if kind == "g2a": result["g2a_samples"] = [x["text"][:600] for x in recs[:3]]
            json.dump(result, open(out_path, "w"), indent=1)
    finally:
        proc.terminate()
        try: proc.wait(30)
        except Exception: proc.kill()
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--set-dir", required=True); ap.add_argument("--out", required=True); ap.add_argument("--cfg", required=True)
    a = ap.parse_args(); cfg = json.loads(a.cfg)
    sets = cfg.get("sets", ["plain", "g2a", "g2b", "judge"])
    S = load_sets(a.set_dir, list(dict.fromkeys(sets + (["plain"] if "plain" not in sets else []))))
    if cfg.get("limit"):
        S = {k: v[: cfg["limit"]] for k, v in S.items()}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    r = run_offline(cfg, S, a.out) if cfg.get("mode", "offline") == "offline" else run_server(cfg, S, a.out)
    print("[worker] RESULT " + json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("prompts_per_s", "decode_tok_per_s", "prefill_tok_per_s", "parse_fail", "out_mean")}
                                           for k, v in r["sets"].items()}), flush=True)
