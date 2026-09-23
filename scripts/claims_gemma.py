"""Semantic claims with Gemma 4 (google/gemma-4-26B-A4B-it, thinking off) served by vLLM inside a Modal container — family 3 of the
synthetic claim data, replacing the Sonnet Batch path. One claim per training activation: only anchors whose drawn family is 'semantic'
(nla.flow.claims.draw_family) and every val anchor get a request: ONE aspect from the wide taxonomy (claims_semantic.ASPECTS_WIDE, source
weighted, number aspects whenever the text has digits), 1-2 claims (val: 3), each with a verbatim quote (string-checked against the text;
failures dropped) and a false twin.

  bench: the same prompts under several GPU layouts of one 4 x B200 container (vllm serve: DP=4 | TP=4 | TP=2 x DP=2 | DP=4 + expert parallel,
         the last also at max_num_seqs 1024), measured after warm-up -> {out}/bench.json (+ the parsed claims of the first layout for the gate)
  gen:   the winning layout; text shards -> {out}/semantic_<name>.parquet (anchor_id, claims, types, quotes, where, twins) + stats_<name>.json
"""
import asyncio, glob, gzip, json, os, random, re, signal, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from claims_semantic import SYSTEM_WIDE, sample_request_wide, user_msg, parse_text, _norm, shown_prefix   # noqa: E402

MODEL = "google/gemma-4-26B-A4B-it"
LAYOUTS = {"a_dp4": ["--data-parallel-size", "4"], "b_tp4": ["--tensor-parallel-size", "4"], "c_tp2dp2": ["--tensor-parallel-size", "2", "--data-parallel-size", "2"],
           "d_dp4ep": ["--data-parallel-size", "4", "--enable-expert-parallel"], "d_dp4ep_mns1024": ["--data-parallel-size", "4", "--enable-expert-parallel"]}
LAYOUT_MNS = {"d_dp4ep_mns1024": 1024}


def _dp(flags):
    return int(flags[flags.index("--data-parallel-size") + 1]) if "--data-parallel-size" in flags else 1


def prompts_for(files, rng, limit=0, skip_ids=None):
    from nla.flow.claims import draw_family
    out = []
    for f in files:
        name = os.path.basename(f)[5:-9]
        for l in gzip.open(f, "rt"):
            r = json.loads(l)
            if not (r.get("is_val") or draw_family(r["anchor_id"]) == "semantic"): continue
            if skip_ids and r["anchor_id"] in skip_ids: continue
            q = sample_request_wide(r, rng); r["_shard"] = name
            out.append((r, q, [{"role": "system", "content": SYSTEM_WIDE}, {"role": "user", "content": user_msg(r, q)}]))
            if limit and len(out) >= limit: return out
    return out


def start_server(flags, mns, port, logf, max_model_len=4096):
    cmd = ["vllm", "serve", MODEL, "--port", str(port), "--dtype", "bfloat16", "--max-model-len", str(max_model_len), "--gpu-memory-utilization", "0.90",
           "--max-num-seqs", str(mns), "--enable-prefix-caching", "--limit-mm-per-prompt", '{"image": 0, "audio": 0}', "--seed", "0", "--disable-uvicorn-access-log"] + flags
    t0 = time.time(); proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    import urllib.request
    while time.time() - t0 < 2400:
        if proc.poll() is not None: return proc, None
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5).status == 200: return proc, time.time() - t0
        except Exception: pass
        time.sleep(5)
    return proc, None


def stop_server(proc):
    try: os.killpg(proc.pid, signal.SIGTERM); proc.wait(timeout=120)
    except Exception:
        try: os.killpg(proc.pid, signal.SIGKILL)
        except Exception: pass
    time.sleep(10)


async def _run(items, port, conc, max_tokens=220, temperature=0.8):
    import aiohttp
    sem = asyncio.Semaphore(conc); res = [None] * len(items); tok = {"prompt": 0, "completion": 0, "fail": 0}
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=conc), timeout=aiohttp.ClientTimeout(total=1800)) as ses:
        async def one(i, msgs):
            body = {"model": MODEL, "messages": msgs, "max_tokens": max_tokens, "temperature": temperature, "top_p": 0.95,
                    "chat_template_kwargs": {"enable_thinking": False}}
            async with sem:
                for att in range(3):
                    try:
                        async with ses.post(url, json=body) as rsp:
                            j = await rsp.json()
                        res[i] = j["choices"][0]["message"]["content"] or ""; u = j.get("usage") or {}
                        tok["prompt"] += u.get("prompt_tokens", 0); tok["completion"] += u.get("completion_tokens", 0); return
                    except Exception: await asyncio.sleep(2 + 3 * att)
                tok["fail"] += 1
        await asyncio.gather(*(one(i, m) for i, (_, _, m) in enumerate(items)))
    return res, tok


def run_items(items, port, conc, **kw):
    return asyncio.run(_run(items, port, conc, **kw))


def verify(items, texts):
    """quote string-match -> per anchor records + counters"""
    recs, st = [], {"requests": len(items), "responses": sum(1 for t in texts if t), "claims_raw": 0, "quote_ok": 0, "twins": 0, "by_aspect": {}}
    for (r, q, _), txt in zip(items, texts):
        if not txt: continue
        P, C = _norm(shown_prefix(r, 1500)), _norm(r["cont_text"]); cl, ty, qs, wh, tw = [], [], [], [], []
        for c, qu, _p, f_ in parse_text(txt):
            st["claims_raw"] += 1; qn = _norm(qu)
            where = "prefix" if qn and qn in P else "continuation" if qn and qn in C else None
            if where is None: continue
            st["quote_ok"] += 1; st["twins"] += bool(f_); tag = f"{q['aspects'][0]}/{q['gran']}/{q['style']}"
            cl.append(c); ty.append(tag); qs.append(qu); wh.append(where); tw.append(f_ or None)
            st["by_aspect"][q["aspects"][0]] = st["by_aspect"].get(q["aspects"][0], 0) + 1
        if cl: recs.append({"anchor_id": r["anchor_id"], "_shard": r["_shard"], "claims": cl, "types": ty, "quotes": qs, "where": wh, "twins": tw})
    st["quote_pass_rate"] = st["quote_ok"] / max(st["claims_raw"], 1)
    return recs, st


def bench(root, out, n=20000, warm=1000, layouts=None, text_glob="text_v1_001.jsonl.gz"):
    import pyarrow as pa, pyarrow.parquet as pq
    os.makedirs(out, exist_ok=True); rng = random.Random(0)
    items = prompts_for(sorted(glob.glob(f"{root}/text/{text_glob}")), rng, limit=n + warm)
    print(f"[gemma-bench] {len(items)} prompts ({warm} warm-up)", flush=True); res = {}
    for k in (layouts or list(LAYOUTS)):
        mns = LAYOUT_MNS.get(k, 512); logf = open(f"{out}/server_{k}.log", "w")
        proc, t_start = start_server(LAYOUTS[k], mns, 8000, logf)
        if t_start is None:
            tail = open(f"{out}/server_{k}.log").read()[-3000:]; res[k] = {"error": "server did not start", "log_tail": tail}
            print(f"[gemma-bench] {k}: FAILED to start\n{tail[-1500:]}", flush=True); stop_server(proc); json.dump(res, open(f"{out}/bench.json", "w"), indent=1); continue
        conc = 2 * mns * _dp(LAYOUTS[k])                                              # 2 x the engines' running slots in flight
        run_items(items[:warm], 8000, conc)
        t0 = time.time(); texts, tok = run_items(items[warm:], 8000, conc); dt = time.time() - t0
        recs, st = verify(items[warm:], texts)
        res[k] = {"engine_start_s": t_start, "max_num_seqs": mns, "seconds": dt, "requests_per_s": len(texts) / dt, "claims_per_s": st["quote_ok"] / dt,
                  "raw_claims_per_s": st["claims_raw"] / dt, "prompt_tokens_per_s": tok["prompt"] / dt, "generated_tokens_per_s": tok["completion"] / dt,
                  "mean_prompt_tokens": tok["prompt"] / max(len(texts), 1), "mean_generated_tokens": tok["completion"] / max(len(texts), 1), "failed_requests": tok["fail"],
                  "quote_pass_rate": st["quote_pass_rate"], "claims_per_request": st["quote_ok"] / max(len(texts), 1)}
        print(f"[gemma-bench] {k}: start {t_start:.0f}s | {res[k]['requests_per_s']:.0f} req/s, {res[k]['claims_per_s']:.0f} verified claims/s, "
              f"{res[k]['prompt_tokens_per_s']:.0f} prompt tok/s, {res[k]['generated_tokens_per_s']:.0f} gen tok/s, quote pass {st['quote_pass_rate']:.3f}", flush=True)
        if not os.path.exists(f"{out}/bench_claims.parquet"):
            pq.write_table(pa.Table.from_pylist([{k2: v for k2, v in r.items() if k2 != "_shard"} | {"shard": r["_shard"]} for r in recs]), f"{out}/bench_claims.parquet", compression="zstd")
            with gzip.open(f"{out}/bench_raw.jsonl.gz", "wt") as f:
                for (r, q, _), t in zip(items[warm:], texts): f.write(json.dumps({"anchor_id": r["anchor_id"], "spec": q, "text": t}) + "\n")
        stop_server(proc); json.dump(res, open(f"{out}/bench.json", "w"), indent=1)
    return res


def gen(root, out, names, layout="a_dp4", commit=None, conc=None, max_model_len=4096):
    """one server for the whole call; shards processed in order, each written + committed as soon as it is done (skip-done)"""
    import pyarrow as pa, pyarrow.parquet as pq
    os.makedirs(out, exist_ok=True); mns = LAYOUT_MNS.get(layout, 512)
    todo = [n for n in names if not os.path.exists(f"{out}/semantic_{n}.parquet")]
    if not todo: return {}
    flags = LAYOUTS[layout] if layout in LAYOUTS else []
    if layout == "single": flags = []
    logf = open(f"{out}/server_{todo[0]}.log", "w"); proc, t_start = start_server(flags, mns, 8000, logf, max_model_len)
    if t_start is None: raise SystemExit("gemma server did not start: " + open(f"{out}/server_{todo[0]}.log").read()[-2000:])
    print(f"[gemma-gen] server up in {t_start:.0f}s ({layout}, max_num_seqs {mns}); {len(todo)} shards", flush=True); allst = {}
    try:
        for n in todo:
            import zlib
            rng = random.Random(zlib.crc32(n.encode())); items = prompts_for([f"{root}/text/text_{n}.jsonl.gz"], rng)
            t0 = time.time(); texts, tok = run_items(items, 8000, conc or (2 * mns * _dp(flags))); dt = time.time() - t0
            recs, st = verify(items, texts)
            st.update(seconds=dt, requests_per_s=len(items) / max(dt, 1e-6), claims_per_s=st["quote_ok"] / max(dt, 1e-6), prompt_tokens=tok["prompt"], generated_tokens=tok["completion"],
                      failed_requests=tok["fail"], claims_kept=sum(len(r["claims"]) for r in recs), anchors_with_claims=len(recs), layout=layout, model=MODEL)
            tbl = pa.Table.from_pylist([{k: v for k, v in r.items() if k != "_shard"} for r in recs]) if recs else None
            if tbl is not None:
                tmp = f"{out}/semantic_{n}.parquet.tmp"; pq.write_table(tbl, tmp, compression="zstd"); os.replace(tmp, f"{out}/semantic_{n}.parquet")
            json.dump(st, open(f"{out}/stats_{n}.json", "w"), indent=1); allst[n] = st
            print(f"[gemma-gen] {n}: {len(items)} requests in {dt:.0f}s ({st['requests_per_s']:.0f}/s), {st['claims_kept']} claims ({st['claims_per_s']:.0f}/s), quote pass {st['quote_pass_rate']:.3f}", flush=True)
            if commit: commit()
    finally:
        stop_server(proc)
    return allst


def gen_stream(root, out, k, K, pattern="text_v*_*.jsonl.gz", layout="a_dp4", commit=None, reload=None, idle_exit_min=90, exclude=()):
    """long-running generator: container k of K owns the text shards with crc32(name) % K == k; it keeps one server up, processes every owned
    shard without a semantic_<name>.parquet, re-scans the volume every 5 min for new shards, and exits after idle_exit_min without new work
    or when {out}/STOP exists."""
    import zlib
    os.makedirs(out, exist_ok=True); mns = LAYOUT_MNS.get(layout, 512); flags = [] if layout == "single" else LAYOUTS[layout]
    logf = open(f"{out}/server_stream_{k}.log", "w"); proc, t_start = start_server(flags, mns, 8000, logf)
    if t_start is None: raise SystemExit("gemma server did not start: " + open(f"{out}/server_stream_{k}.log").read()[-2000:])
    print(f"[gemma-stream {k}/{K}] server up in {t_start:.0f}s ({layout})", flush=True); idle_since = time.time(); n_done = 0
    try:
        while not os.path.exists(f"{out}/STOP"):
            if reload: reload()
            names = sorted(os.path.basename(f)[5:-9] for f in glob.glob(f"{root}/text/{pattern}"))
            todo = [n for n in names if zlib.crc32(n.encode()) % K == k and n not in exclude and not os.path.exists(f"{out}/semantic_{n}.parquet")
                    and not os.path.exists(f"{out}/stats_{n}.json")]
            if not todo:
                if (time.time() - idle_since) / 60 > idle_exit_min: break
                time.sleep(300); continue
            for n in todo:
                gen_one(root, out, n, mns, flags, layout)
                n_done += 1; idle_since = time.time()
                if commit: commit()
    finally:
        stop_server(proc)
    print(f"[gemma-stream {k}/{K}] exit after {n_done} shards", flush=True)
    return n_done


def gen_one(root, out, n, mns, flags, layout):
    import zlib, pyarrow as pa, pyarrow.parquet as pq
    rng = random.Random(zlib.crc32(n.encode())); items = prompts_for([f"{root}/text/text_{n}.jsonl.gz"], rng)
    t0 = time.time(); texts, tok = run_items(items, 8000, 2 * mns * _dp(flags)); dt = time.time() - t0
    recs, st = verify(items, texts)
    st.update(seconds=dt, requests_per_s=len(items) / max(dt, 1e-6), claims_per_s=st["quote_ok"] / max(dt, 1e-6), prompt_tokens=tok["prompt"], generated_tokens=tok["completion"],
              failed_requests=tok["fail"], claims_kept=sum(len(r["claims"]) for r in recs), anchors_with_claims=len(recs), layout=layout, model=MODEL)
    if recs:
        tmp = f"{out}/semantic_{n}.parquet.tmp"; pq.write_table(pa.Table.from_pylist([{k2: v for k2, v in r.items() if k2 != "_shard"} for r in recs]), tmp, compression="zstd")
        os.replace(tmp, f"{out}/semantic_{n}.parquet")
    json.dump(st, open(f"{out}/stats_{n}.json", "w"), indent=1)
    print(f"[gemma-gen] {n}: {len(items)} requests in {dt:.0f}s ({st['requests_per_s']:.0f}/s), {st['claims_kept']} claims ({st['claims_per_s']:.0f}/s), quote pass {st['quote_pass_rate']:.3f}", flush=True)
    return st
