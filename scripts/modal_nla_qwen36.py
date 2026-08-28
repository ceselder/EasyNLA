"""Modal app for the Qwen 3.6 27B NLA work (Experiment B: EMA'd AR critic).

Stages are separate functions so each can be run and debugged alone:

  modal run modal/nla_qwen36.py::harness_check        # ~1 min, proves the image
  modal run modal/nla_qwen36.py::smoke_load           # ~10 min, loads the 27B
  modal run --detach modal/nla_qwen36.py::measure_alpha
  ...

`--detach` is REQUIRED for anything long: an ephemeral app dies with the local
client, which is what killed the DSv4 sweep's LO arm.
"""

import os

import modal

APP_NAME = "nla-qwen36-ema"
VOLUME = "nla-qwen36-ema"          # our own volume, per the workspace house rules
BASE_MODEL = "Qwen/Qwen3.6-27B"
LAYER_INDEX = 42                   # ~65% of 64 layers, per the experiments brief
REPO_LOCAL = "/home/chert/nla-ema/EasyNLA"
REPO_REMOTE = "/root/easyNLA"

# transformers 5.x is REQUIRED: qwen3_5 does not exist in 4.57.1 (verified
# against the wheel). That is also why this app never touches vllm/vllm-lens —
# vllm-lens 1.1.0 pins transformers 4.57.1, so the fast RL path and this model
# are mutually exclusive. We use the HF-rollout trainer instead.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.9.0",
        "transformers>=5.16.0",
        "accelerate", "peft", "bitsandbytes",
        "pyarrow", "pandas", "numpy",
        "wandb", "anthropic", "huggingface_hub[hf_transfer]",
        "safetensors", "sentencepiece", "protobuf", "pyyaml",
        # transformers decorates the qwen3_5 linear-attention ops with
        # use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule",
        # "fla"). Without `kernels` it silently falls back to
        # torch_chunk_gated_delta_rule, which materializes chunk x chunk
        # intermediates per layer and OOM'd a 180GB B200.
        "kernels",
    )
    .env({
        "HF_HOME": "/vol/hf_cache",           # keep the 15-shard pull on the volume
        "HF_XET_HIGH_PERFORMANCE": "1",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    .add_local_dir(REPO_LOCAL, REPO_REMOTE, copy=True,
                   ignore=[".git", ".venv", "__pycache__", "*.pyc"])
)

app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)

# Keys come from the local env (see ~/.chert_env), never baked into the image.
def _secret():
    keys = {}
    for k in ("WANDB_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN"):
        v = os.environ.get(k)
        if v:
            keys[k] = v
    return modal.Secret.from_dict(keys)


@app.function(gpu="B200", volumes={"/vol": vol}, timeout=15 * 60,
              secrets=[_secret()])
def harness_check():
    """Cheapest possible proof that image + GPU + volume + imports all work."""
    import subprocess
    import sys

    sys.path.insert(0, REPO_REMOTE)
    import torch

    print("=== gpu ===", flush=True)
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                          "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip())
    print(f"torch {torch.__version__} | cuda {torch.version.cuda} | "
          f"devices {torch.cuda.device_count()}", flush=True)

    import transformers
    print(f"transformers {transformers.__version__}", flush=True)
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    assert "qwen3_5" in CONFIG_MAPPING_NAMES, "qwen3_5 missing — image pin is wrong"
    print("qwen3_5 in CONFIG_MAPPING: True", flush=True)

    # the repo must import, including our new modules
    from nla.critic_ema import CriticEMA, NoEMA
    from nla.utils.arch_adapters import resolve_lora_target_modules
    print("easyNLA imports OK (incl. nla.critic_ema)", flush=True)

    os.makedirs("/vol/scratch", exist_ok=True)
    with open("/vol/scratch/harness_ok", "w") as f:
        f.write("ok\n")
    vol.commit()
    print("volume write OK -> /vol/scratch/harness_ok", flush=True)
    return "harness_check passed"


@app.function(gpu="B200", volumes={"/vol": vol}, timeout=90 * 60,
              secrets=[_secret()])
def smoke_load():
    """Load the real 27B and check every arch assumption the port depends on.

    This is the test my local mock could NOT make: that resolve_decoder_layers
    finds the decoder on the genuine Qwen3.6 module tree, and that the LoRA
    regex matches real module names.
    """
    import re
    import sys
    import time

    sys.path.insert(0, REPO_REMOTE)
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from nla.utils.arch_adapters import (
        resolve_decoder_layers, resolve_lora_target_modules, resolve_text_config,
    )

    t0 = time.time()
    cfg = AutoConfig.from_pretrained(BASE_MODEL)
    tcfg = resolve_text_config(cfg)
    print(f"model_type={getattr(cfg,'model_type',None)} "
          f"text_model_type={getattr(tcfg,'model_type',None)} "
          f"layers={getattr(tcfg,'num_hidden_layers',None)} "
          f"hidden={getattr(tcfg,'hidden_size',None)}", flush=True)

    print(f"loading {BASE_MODEL} (bf16)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    print(f"loaded in {time.time()-t0:.0f}s | "
          f"{torch.cuda.memory_allocated()/1e9:.1f}GB allocated", flush=True)

    # --- the assumption under test ---
    layers = resolve_decoder_layers(model)
    print(f"resolve_decoder_layers -> {type(layers).__name__}, len={len(layers)}", flush=True)
    assert len(layers) == tcfg.num_hidden_layers, "decoder layer count mismatch"
    print(f"layer[{LAYER_INDEX}] class = {type(layers[LAYER_INDEX]).__name__}", flush=True)

    # --- LoRA regex against REAL module names ---
    pat = resolve_lora_target_modules(cfg)
    names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    hit = [n for n in names if re.fullmatch(pat, n)]
    lm_hit = [n for n in hit if "language_model" in n]
    bad = [n for n in hit if ("visual" in n or n.startswith("mtp."))]
    print(f"Linear modules: {len(names)} | regex matches: {len(hit)} "
          f"| in language_model: {len(lm_hit)} | vision/mtp wrongly hit: {len(bad)}", flush=True)
    assert not bad, f"regex leaked outside the language model: {bad[:5]}"
    assert len(hit) > 0, "regex matched nothing"
    for n in hit[:4] + hit[-2:]:
        print("   match:", n, flush=True)

    # --- residual-stream hook at the capture layer ---
    captured = {}

    def _hook(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["shape"] = tuple(h.shape)
        captured["norm_p75"] = float(h[0].float().norm(dim=-1).quantile(0.75))

    hh = layers[LAYER_INDEX].register_forward_hook(_hook)
    ids = tok("The Ostrogothic kingdom in Italy collapsed after the Gothic War.",
              return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        model(**ids)
    hh.remove()
    print(f"hook at layer {LAYER_INDEX}: shape={captured.get('shape')} "
          f"norm_p75={captured.get('norm_p75'):.1f}", flush=True)
    assert captured.get("shape"), "forward hook never fired"

    # --- generation throughput at the real rollout shape ---
    # Decides whether HF rollouts are viable for 16x128 or whether we need a
    # different plan. 8 sequences x 256 new tokens, then extrapolate.
    prompt = ["Explain what the model is representing here."] * 8
    enc = tok(prompt, return_tensors="pt", padding=True).to("cuda:0")
    torch.cuda.synchronize()
    tg = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=256, do_sample=True,
                             temperature=1.0, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    dt = time.time() - tg
    new_tok = int((out.shape[1] - enc["input_ids"].shape[1]) * out.shape[0])
    print(f"generate: 8 seqs x 256 tok in {dt:.1f}s "
          f"({new_tok/dt:.0f} tok/s) -> 2048 rollouts ~= {dt*2048/8/60:.1f} min/step "
          f"on ONE B200", flush=True)
    return {
        "loaded_s": round(time.time() - t0),
        "layers": len(layers),
        "regex_matches": len(hit),
        "hook_shape": captured.get("shape"),
        "norm_p75_sample": captured.get("norm_p75"),
        "gen_tok_per_s": round(new_tok / dt),
        "est_min_per_step_1gpu": round(dt * 2048 / 8 / 60, 1),
    }


@app.function(gpu="B200", volumes={"/vol": vol}, timeout=60 * 60,
              secrets=[_secret()])
def throughput_sweep():
    """Measure generation throughput vs batch size — the number that decides
    whether HF rollouts can carry 16x128 = 2048 rollouts/step.

    The smoke test measured batch 8 and linearly extrapolated to 2048, giving
    ~43 min/step. That is wrong: batch-8 decode on a B200 is latency-bound with
    ~129GB idle. Batched decode amortizes the weight read across the batch, so
    tok/s should climb steeply until compute- or memory-bound.
    """
    import sys
    import time

    sys.path.insert(0, REPO_REMOTE)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="cuda:0", low_cpu_mem_usage=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # required for correct batched generation
    model.eval()
    print(f"model on GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    PROMPT = ("Below is a passage. Describe what a language model would be "
              "representing internally while reading it. Passage: The Ostrogothic "
              "kingdom in Italy collapsed after the Gothic War of 535-554, and "
              "historians debate whether the reconquest did more damage than the "
              "Gothic rule it replaced.")
    NEW = 256
    TARGET = 2048                      # 16 x 128
    results = []
    for bs in (8, 32, 64, 128, 256):
        try:
            enc = tok([PROMPT] * bs, return_tensors="pt", padding=True).to("cuda:0")
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=NEW, do_sample=True,
                                     temperature=1.0, pad_token_id=tok.pad_token_id)
            torch.cuda.synchronize()
            dt = time.time() - t0
            gen = int((out.shape[1] - enc["input_ids"].shape[1]) * bs)
            tps = gen / dt
            peak = torch.cuda.max_memory_allocated() / 1e9
            per_step_min = (TARGET / bs) * dt / 60
            results.append((bs, round(tps), round(peak, 1), round(per_step_min, 1)))
            print(f"  bs={bs:4d}  {dt:6.1f}s  {tps:7.0f} tok/s  peak {peak:5.1f}GB  "
                  f"-> {per_step_min:6.1f} min/step for {TARGET} rollouts", flush=True)
            del out, enc
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"  bs={bs:4d}  OOM", flush=True)
            torch.cuda.empty_cache()
            break
    best = min(results, key=lambda r: r[3]) if results else None
    if best:
        bs, tps, peak, mins = best
        print(f"\nBEST: bs={bs} -> {mins} min/step on ONE B200 "
              f"({mins*200/60:.1f} h for a 200-step arm; "
              f"{mins*200/60/8:.1f} h if sharded over 8 GPUs)", flush=True)
    return {"sweep": results, "best": best}


DATA = "/vol/data"
OPUS5 = "ceselder/easynla-dsv4-warmstart-opus5"


@app.function(volumes={"/vol": vol}, timeout=90 * 60, cpu=8.0, memory=65536,
              secrets=[_secret()])
def prepare_rows():
    """CPU stage: build ONE shared row pool of (text, explanation) for Qwen3.6.

    Source is the Opus-5 warm start, not asher577's: its sidecar records
    api_summaries.model = claude-sonnet-4-6, and the DSv4 result was that Opus-5
    explanations lift held-out AR FVE 36.1 -> 45.3 at matched steps. Same corpus,
    same doc_ids, same detokenized_text_truncated, so the text positions line up.

    Its activation_vector columns are DISCARDED — they are DSv4 layer-28, 4096-d.
    We keep only text + explanation and re-extract on Qwen3.6 (5120-d, layer 42).

    Celeste's design rule (2026-08-27): AV and AR SFT train on the SAME rows.
    The published av/ar splits are disjoint halves, so they are unioned here into
    a shared pool keyed on doc_id.
    """
    import os
    import re

    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    os.makedirs(DATA, exist_ok=True)
    KEEP = ["prompt", "response", "doc_id", "detokenized_text_truncated", "n_raw_tokens"]

    def _explanation_from_av(resp):
        if resp is None:
            return None
        m = re.search(r"<explanation>(.*?)</explanation>", resp, re.S)
        return (m.group(1) if m else resp).strip() or None

    def _explanation_from_ar(prompt):
        if prompt is None:
            return None
        m = re.search(r"<text>(.*?)</text>", prompt, re.S)
        return (m.group(1).strip() or None) if m else None

    pool = {}          # doc_id -> {"text":..., "explanation":...}
    stats = {}
    for split, kind in (("av_sft_shuf", "av"), ("ar_sft_shuf", "ar"),
                        ("av_sft_val", "av"), ("ar_sft_val", "ar")):
        path = hf_hub_download(OPUS5, f"{split}.parquet", repo_type="dataset",
                              cache_dir="/vol/hf_cache")
        pf = pq.ParquetFile(path)
        cols = [c for c in KEEP if c in pf.schema_arrow.names]
        n_new = n_seen = n_noexpl = 0
        for batch in pf.iter_batches(batch_size=20000, columns=cols):
            d = batch.to_pydict()
            for i in range(batch.num_rows):
                n_seen += 1
                did = d["doc_id"][i]
                text = d["detokenized_text_truncated"][i]
                if not text:
                    continue
                expl = (_explanation_from_av(d.get("response", [None])[i])
                        if kind == "av" else
                        _explanation_from_ar(d.get("prompt", [None])[i]))
                if not expl:
                    n_noexpl += 1
                    continue
                # Key on the TEXT, not doc_id: positions_per_doc=10, so ~10
                # rows share a doc_id and keying on it discards 90% of the data
                # (measured: 363,961 av rows collapsed to 36,734 docs).
                # detokenized_text_truncated is the prefix up to the capture
                # position, so it is unique per (doc, position).
                key = text
                if key not in pool:
                    pool[key] = {"doc_id": did, "text": text, "explanation": expl,
                                 "n_raw_tokens": d.get("n_raw_tokens", [0])[i] or 0,
                                 "is_val": split.endswith("_val")}
                    n_new += 1
        stats[split] = (n_seen, n_new, n_noexpl)
        print(f"  {split:14} rows={n_seen:7d} new={n_new:7d} "
              f"no_explanation={n_noexpl:6d} pool={len(pool):7d}", flush=True)

    rows = list(pool.values())
    # doc-level val split is inherited from which file a doc_id first appeared in
    n_val = sum(1 for r in rows if r["is_val"])
    n_docs = len({r["doc_id"] for r in rows})
    tbl = pa.table({
        "doc_id": pa.array([r["doc_id"] for r in rows]),
        "text": pa.array([r["text"] for r in rows]),
        "explanation": pa.array([r["explanation"] for r in rows]),
        "n_raw_tokens": pa.array([r["n_raw_tokens"] for r in rows]),
        "is_val": pa.array([r["is_val"] for r in rows]),
    })
    out = f"{DATA}/shared_pool.parquet"
    pq.write_table(tbl, out, compression="zstd")
    vol.commit()
    print(f"\nwrote {out}: {len(rows)} unique (doc, position) rows across "
          f"{n_docs} docs ({len(rows)-n_val} train / {n_val} val)", flush=True)
    print(f"  size on volume: {os.path.getsize(out)/1e6:.0f} MB", flush=True)
    ex = rows[0]
    print(f"\n  sample doc_id     {ex['doc_id']}")
    print(f"  sample text       {ex['text'][:90]!r}")
    print(f"  sample explanation{ex['explanation'][:90]!r}")
    return {"pool": len(rows), "val": n_val, "per_split": stats}


@app.function(gpu="B200", volumes={"/vol": vol}, timeout=14 * 60 * 60,
              secrets=[_secret()])
def extract_activations(batch_size: int = 512, max_length: int = 1024,
                        limit: int = 0, longest: int = 0):
    """Stage 0: re-extract Qwen3.6 layer-42 residuals for the shared pool.

    Captures the OUTPUT of block LAYER_INDEX at the LAST REAL token —
    detokenized_text_truncated is the prefix ending at the capture position, so
    the final token IS that position. --ar-num-layers must then be
    LAYER_INDEX+1 = 43 (the critic needs block 42 to exist).

    Also computes alpha = p75 of the activation L2 norms over the corpus, which
    the brief asks for. Recorded as a diagnostic; injection_scale is left absent
    (= raw injection) to match the reference configuration.
    """
    import json
    import os
    import sys
    import time

    sys.path.insert(0, REPO_REMOTE)
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nla.utils.arch_adapters import resolve_decoder_layers

    outdir = f"{DATA}/acts_qwen36_L{LAYER_INDEX}"
    # Clear stale output first. shard_id restarts at 0 every run, so a previous
    # (or crashed) run's shards would otherwise survive alongside this run's and
    # silently become training data. Observed: a crashed bs=256 run left
    # shards 0001-0008 (~225k rows) next to a 5120-row slice's shard_0000.
    if os.path.isdir(outdir):
        stale = [f for f in os.listdir(outdir)
                 if f.endswith((".parquet", ".json")) or f == "_COMPLETE"]
        for f in stale:
            os.remove(os.path.join(outdir, f))
        if stale:
            print(f"cleared {len(stale)} stale file(s) from {outdir}", flush=True)
    os.makedirs(outdir, exist_ok=True)

    tbl = pq.read_table(f"{DATA}/shared_pool.parquet")
    rows = tbl.to_pydict()
    n_all = len(rows["text"])
    idx_all = list(range(n_all))
    if limit:
        idx_all = idx_all[:limit]
    if longest:
        # Exercise the conv1d 32-bit-index path directly: the overflow lives in
        # the long-sequence tail, which a head-of-pool slice never reaches.
        idx_all = sorted(idx_all, key=lambda i: -len(rows["text"][i]))[:longest]
    print(f"pool: {n_all} rows | extracting {len(idx_all)}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"          # gather at (len-1) per row

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="cuda:0", low_cpu_mem_usage=True)
    model.eval()
    layers = resolve_decoder_layers(model)
    assert len(layers) == 64, f"expected 64 layers, got {len(layers)}"
    print(f"hooking layers[{LAYER_INDEX}] = {type(layers[LAYER_INDEX]).__name__}", flush=True)

    # sort by token length so batches pad minimally (big throughput win)
    print("tokenizing + length-sorting...", flush=True)
    enc_all = tok([rows["text"][i] for i in idx_all], add_special_tokens=False,
                  truncation=True, max_length=max_length)["input_ids"]
    order = sorted(range(len(idx_all)), key=lambda k: len(enc_all[k]))
    print(f"  token lengths: min={len(enc_all[order[0]])} "
          f"med={len(enc_all[order[len(order)//2]])} "
          f"max={len(enc_all[order[-1]])}", flush=True)

    grab = {}

    class _StopForward(Exception):
        """Abort the forward once the capture layer has produced its output."""

    def _hook(_m, _i, out):
        grab["h"] = out[0] if isinstance(out, tuple) else out
        raise _StopForward          # layers 43..63 are never computed

    handle = layers[LAYER_INDEX].register_forward_hook(_hook)

    SHARD = 25000
    buf_vec, buf_meta, norms = [], [], []
    shard_id, n_done, t0 = 0, 0, time.time()

    def _flush():
        nonlocal shard_id, buf_vec, buf_meta
        if not buf_vec:
            return
        arr = np.stack(buf_vec).astype(np.float32)
        t = pa.table({
            "doc_id": pa.array([m[0] for m in buf_meta]),
            "text": pa.array([m[1] for m in buf_meta]),
            "explanation": pa.array([m[2] for m in buf_meta]),
            "is_val": pa.array([m[3] for m in buf_meta]),
            "n_raw_tokens": pa.array([m[4] for m in buf_meta]),
            "activation_layer": pa.array([LAYER_INDEX] * len(buf_meta)),
            "activation_vector": pa.FixedSizeListArray.from_arrays(
                pa.array(arr.reshape(-1)), arr.shape[1]),
        })
        p = f"{outdir}/shard_{shard_id:04d}.parquet"
        pq.write_table(t, p, compression="zstd")
        print(f"    wrote {p} ({len(buf_meta)} rows, "
              f"{os.path.getsize(p)/1e6:.0f} MB)", flush=True)
        shard_id += 1
        buf_vec, buf_meta = [], []
        vol.commit()

    # TOKEN-BUDGET batching, not fixed batch size. The hybrid linear-attention
    # layers run causal_conv1d with C=10240 channels, and F.conv1d uses 32-bit
    # index math: batch*seq*C must stay under 2^31 = 2.15e9, i.e.
    # batch*seq <= 209,715 tokens. Fixed batch 256 x seq 1024 = 262,144 tokens
    # -> 2.68e9 -> "canUse32BitIndexMath ... got false". Budget 131,072 is the
    # combination already proven on the validation slice (1.34e9).
    # Rows are length-sorted, so the last row of a candidate batch is the
    # longest and sets the padded width.
    token_budget = 131072
    max_batch = max(1, batch_size)
    i_ord, n_batches = 0, 0
    while i_ord < len(order):
        n = 0
        while i_ord + n < len(order) and n < max_batch:
            cand = len(enc_all[order[i_ord + n]])
            if n > 0 and (n + 1) * cand > token_budget:
                break
            n += 1
        chunk = order[i_ord:i_ord + n]
        i_ord += n
        n_batches += 1
        ids_list = [enc_all[k] for k in chunk]
        maxlen = max(len(x) for x in ids_list)
        bx = torch.full((len(chunk), maxlen), tok.pad_token_id, dtype=torch.long)
        am = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        for r, ids in enumerate(ids_list):
            bx[r, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            am[r, :len(ids)] = 1
        bx, am = bx.to("cuda:0"), am.to("cuda:0")
        # model.model(...), NOT model(...): the CausalLM wrapper runs lm_head
        # over the whole sequence, which at vocab 248077 x batch 256 tried to
        # allocate 54.7GB and OOM'd. We only ever need a hidden state.
        # use_cache=False so aborting mid-forward leaves no partial cache state
        # in the hybrid linear-attention layers.
        try:
            with torch.no_grad():
                model.model(input_ids=bx, attention_mask=am, use_cache=False)
        except _StopForward:
            pass
        h = grab["h"]                                   # [B, T, d]
        last = am.sum(dim=1) - 1                        # last REAL token index
        vecs = h[torch.arange(h.shape[0], device=h.device), last].float()
        nrm = vecs.norm(dim=-1)
        norms.append(nrm.cpu().numpy())
        vecs_np = vecs.cpu().numpy()
        for r, k in enumerate(chunk):
            i = idx_all[k]
            buf_vec.append(vecs_np[r])
            buf_meta.append((rows["doc_id"][i], rows["text"][i],
                             rows["explanation"][i], rows["is_val"][i],
                             int(rows["n_raw_tokens"][i]), ))
        n_done += len(chunk)
        if len(buf_vec) >= SHARD:
            _flush()
        if n_batches % 20 == 1:
            el = time.time() - t0
            rate = n_done / max(el, 1e-9)
            print(f"  {n_done}/{len(order)} ({100*n_done/len(order):.1f}%) "
                  f"{rate:.0f} rows/s  eta {(len(order)-n_done)/max(rate,1e-9)/60:.0f} min",
                  flush=True)
    _flush()
    handle.remove()

    alln = np.concatenate(norms)
    alpha = float(np.percentile(alln, 75))
    stats = {
        "rows": int(n_done), "d_model": int(len(buf_vec[0]) if buf_vec else 5120),
        "layer_index": LAYER_INDEX,
        "norm_p25": float(np.percentile(alln, 25)),
        "norm_p50": float(np.percentile(alln, 50)),
        "alpha_norm_p75": alpha,
        "norm_p95": float(np.percentile(alln, 95)),
        "norm_mean": float(alln.mean()), "norm_std": float(alln.std()),
        "shards": shard_id, "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    with open(f"{outdir}/extraction_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    # Written LAST and only on success: downstream stages must refuse to read
    # this directory without it, so a crashed run cannot be consumed as data.
    with open(f"{outdir}/_COMPLETE", "w") as f:
        f.write(f"rows={n_done} shards={shard_id} alpha={alpha:.4f}\n")
    vol.commit()
    print("\n=== extraction stats ===", flush=True)
    for k, v in stats.items():
        print(f"  {k:18} {v}", flush=True)
    print(f"\nALPHA (p75 of layer-{LAYER_INDEX} activation norms) = {alpha:.2f}", flush=True)
    return stats


# Recomputed for the Qwen3.6 tokenizer. ㈎ (U+320E) encodes to TWO tokens here,
# and nla/config.py asserts single-token, so the marker had to change. Derived
# via nla.config.compute_canonical_neighbors, not by hand.
QWEN36_TOKENS = {
    "injection_char": "㈜",                     # ㈜  (was ㈎)
    "injection_token_id": 158983,                   # was 149705
    "injection_left_neighbor_id": 29,               # was 29
    "injection_right_neighbor_id": 510,             # was 522
    "critic_suffix_ids": [1272, 29, 361, 1648, 29],  # was [1318,29,366,1708,29]
}
ALPHA_P75 = 95.708          # measured over all 742,652 rows at layer 42


@app.function(volumes={"/vol": vol}, timeout=4 * 60 * 60, cpu=8.0,
              memory=131072, secrets=[_secret()])
def build_sft_datasets(n_train: int = 500_000, seed: int = 42):
    """Materialize AV and AR SFT parquets from the extracted activations.

    Both stages train on the SAME rows (Celeste, 2026-08-27), and per Celeste
    2026-08-28: 500k rows each, same data.

    Prompt templates are read VERBATIM from the source dataset's sidecar rather
    than retyped — the marker's canonical left/right neighbour token ids depend
    on the exact surrounding characters, so whitespace drift in the template
    would break config.py's neighbour assertions in a way that looks like
    tokenizer drift.
    """
    import glob
    import json
    import os
    import random
    import sys

    sys.path.insert(0, REPO_REMOTE)
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml
    from huggingface_hub import hf_hub_download

    actdir = f"{DATA}/acts_qwen36_L{LAYER_INDEX}"
    if not os.path.exists(f"{actdir}/_COMPLETE"):
        raise SystemExit(f"{actdir} has no _COMPLETE marker — extraction did not "
                         f"finish, refusing to build datasets from a partial run.")
    print(open(f"{actdir}/_COMPLETE").read().strip(), flush=True)

    # verbatim templates from the source sidecar
    side = hf_hub_download(OPUS5, "ar_sft_shuf.parquet.nla_meta.yaml",
                          repo_type="dataset", cache_dir="/vol/hf_cache")
    src = yaml.safe_load(open(side))
    ACTOR = src["prompt_templates"]["actor"]
    CRITIC = src["prompt_templates"]["critic"]
    assert "{injection_char}" in ACTOR and "{explanation}" in CRITIC
    print(f"templates loaded: actor {len(ACTOR)} chars, critic {CRITIC!r}", flush=True)
    # The parquet stores the PLACEHOLDER; train_sft swaps in cfg.injection_char.
    ACTOR_PLACEHOLDER = ACTOR.replace("{injection_char}", "<INJECT>")

    shards = sorted(glob.glob(f"{actdir}/shard_*.parquet"))
    print(f"reading {len(shards)} shards...", flush=True)
    train_idx, val_idx, tables = [], [], []
    off = 0
    for sp in shards:
        t = pq.read_table(sp)
        tables.append(t)
        iv = t.column("is_val").to_pylist()
        for i, v in enumerate(iv):
            (val_idx if v else train_idx).append((len(tables) - 1, i))
        off += t.num_rows
    print(f"  rows={off} train={len(train_idx)} val={len(val_idx)}", flush=True)

    # DOC-LEVEL split, per nla/val_split.py. The inherited per-file is_val flag
    # is a FILE split, so a document's ~10 positions can straddle train and val
    # — exactly the leak val_split.py warns about ("a row-index split leaves
    # ~zero docs fully unseen"). permille=20 matches the published convention
    # (is_val_doc(doc_id, 20)).
    from nla.val_split import is_val_doc
    VAL_PERMILLE = 20
    leaked = 0
    old_val_docs = {tables[ti].column("doc_id")[ri].as_py() for (ti, ri) in val_idx}
    old_train_docs = {tables[ti].column("doc_id")[ri].as_py() for (ti, ri) in train_idx}
    leaked = len(old_val_docs & old_train_docs)
    print(f"  inherited split: {len(old_val_docs)} val docs, "
          f"{leaked} of them ALSO in train ({100*leaked/max(1,len(old_val_docs)):.1f}% leaked) "
          f"-> discarding it", flush=True)

    all_idx = train_idx + val_idx
    train_idx, val_idx = [], []
    for (ti, ri) in all_idx:
        d = tables[ti].column("doc_id")[ri].as_py()
        (val_idx if is_val_doc(d, VAL_PERMILLE) else train_idx).append((ti, ri))
    v_docs = {tables[ti].column("doc_id")[ri].as_py() for (ti, ri) in val_idx}
    t_docs = {tables[ti].column("doc_id")[ri].as_py() for (ti, ri) in train_idx}
    assert not (v_docs & t_docs), "doc-level split still overlaps"
    print(f"  doc-level split (permille={VAL_PERMILLE}): "
          f"train={len(train_idx)} rows/{len(t_docs)} docs  "
          f"val={len(val_idx)} rows/{len(v_docs)} docs  overlap=0", flush=True)

    rng = random.Random(seed)
    rng.shuffle(train_idx)
    if n_train and n_train < len(train_idx):
        train_idx = train_idx[:n_train]
    print(f"  sampled train={len(train_idx)} (seed={seed})", flush=True)

    def _emit(pairs, stage, split):
        prompts, responses, vecs, docs, texts, nraw = [], [], [], [], [], []
        for (ti, ri) in pairs:
            t = tables[ti]
            expl = t.column("explanation")[ri].as_py()
            if stage == "av":
                prompts.append([{"role": "user", "content": ACTOR_PLACEHOLDER}])
                responses.append(f"<explanation>\n{expl}\n</explanation>")
            else:
                prompts.append(CRITIC.replace("{explanation}", expl))
            vecs.append(t.column("activation_vector")[ri].as_py())
            docs.append(t.column("doc_id")[ri].as_py())
            texts.append(t.column("text")[ri].as_py())
            nraw.append(t.column("n_raw_tokens")[ri].as_py())
        n = len(vecs)
        cols = {
            "prompt": pa.array(prompts),
            "activation_vector": pa.array(vecs, type=pa.list_(pa.float32(), 5120)),
            "n_raw_tokens": pa.array(nraw, type=pa.int64()),
            "activation_layer": pa.array([LAYER_INDEX] * n, type=pa.int64()),
            "doc_id": pa.array(docs),
            "detokenized_text_truncated": pa.array(texts),
        }
        if stage == "av":
            cols["response"] = pa.array(responses)
        out = f"{DATA}/sft/{stage}_sft_{split}.parquet"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        pq.write_table(pa.table(cols), out, compression="zstd", row_group_size=5000)

        toks = dict(QWEN36_TOKENS)
        if stage == "av":
            toks["critic_suffix_ids"] = None
        meta = {
            "dataset_id": f"{stage}_sft_Qwen3.6-27B_L{LAYER_INDEX}_opus5expl_{split}",
            "stage": f"{stage}_sft",
            "row_count": n,
            "extraction": {
                "base_model": BASE_MODEL,
                "d_model": 5120,
                "layer_index": LAYER_INDEX,
                "norm": "none",
                "corpus": src["extraction"]["corpus"],
                "positions_per_doc": 10,   # source sidecar says 1, but ~10 rows
                                           # share a doc_id (val_split.py agrees)
                "val_doc_permille": VAL_PERMILLE,
                # measured diagnostic; injection_scale intentionally ABSENT
                # (absent => raw injection, matching the reference config)
                "activation_norm_p75": ALPHA_P75,
            },
            "kind": "nla_dataset",
            "schema_version": 1,
            "keep_debug_metadata": True,
            "tokens": toks,
            "prompt_templates": {"actor": ACTOR, "critic": CRITIC},
            "api_summaries": {"model": "claude-opus-5",
                              "note": "explanations inherited from "
                                      f"{OPUS5}; activations re-extracted on "
                                      f"{BASE_MODEL} layer {LAYER_INDEX}"},
        }
        with open(f"{out}.nla_meta.yaml", "w") as f:
            yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)
        print(f"  wrote {out} ({n} rows, {os.path.getsize(out)/1e9:.2f} GB) + sidecar",
              flush=True)
        return n

    counts = {}
    for stage in ("av", "ar"):
        counts[f"{stage}_train"] = _emit(train_idx, stage, "train")
        counts[f"{stage}_val"] = _emit(val_idx, stage, "val")
    vol.commit()
    print("\n=== built ===", flush=True)
    for k, v in counts.items():
        print(f"  {k:10} {v}", flush=True)
    return counts


@app.function(volumes={"/vol": vol}, timeout=60 * 60, cpu=4.0, memory=32768,
              secrets=[_secret()])
def validate_datasets():
    """Run the built datasets through the repo's OWN loader and assertions.

    load_nla_config re-derives the injection token id and the marker's canonical
    left/right neighbours from the LIVE tokenizer and asserts they match the
    sidecar. verify_critic_suffix checks the tokenized AR prompt ends with the
    recorded suffix. Both are exactly the checks that would otherwise fire at
    trainer startup, several GPU-minutes in.
    """
    import sys

    sys.path.insert(0, REPO_REMOTE)
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    from nla.config import load_nla_config, verify_critic_suffix
    from nla.schema import INJECT_PLACEHOLDER

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    ok = True
    for stage in ("av", "ar"):
        for split in ("train", "val"):
            p = f"{DATA}/sft/{stage}_sft_{split}.parquet"
            try:
                cfg = load_nla_config(p, tok)
            except AssertionError as e:
                print(f"  FAIL {stage}_{split}: {str(e)[:200]}", flush=True)
                ok = False
                continue
            print(f"  OK   {stage}_{split}: d_model={cfg.d_model} "
                  f"inj_id={cfg.injection_token_id} "
                  f"L/R={cfg.injection_left_neighbor_id}/{cfg.injection_right_neighbor_id} "
                  f"mse_scale={cfg.mse_scale:.2f} "
                  f"inj_scale={cfg.injection_scale} "
                  f"layer={cfg.extraction_layer_index}", flush=True)

            t = pq.ParquetFile(p).read_row_group(0).slice(0, 3)
            if stage == "ar":
                for i in range(3):
                    ids = tok.encode(t.column("prompt")[i].as_py(),
                                     add_special_tokens=False)
                    try:
                        verify_critic_suffix(ids, cfg.critic_suffix_ids,
                                             context=f"{stage}_{split} row {i}")
                    except AssertionError as e:
                        print(f"  FAIL suffix {stage}_{split}[{i}]: {str(e)[:180]}",
                              flush=True)
                        ok = False
                print(f"       critic suffix verified on 3 rows", flush=True)
            else:
                c = t.column("prompt")[0].as_py()[0]["content"]
                has_ph = INJECT_PLACEHOLDER in c
                # the marker must resolve to EXACTLY one token in the live prompt
                live = c.replace(INJECT_PLACEHOLDER, cfg.injection_char)
                n_marker = tok.encode(live, add_special_tokens=False).count(
                    cfg.injection_token_id)
                print(f"       placeholder present={has_ph} "
                      f"marker_tokens_in_prompt={n_marker} (want 1)", flush=True)
                if not has_ph or n_marker != 1:
                    ok = False
                r = t.column("response")[0].as_py()
                print(f"       response starts/ends: {r[:24]!r} ... {r[-18:]!r}",
                      flush=True)
    print(f"\n{'ALL CHECKS PASSED' if ok else 'VALIDATION FAILED'}", flush=True)
    if not ok:
        raise SystemExit("dataset validation failed")
    return {"validated": True}


CKPT = "/vol/ckpts"


def _run_sft(mode: str, lr: float, extra: list[str], steps: int = 0,
             bs: int = 32, accum: int = 2, ckpt: bool = True):
    """Shared driver for AV/AR SFT. Runs the repo trainer as a subprocess so its
    argparse, sidecar assertions and wandb logging behave exactly as documented.
    """
    import os
    import subprocess
    import sys

    save_dir = f"{CKPT}/qwen36_{mode}"
    os.makedirs(save_dir, exist_ok=True)
    data = f"{DATA}/sft/{mode}_sft_train.parquet"
    # Held-out parquet is ALWAYS the AV-format one, for both modes: AV uses it
    # for val token-CE/ppl, and AR's load_heldout_explanation_pairs reads the
    # `response` column to recover the explanation text (the AR parquet has no
    # `response` — its explanation is baked into `prompt`). Same rows either way,
    # since av_sft_val and ar_sft_val are built from the identical val_idx.
    val = f"{DATA}/sft/av_sft_val.parquet"
    for p in (data, val):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} — run build_sft_datasets first")

    cmd = [
        sys.executable, "-m", "nla.train_sft",
        "--mode", mode,
        "--base-ckpt", BASE_MODEL,
        "--parquet", data,
        "--sidecar", data,
        "--heldout-parquet", val,      # honest held-out FVE during training
        "--save-dir", save_dir,
        # Effective batch stays 64 (the brief's value) via 32 x 2 accumulation.
        # Gradient checkpointing is forced ON: train_sft defaults it OFF for AR
        # ("smaller model + shorter seq fits comfortably"), which was tuned for
        # an 8B base. At 27B the 43-layer critic is ~34GB and batch 64 x 1024
        # activations OOM'd a 180GB B200 at 176GB allocated.
        "--batch-size", str(bs),
        "--gradient-accumulation-steps", str(accum),
        ("--gradient-checkpointing" if ckpt else "--no-gradient-checkpointing"),
        "--max-len", "1024",
        "--lr", str(lr),
        "--min-lr", "2e-6",
        "--lr-warmup-steps", "50",
        "--max-grad-norm", "1.0",
        # LoRA r64/alpha16 (+ rsLoRA, hardcoded in train_sft) per Celeste's default.
        # bf16 base rather than 4bit: 27B is 53.8GB on a 180GB B200, so there is
        # no need to quantize, and the RL stage loads the same dtype.
        "--use-lora", "--lora-r", "64", "--lora-alpha", "16",
        "--quant", "none",
        "--save-every", "1000",
        "--heldout-every", "250",
        "--seed", "0",
        "--wandb-project", "easynla-qwen36-ema",
        "--wandb-name", f"{mode}_sft_qwen36_L{LAYER_INDEX}",
        # comma-separated, not nargs
        "--wandb-tags", f"qwen3.6-27b,L{LAYER_INDEX},opus5-expl,{mode}_sft",
    ] + extra
    if steps:
        cmd += ["--num-steps", str(steps)]     # else default = exactly one epoch

    print("CMD: " + " ".join(cmd), flush=True)
    # expandable_segments fixes the fragmentation that OOM'd batch 64 with
    # 26.6GB reserved-but-unallocated. Safe here: CLAUDE.md's warning against it
    # applies to vLLM's IPC weight sync, and this path never loads vLLM.
    env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    p = subprocess.run(cmd, cwd=REPO_REMOTE, env=env)
    vol.commit()
    if p.returncode != 0:
        raise SystemExit(f"{mode} SFT exited {p.returncode}")
    return {"mode": mode, "save_dir": save_dir, "returncode": p.returncode}


@app.function(gpu="B200", volumes={"/vol": vol}, timeout=22 * 60 * 60,
              secrets=[_secret()])
def train_av_sft(steps: int = 0, bs: int = 32, accum: int = 2,
                 ckpt: bool = True):
    """AV verbalizer: activation -> explanation text. lr 1e-4 per the brief.

    Checkpointing OFF by default here: AV sequences are only ~270 tokens
    (prompt ~130 + response ~141), so at batch 64 that is ~17k tokens against a
    54GB bf16 model on a 180GB card — recompute cost dominated the memory it
    saved (14.2s/step with it on).
    """
    return _run_sft("av", 1e-4, [], steps, bs=bs, accum=accum, ckpt=ckpt)


@app.function(gpu="B200", volumes={"/vol": vol}, timeout=22 * 60 * 60,
              secrets=[_secret()])
def train_ar_sft(steps: int = 0, bs: int = 32, accum: int = 2,
                 ckpt: bool = True):
    """AR critic: explanation text -> activation. lr 2e-5 per the brief.

    --ar-num-layers = LAYER_INDEX + 1: the critic reads the OUTPUT of block 42,
    so block 42 must exist. A mismatch here silently trains a wrong-depth critic.
    """
    return _run_sft("ar", 2e-5, ["--ar-num-layers", str(LAYER_INDEX + 1)], steps,
                    bs=bs, accum=accum, ckpt=ckpt)
