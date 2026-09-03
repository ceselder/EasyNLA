"""Modal app for the NLA experiments (Qwen3-8B first, then Qwen3.6-27B).

One image (the B200-validated Qwen3.6 stack: vllm 0.21 + vllm-lens 1.1.0 +
transformers 5.5.4) serves every stage. The repo is mounted at container start
(add_local_dir copy=False), so code edits need no image rebuild.

Every long stage is launched DETACHED (an ephemeral app dies with the client):

  modal run --detach scripts/modal_nla_exp.py --task harness
  modal run --detach scripts/modal_nla_exp.py --task extract --nshards 4
  modal run --detach scripts/modal_nla_exp.py --task build
  modal run --detach scripts/modal_nla_exp.py --task sft --mode av --nproc 4 --tag av_sft
  modal run --detach scripts/modal_nla_exp.py --task rl --nproc 4 --tag rl_base --extra "..."

Volumes: /vol = nla-exp (ours). /vol_q36 = nla-qwen36-ema (read-only: the shared
Opus-5 row pool + the Qwen3.6 L42 activations + SFT checkpoints from August).
"""
import os
import shlex

import modal

# NLA_APP_NAME=nla-<run-tag> modal run ... -> one dashboard row per run instead of "nla-exp" x N
APP_NAME = os.environ.get("NLA_APP_NAME", "nla-exp")
VOL = "nla-exp"
VOL_Q36 = "nla-qwen36-ema"
REPO_LOCAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_REMOTE = "/root/easyNLA"
DATA = "/vol/data"
CKPT = "/vol/ckpts"
HF_CACHE = "/vol/hf_cache"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install("uv")
    # vllm 0.21.0 pins its own torch (cu13x wheel; Modal B200 hosts run driver>=580).
    # transformers 5.5.4 has qwen3_5 AND is accepted by vllm 0.21; vllm-lens 1.1.0's
    # hook still fires on 0.21 (GPUModelRunner.input_batch exists). Validated on
    # B200 for Qwen3.6-27B in July; Qwen3-8B uses the same stack.
    .run_commands(
        "uv pip install --system --python $(which python) "
        "'vllm==0.21.0' 'vllm-lens==1.1.0' 'transformers==5.5.4' "
        "peft bitsandbytes wandb accelerate datasets pyarrow pandas numpy "
        "anthropic openai 'huggingface_hub[hf_xet]' safetensors sentencepiece "
        "protobuf pyyaml orjson httpx tqdm flash-linear-attention scipy"
        # NB: no `kernels` — its current release breaks transformers 5.5.4's
        # hub_kernels import (LayerRepository requires revision/version).
    )
    .env({
        "HF_HOME": HF_CACHE,
        "HF_XET_HIGH_PERFORMANCE": "1",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        # auto-picked FLASHINFER JIT-compiles (needs nvcc) and dies in a slim image
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",   # sampler JIT needs nvcc (absent here)
        "VLLM_USE_DEEP_GEMM": "0",
        "VLLM_DEEP_GEMM_WARMUP": "skip",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "PYTHONPATH": REPO_REMOTE,
    })
    .add_local_dir(REPO_LOCAL, REPO_REMOTE, copy=False,
                   ignore=[".git", ".venv", "__pycache__", "*.pyc", "*.parquet"])
)

app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name(VOL, create_if_missing=True)
vol_q36 = modal.Volume.from_name(VOL_Q36)
SECRETS = [modal.Secret.from_name("nla-exp-secrets")]
VOLS = {"/vol": vol, "/vol_q36": vol_q36}

BASE_8B = "Qwen/Qwen3-8B"
LAYER_8B = 24
D_8B = 4096
OPUS5 = "ceselder/easynla-dsv4-warmstart-opus5"
TEST_PERMILLE = 10


def _prep(patch_lens: bool = True):
    """Per-container prep: apply the idempotent vllm-lens patch (site-packages is
    ephemeral, so it must run in every container BEFORE vllm_lens is imported)."""
    import subprocess
    import sys
    os.chdir(REPO_REMOTE)
    # A consumer launched seconds after a producer's print (but before its commit
    # propagated) saw a missing/partial checkpoint dir once; reload is cheap.
    try:
        vol.reload()
    except Exception as e:
        print(f"[prep] vol.reload skipped: {e}", flush=True)
    if patch_lens:
        r = subprocess.run([sys.executable, "utils/patch_vllm_lens.py"],
                           capture_output=True, text=True)
        print("[prep] patch_vllm_lens:", (r.stdout + r.stderr).strip()[-400:], flush=True)
        if r.returncode != 0:
            raise SystemExit("vllm-lens patch failed")
    os.environ.setdefault("WANDB_DIR", "/root/wandb")
    # FlashInfer's top-k/top-p sampler JIT-compiles with nvcc at engine init (the
    # slim image has no CUDA toolkit) -> use vLLM's PyTorch sampler instead.
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    # DeepGEMM (FP8) warmup imports a vendored deep_gemm that needs nvcc too; the
    # models here are bf16, so disable it outright.
    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    os.environ["VLLM_DEEP_GEMM_WARMUP"] = "skip"


def _run(cmd, env_extra=None, cwd=REPO_REMOTE):
    import subprocess
    print("CMD:", " ".join(shlex.quote(c) for c in cmd), flush=True)
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    p = subprocess.run(cmd, cwd=cwd, env=env)
    vol.commit()
    if p.returncode != 0:
        raise SystemExit(f"command exited {p.returncode}")


# --------------------------------------------------------------------------- checks
@app.function(gpu="B200", volumes=VOLS, timeout=30 * 60, secrets=SECRETS)
def harness_check():
    import subprocess
    import sys
    _prep()
    import torch
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout)
    print(f"torch {torch.__version__} cuda {torch.version.cuda} devices {torch.cuda.device_count()}")
    import transformers, vllm, vllm_lens, peft, bitsandbytes
    print(f"transformers {transformers.__version__} vllm {vllm.__version__} "
          f"vllm_lens {vllm_lens.__version__} peft {peft.__version__} bnb {bitsandbytes.__version__}")
    from nla.critic_ema import CriticEMA  # noqa
    from nla.utils.halluc_eval import judge_hallucination  # noqa
    import vllm_lens._worker_ext as wx
    src = open(wx.__file__).read()
    for marker in ("_meta5", "get_and_reset_steer_log", "log_key=per_req_log_key",
                   "get_and_reset_steer_count"):
        print(f"  lens patch marker {marker!r}: {marker in src}")
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    print("matmul ok:", (x @ x).float().abs().mean().item())
    for d in (DATA, CKPT, HF_CACHE):
        os.makedirs(d, exist_ok=True)
    print("q36 volume:", os.listdir("/vol_q36"), os.listdir("/vol_q36/data"))
    vol.commit()
    return "ok"


# ----------------------------------------------------------------- stage 0: extract
@app.function(gpu="B200", volumes=VOLS, timeout=6 * 60 * 60, secrets=SECRETS)
def extract_qwen3_8b(shard: int = 0, nshards: int = 1, limit: int = 0,
                     max_length: int = 4096, token_budget: int = 196608):
    """Re-extract Qwen3-8B layer-24 residuals (output of block 24, LAST real token)
    for the shared Opus-5 row pool. Left-truncates long prefixes so the final token
    is always the true capture position (a right-truncation would silently move it).
    """
    import json
    import time
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _prep(patch_lens=False)
    from nla.utils.arch_adapters import resolve_decoder_layers

    outdir = f"{DATA}/acts_qwen3_8b_L{LAYER_8B}"
    os.makedirs(outdir, exist_ok=True)
    pool = pq.read_table("/vol_q36/data/shared_pool.parquet").to_pydict()
    n_all = len(pool["text"])
    idx = [i for i in range(n_all) if i % nshards == shard]
    if limit:
        idx = idx[:limit]
    print(f"pool={n_all} shard {shard}/{nshards} -> {len(idx)} rows", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE_8B)
    tok.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        BASE_8B, dtype=torch.bfloat16, device_map="cuda:0", low_cpu_mem_usage=True,
        attn_implementation="sdpa")
    model.eval()
    layers = resolve_decoder_layers(model)
    assert len(layers) == 36
    enc = tok([pool["text"][i] for i in idx], add_special_tokens=False,
              truncation=True, max_length=max_length)["input_ids"]
    lens = [len(e) for e in enc]
    n_trunc = sum(1 for L in lens if L >= max_length)
    order = sorted(range(len(idx)), key=lambda k: lens[k])
    print(f"tokens: min={min(lens)} med={lens[order[len(order)//2]]} max={max(lens)} "
          f"truncated={n_trunc}", flush=True)

    grab = {}

    class _Stop(Exception):
        pass

    def hook(_m, _i, out):
        grab["h"] = out[0] if isinstance(out, tuple) else out
        raise _Stop

    handle = layers[LAYER_8B].register_forward_hook(hook)
    SHARD_ROWS = 25000
    buf_vec, buf_meta, norms = [], [], []
    part, done, t0 = 0, 0, time.time()

    def flush():
        nonlocal part, buf_vec, buf_meta
        if not buf_vec:
            return
        arr = np.stack(buf_vec).astype(np.float32)
        t = pa.table({
            "doc_id": pa.array([m[0] for m in buf_meta]),
            "detokenized_text_truncated": pa.array([m[1] for m in buf_meta]),
            "explanation": pa.array([m[2] for m in buf_meta]),
            "n_raw_tokens": pa.array([m[3] for m in buf_meta], pa.int64()),
            "n_tokens_qwen3": pa.array([m[4] for m in buf_meta], pa.int64()),
            "activation_layer": pa.array([LAYER_8B] * len(buf_meta), pa.int64()),
            "activation_vector": pa.FixedSizeListArray.from_arrays(
                pa.array(arr.reshape(-1)), arr.shape[1]),
        })
        p = f"{outdir}/shard{shard:02d}_part{part:03d}.parquet"
        pq.write_table(t, p, compression="zstd", row_group_size=5000)
        print(f"  wrote {p} ({len(buf_meta)} rows)", flush=True)
        part += 1
        buf_vec, buf_meta = [], []
        vol.commit()

    i_ord, nb = 0, 0
    while i_ord < len(order):
        n = 0
        while i_ord + n < len(order) and n < 512:
            cand = lens[order[i_ord + n]]
            if n > 0 and (n + 1) * cand > token_budget:
                break
            n += 1
        chunk = order[i_ord:i_ord + n]
        i_ord += n
        nb += 1
        ml = max(lens[k] for k in chunk)
        bx = torch.full((len(chunk), ml), tok.pad_token_id or 0, dtype=torch.long)
        am = torch.zeros((len(chunk), ml), dtype=torch.long)
        for r, k in enumerate(chunk):
            bx[r, :lens[k]] = torch.tensor(enc[k], dtype=torch.long)
            am[r, :lens[k]] = 1
        bx, am = bx.cuda(), am.cuda()
        try:
            with torch.no_grad():
                model.model(input_ids=bx, attention_mask=am, use_cache=False)
        except _Stop:
            pass
        h = grab["h"]
        last = am.sum(1) - 1
        vecs = h[torch.arange(h.shape[0], device=h.device), last].float()
        norms.append(vecs.norm(dim=-1).cpu().numpy())
        vn = vecs.cpu().numpy()
        for r, k in enumerate(chunk):
            i = idx[k]
            buf_vec.append(vn[r])
            buf_meta.append((pool["doc_id"][i], pool["text"][i], pool["explanation"][i],
                             int(pool["n_raw_tokens"][i]), lens[k]))
        done += len(chunk)
        if len(buf_vec) >= SHARD_ROWS:
            flush()
        if nb % 25 == 1:
            el = time.time() - t0
            print(f"  {done}/{len(order)} {100*done/len(order):.1f}% {done/el:.0f} rows/s "
                  f"eta {(len(order)-done)/max(done/el,1e-9)/60:.0f} min", flush=True)
    flush()
    handle.remove()
    alln = np.concatenate(norms)
    stats = {"shard": shard, "nshards": nshards, "rows": done, "d_model": D_8B,
             "layer_index": LAYER_8B, "max_length": max_length, "n_truncated_left": n_trunc,
             "norm_p25": float(np.percentile(alln, 25)), "norm_p50": float(np.percentile(alln, 50)),
             "norm_p75": float(np.percentile(alln, 75)), "norm_p95": float(np.percentile(alln, 95)),
             "elapsed_min": (time.time() - t0) / 60}
    json.dump(stats, open(f"{outdir}/stats_shard{shard:02d}.json", "w"), indent=2)
    open(f"{outdir}/_COMPLETE_shard{shard:02d}", "w").write(json.dumps(stats))
    vol.commit()
    print(json.dumps(stats, indent=2), flush=True)
    return stats


# ------------------------------------------------------------ stage 3: build SFT data
@app.function(volumes=VOLS, timeout=3 * 60 * 60, cpu=8.0, memory=131072, secrets=SECRETS)
def build_datasets(model_tag: str = "qwen3_8b", n_eval_fixed: int = 1024):
    """AV/AR SFT parquets + sidecars from the extracted shards, 99/1 doc-level split
    (nla.val_split.is_val_doc(doc_id, 10) — the same rule as the HF dataset)."""
    import glob
    import json
    import random
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    _prep(patch_lens=False)
    from nla.datagen.injection_tokens import build_token_meta
    from nla.val_split import is_val_doc
    from nla.schema import INJECT_PLACEHOLDER

    if model_tag == "qwen3_8b":
        base, layer, d = BASE_8B, LAYER_8B, D_8B
        actdir = f"{DATA}/acts_qwen3_8b_L{LAYER_8B}"
        complete = glob.glob(f"{actdir}/_COMPLETE_shard*")
        shards = sorted(glob.glob(f"{actdir}/shard*_part*.parquet"))
        assert complete and shards, f"no completed extraction under {actdir}"
        n_sh = json.loads(open(complete[0]).read())["nshards"]
        assert len(complete) == n_sh, f"{len(complete)}/{n_sh} shards complete"
    elif model_tag == "qwen36_27b":
        # August extraction (nla-qwen36-ema volume): 30 shards, all 742k rows, cols
        # doc_id/text/explanation/is_val/n_raw_tokens/activation_layer/activation_vector
        base, layer, d = "Qwen/Qwen3.6-27B", 42, 5120
        actdir = "/vol_q36/data/acts_qwen36_L42"
        assert os.path.exists(f"{actdir}/_COMPLETE"), f"{actdir} incomplete"
        shards = sorted(glob.glob(f"{actdir}/shard_*.parquet"))
    else:
        raise SystemExit(model_tag)

    side = yaml.safe_load(open(hf_hub_download(
        OPUS5, "ar_sft_shuf.parquet.nla_meta.yaml", repo_type="dataset")))
    side_av = yaml.safe_load(open(hf_hub_download(
        OPUS5, "av_sft_shuf.parquet.nla_meta.yaml", repo_type="dataset")))
    ACTOR = side_av["prompt_templates"]["actor"]
    CRITIC = side["prompt_templates"]["critic"]
    assert "{injection_char}" in ACTOR and "{explanation}" in CRITIC
    tok = AutoTokenizer.from_pretrained(base)
    tmeta = build_token_meta(tok, ACTOR, critic_template=CRITIC)
    print(f"tokens: {tmeta}", flush=True)
    ACTOR_PH = ACTOR.replace("{injection_char}", INJECT_PLACEHOLDER)

    tables = []
    for sp in shards:
        t = pq.read_table(sp)
        if "text" in t.column_names and "detokenized_text_truncated" not in t.column_names:
            t = t.rename_columns([("detokenized_text_truncated" if c == "text" else c)
                                  for c in t.column_names])
        keep = ["doc_id", "detokenized_text_truncated", "explanation", "n_raw_tokens",
                "activation_layer", "activation_vector"]
        tables.append(t.select(keep))
    tbl = pa.concat_tables(tables)
    del tables
    dids = tbl.column("doc_id").to_pylist()
    is_test = [is_val_doc(x, TEST_PERMILLE) for x in dids]
    tr_idx = [i for i, t in enumerate(is_test) if not t]
    te_idx = [i for i, t in enumerate(is_test) if t]
    random.Random(0).shuffle(tr_idx)
    random.Random(1).shuffle(te_idx)
    print(f"rows={tbl.num_rows} train={len(tr_idx)} test={len(te_idx)} "
          f"docs train={len({dids[i] for i in tr_idx})} test={len({dids[i] for i in te_idx})}", flush=True)
    assert not ({dids[i] for i in tr_idx} & {dids[i] for i in te_idx})

    out = f"{DATA}/{model_tag}"
    os.makedirs(out, exist_ok=True)

    def emit(idx, stage, split):
        sub = tbl.take(pa.array(idx))
        expl = sub.column("explanation").to_pylist()
        n = len(idx)
        cols = {
            "activation_vector": sub.column("activation_vector").cast(pa.list_(pa.float32(), d)),
            "n_raw_tokens": sub.column("n_raw_tokens"),
            "activation_layer": sub.column("activation_layer"),
            "doc_id": sub.column("doc_id"),
            "detokenized_text_truncated": sub.column("detokenized_text_truncated"),
        }
        if stage == "av":
            cols["prompt"] = pa.array([[{"role": "user", "content": ACTOR_PH}]] * n)
            cols["response"] = pa.array([f"<explanation>\n{e}\n</explanation>" for e in expl])
        else:
            cols["prompt"] = pa.array([CRITIC.replace("{explanation}", e) for e in expl])
        path = f"{out}/{stage}_sft_{split}.parquet"
        pq.write_table(pa.table(cols), path, compression="zstd", row_group_size=5000)
        toks = {"injection_char": tmeta.injection_char,
                "injection_token_id": tmeta.injection_token_id,
                "injection_left_neighbor_id": tmeta.injection_left_neighbor_id,
                "injection_right_neighbor_id": tmeta.injection_right_neighbor_id,
                "critic_suffix_ids": tmeta.critic_suffix_ids if stage == "ar" else None}
        meta = {
            "dataset_id": f"{stage}_sft_{model_tag}_L{layer}_opus5expl_{split}",
            "stage": f"{stage}_sft", "row_count": n,
            "extraction": {"base_model": base, "d_model": d, "layer_index": layer,
                           "norm": "none", "corpus": "finefineweb prefixes (Opus-5 pool)",
                           "positions_per_doc": 10, "test_doc_permille": TEST_PERMILLE},
            "kind": "nla_dataset", "schema_version": 1, "keep_debug_metadata": True,
            "tokens": toks, "prompt_templates": {"actor": ACTOR, "critic": CRITIC},
            "api_summaries": {"model": "claude-opus-5",
                              "note": f"explanations from {OPUS5}; activations re-extracted "
                                      f"on {base} layer {layer} (last token, left-truncated 4096)"},
        }
        yaml.safe_dump(meta, open(f"{path}.nla_meta.yaml", "w"), sort_keys=False, allow_unicode=True)
        print(f"  wrote {path} rows={n} {os.path.getsize(path)/1e9:.2f} GB", flush=True)
        return n

    counts = {}
    for stage in ("av", "ar"):
        counts[f"{stage}_train"] = emit(tr_idx, stage, "train")
        counts[f"{stage}_test"] = emit(te_idx, stage, "test")
    # fixed eval subset (first n_eval_fixed shuffled test rows) for cheap repeated evals
    counts["av_eval"] = emit(te_idx[:n_eval_fixed], "av", "eval")
    vol.commit()

    # validate through the repo's own loader (the same asserts the trainers run)
    from nla.config import load_nla_config, verify_critic_suffix
    for stage in ("av", "ar"):
        cfg = load_nla_config(f"{out}/{stage}_sft_train.parquet", tok)
        print(f"  validated {stage}: inj={cfg.injection_token_id} L/R="
              f"{cfg.injection_left_neighbor_id}/{cfg.injection_right_neighbor_id} "
              f"mse_scale={cfg.mse_scale:.2f} layer={cfg.extraction_layer_index}", flush=True)
    t = pq.ParquetFile(f"{out}/ar_sft_train.parquet").read_row_group(0).slice(0, 3)
    for i in range(3):
        verify_critic_suffix(tok.encode(t.column("prompt")[i].as_py(), add_special_tokens=False),
                             tmeta.critic_suffix_ids, context=f"row {i}")
    print("counts:", counts, flush=True)
    return counts


# --------------------------------------------------------------------- SFT / merge
@app.function(gpu="B200:4", volumes=VOLS, timeout=23 * 60 * 60, secrets=SECRETS)
def train_sft(mode: str, tag: str, nproc: int = 4, model_tag: str = "qwen3_8b",
              bs: int = 16, accum: int = 1, extra: str = "", data_dir: str = "",
              data_suffix: str = ""):
    """torchrun -m nla.train_sft (one epoch by default). extra = additional CLI args."""
    import sys
    _prep(patch_lens=False)
    base = BASE_8B if model_tag == "qwen3_8b" else "Qwen/Qwen3.6-27B"
    layer = LAYER_8B if model_tag == "qwen3_8b" else 42
    ddir = data_dir or f"{DATA}/{model_tag}"
    save_dir = f"{CKPT}/{model_tag}/{tag}"
    os.makedirs(save_dir, exist_ok=True)
    data = f"{ddir}/{mode}_sft_train{data_suffix}.parquet"
    val = f"{ddir}/av_sft_test.parquet"
    launcher = ([sys.executable, "-m", "torch.distributed.run", "--standalone",
                 f"--nproc_per_node={nproc}"] if nproc > 1 else [sys.executable])
    cmd = launcher + [
        "-m", "nla.train_sft", "--mode", mode, "--base-ckpt", base,
        "--parquet", data, "--sidecar", data, "--heldout-parquet", val,
        "--save-dir", save_dir, "--batch-size", str(bs),
        "--gradient-accumulation-steps", str(accum), "--max-len", "1024",
        "--save-every", "1000", "--heldout-every", "250", "--seed", "0",
        "--wandb-project", f"nla-exp-{model_tag}", "--wandb-name", tag,
        "--wandb-tags", f"{model_tag},L{layer},{mode}_sft",
    ]
    if mode == "ar":
        cmd += ["--ar-num-layers", str(layer + 1)]
    cmd += shlex.split(extra)
    _run(cmd, env_extra={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    return save_dir


@app.function(gpu="B200", volumes=VOLS, timeout=2 * 60 * 60, secrets=SECRETS)
def merge_av(av_dir: str, out: str, model_tag: str = "qwen3_8b"):
    import sys
    _prep(patch_lens=False)
    base = BASE_8B if model_tag == "qwen3_8b" else "Qwen/Qwen3.6-27B"
    _run([sys.executable, "scripts/merge_lora_to_hf.py", "--base-ckpt", base,
          "--av-dir", av_dir, "--av-out", out, "--ar-dir", "/dev/null",
          "--ar-out", "/dev/null", "--mode", "av"])
    return out


# ------------------------------------------------------------------------------ RL
@app.function(gpu="B200:4", volumes=VOLS, timeout=23 * 60 * 60, secrets=SECRETS)
def train_rl(tag: str, nproc: int = 4, model_tag: str = "qwen3_8b", extra: str = "",
             config: str = "configs/rl_vllm.yaml"):
    """torchrun -m nla.train_rl_vllm --config <config> <extra>. All run-specific
    arguments (checkpoints, data, reward mode, EMA, ...) come through `extra`."""
    import sys
    _prep(patch_lens=True)
    save_dir = f"{CKPT}/{model_tag}/{tag}"
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
           f"--nproc_per_node={nproc}", "-m", "nla.train_rl_vllm",
           "--config", config, "--save-dir", save_dir,
           "--wandb-project", f"nla-exp-{model_tag}", "--wandb-name", tag]
    cmd += shlex.split(extra)
    # IPC weight sync needs the legacy allocator (no expandable_segments).
    env = dict(os.environ)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    import subprocess
    print("CMD:", " ".join(shlex.quote(c) for c in cmd), flush=True)
    p = subprocess.run(cmd, cwd=REPO_REMOTE, env=env)
    vol.commit()
    if p.returncode != 0:
        raise SystemExit(f"rl exited {p.returncode}")
    return save_dir


# --------------------------------------------------------------------- generic shell
@app.function(gpu="B200", volumes=VOLS, timeout=23 * 60 * 60, secrets=SECRETS)
def shell(cmd: str, patch_lens: bool = True):
    """Run an arbitrary command in the repo on one B200 (scripts: mining, judging, eval)."""
    _prep(patch_lens=patch_lens)
    _run(["bash", "-lc", cmd])
    return "ok"



@app.function(volumes=VOLS, timeout=20 * 60, cpu=4.0, memory=16384, secrets=SECRETS)
def probe_tokenizer(merged: str = f"{CKPT}/qwen3_8b/av_sft_merged", base: str = BASE_8B,
                    char: str = "㈎"):
    """CPU check: does the saved tokenizer round-trip the injection char?"""
    import json
    import transformers
    from transformers import AutoTokenizer
    _prep(patch_lens=False)
    vol.reload()
    print("transformers", transformers.__version__)
    print("merged dir:", sorted(os.listdir(merged)) if os.path.isdir(merged) else "MISSING")
    for name in (base, merged):
        try:
            t = AutoTokenizer.from_pretrained(name)
            ids = t.encode(char, add_special_tokens=False)
            print(f"{name}: {type(t).__name__} vocab={len(t)} encode({char!r})={ids} "
                  f"decode={t.decode(ids)!r} | 'hello'->{t.encode('hello', add_special_tokens=False)}")
        except Exception as e:
            print(f"{name}: FAILED {type(e).__name__}: {str(e)[:200]}")
    cfg = os.path.join(merged, "tokenizer_config.json")
    if os.path.exists(cfg):
        d = json.load(open(cfg))
        print("tokenizer_config keys:", sorted(d)[:40])
        print("class:", d.get("tokenizer_class"), "| add_prefix_space:", d.get("add_prefix_space"),
              "| split_special_tokens:", d.get("split_special_tokens"))
    return "ok"


@app.function(volumes=VOLS, timeout=24 * 60 * 60, cpu=4.0, memory=32768, secrets=SECRETS)
def judge_batch(rollouts_dir: str, source_parquet: str, out_dir: str, phase: str = "both",
                limit: int = 0, max_parts: int = 0, per_batch: int = 40000):
    """Sonnet-5 bulk hallucination scoring through the Anthropic Batch API (CPU only)."""
    import sys
    _prep(patch_lens=False)
    cmd = [sys.executable, "scripts/judge_hallucination_batch.py", "--phase", phase,
           "--rollouts-dir", rollouts_dir, "--source-parquet", source_parquet,
           "--out-dir", out_dir, "--per-batch", str(per_batch)]
    if limit:
        cmd += ["--limit", str(limit)]
    if max_parts:
        cmd += ["--max-parts", str(max_parts)]
    _run(cmd)
    return out_dir


@app.function(volumes=VOLS, timeout=24 * 60 * 60, cpu=4.0, memory=32768, secrets=SECRETS)
def judge_sync(rollouts_dir: str, source_parquet: str, out_dir: str,
               part_glob: str = "rollouts_shard*_part*.parquet", concurrency: int = 128,
               follow: bool = True, n_complete: int = 4):
    """Sonnet-5 bulk scoring over the synchronous API, high-priority key first."""
    import sys
    _prep(patch_lens=False)
    cmd = [sys.executable, "scripts/judge_hallucination_sync.py", "--rollouts-dir", rollouts_dir,
           "--source-parquet", source_parquet, "--out-dir", out_dir, "--part-glob", part_glob,
           "--concurrency", str(concurrency), "--n-complete", str(n_complete)]
    if follow:
        cmd.append("--follow")
    _run(cmd, env_extra={"NLA_JUDGE_PREFER_FALLBACK": "1"})
    return out_dir


@app.function(volumes=VOLS, timeout=2 * 60 * 60, cpu=8.0, memory=65536, secrets=SECRETS)
def split_sft_rl(model_tag: str = "qwen3_8b", n_sft: int = 500_000):
    """Protocol: warm-start (AV and AR, same rows) on the first n_sft rows of the shuffled
    train split; the REST of train is the RL activation pool. Writes
    {av,ar}_sft_train500k.parquet + av_sft_rl.parquet (+ sidecars) next to the originals."""
    import shutil
    import pyarrow.parquet as pq
    import yaml
    _prep(patch_lens=False)
    d = f"{DATA}/{model_tag}"
    out = {}
    for stage in ("av", "ar"):
        src = f"{d}/{stage}_sft_train.parquet"
        pf = pq.ParquetFile(src)
        side = yaml.safe_load(open(src + ".nla_meta.yaml"))
        sft_path = f"{d}/{stage}_sft_train500k.parquet"
        rl_path = f"{d}/{stage}_sft_rl.parquet"
        w_sft = pq.ParquetWriter(sft_path, pf.schema_arrow, compression="zstd")
        w_rl = pq.ParquetWriter(rl_path, pf.schema_arrow, compression="zstd")
        n_s = n_r = 0
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg)
            if n_s < n_sft:
                take = min(n_sft - n_s, t.num_rows)
                w_sft.write_table(t.slice(0, take), row_group_size=5000); n_s += take
                if take < t.num_rows:
                    w_rl.write_table(t.slice(take), row_group_size=5000); n_r += t.num_rows - take
            else:
                w_rl.write_table(t, row_group_size=5000); n_r += t.num_rows
        w_sft.close(); w_rl.close()
        for path, n, tag in ((sft_path, n_s, "sft500k"), (rl_path, n_r, "rl")):
            m = dict(side); m["row_count"] = n; m["dataset_id"] = side["dataset_id"] + "_" + tag
            yaml.safe_dump(m, open(path + ".nla_meta.yaml", "w"), sort_keys=False, allow_unicode=True)
        out[stage] = {"sft": n_s, "rl": n_r}
        print(f"{stage}: sft={n_s} rl={n_r}", flush=True)
    vol.commit()
    return out


@app.function(volumes=VOLS, timeout=6 * 60 * 60, cpu=8.0, memory=65536, secrets=SECRETS)
def pyrun(cmd: str):
    """Run a CPU-only command in the repo with the volumes mounted (stats, data builds)."""
    _prep(patch_lens=False)
    _run(["bash", "-lc", cmd])
    return "ok"

# ------------------------------------------------------------------------ entrypoint
@app.local_entrypoint()
def main(task: str, mode: str = "av", tag: str = "", nproc: int = 4, nshards: int = 1,
         limit: int = 0, extra: str = "", model_tag: str = "qwen3_8b", gpus: int = 0,
         bs: int = 16, accum: int = 1, config: str = "configs/rl_vllm.yaml",
         av_dir: str = "", out: str = "", cmd: str = "", data_dir: str = "",
         glob: str = "", ncomplete: int = 0):
    if task == "harness":
        print(harness_check.remote())
    elif task == "extract":
        calls = [extract_qwen3_8b.spawn(shard=s, nshards=nshards, limit=limit)
                 for s in range(nshards)]
        for c in calls:
            print(c.get())
    elif task == "build":
        print(build_datasets.remote(model_tag=model_tag))
    elif task == "sft":
        f = train_sft.with_options(gpu=f"B200:{gpus or nproc}")
        print(f.remote(mode=mode, tag=tag, nproc=nproc, model_tag=model_tag,
                       bs=bs, accum=accum, extra=extra, data_dir=data_dir, data_suffix=out))
    elif task == "merge_av":
        print(merge_av.remote(av_dir=av_dir, out=out, model_tag=model_tag))
    elif task == "rl":
        f = train_rl.with_options(gpu=f"B200:{gpus or nproc}")
        print(f.remote(tag=tag, nproc=nproc, model_tag=model_tag, extra=extra, config=config))
    elif task == "shell":
        f = shell.with_options(gpu=f"B200:{gpus or 1}")
        print(f.remote(cmd=cmd))
    elif task == "judge_sync":
        # --glob: ONE scorer on that part glob (--ncomplete = #_COMPLETE files that end
        # follow mode); else one scorer per mining shard, all polling the same dir
        rd, sp, od = cmd.split("|")
        if glob:
            print(judge_sync.remote(rollouts_dir=rd, source_parquet=sp, out_dir=od, part_glob=glob,
                                    concurrency=limit or 128, n_complete=ncomplete or nshards))
        else:
            calls = [judge_sync.spawn(rollouts_dir=rd, source_parquet=sp, out_dir=od,
                                      part_glob=f"rollouts_shard{s:02d}_*part*.parquet",
                                      concurrency=limit or 128, n_complete=ncomplete or nshards)
                     for s in range(nshards)]
            for c in calls:
                print(c.get())
    elif task == "judge_batch":
        print(judge_batch.remote(rollouts_dir=cmd.split("|")[0], source_parquet=cmd.split("|")[1],
                                 out_dir=cmd.split("|")[2], phase=mode, limit=limit, max_parts=nshards))
    elif task == "split":
        print(split_sft_rl.remote(model_tag=model_tag, n_sft=limit or 500_000))
    elif task == "pyrun":
        print(pyrun.remote(cmd=cmd))
    elif task == "probe_tok":
        print(probe_tokenizer.remote())
    elif task == "shells":
        # N parallel one-GPU shells; `{shard}` / `{nshards}` are substituted in cmd.
        f = shell.with_options(gpu=f"B200:{gpus or 1}")
        calls = [f.spawn(cmd=cmd.format(shard=s, nshards=nshards)) for s in range(nshards)]
        for c in calls:
            print(c.get())
    else:
        raise SystemExit(f"unknown task {task}")
