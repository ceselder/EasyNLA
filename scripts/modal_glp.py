"""Modal app for pretraining the activation flow prior (GLP-style) on Qwen3.6-27B layer-42 activations.
  modal run scripts/modal_glp.py --task smoke                                   # B200:3, tiny model, end-to-end check
  modal run --detach scripts/modal_glp.py --task pretrain --tag glp27b_default     # B200:8, configs/glp/default_27b_l42.yaml
  modal run --detach scripts/modal_glp.py --task pretrain --tag glp27b_lr1e4 --sets "train.lr=1e-4"   # one-flag ablation
"""
import os, sys
import modal
for _p in (os.path.dirname(os.path.abspath(__file__)), os.path.join(os.environ.get("PYTHONPATH", "/root/easyNLA").split(":")[0], "scripts")):
    if os.path.isdir(_p) and _p not in sys.path: sys.path.insert(0, _p)
from modal_nla_exp import image_base, SECRETS, REPO_LOCAL, REPO_REMOTE, REPO_IGNORE  # noqa: E402

vol_glp = modal.Volume.from_name("nla-glp", create_if_missing=True)
vol_exp = modal.Volume.from_name("nla-exp")
vol_q36 = modal.Volume.from_name("nla-qwen36-ema")
VOLS = {"/vol_glp": vol_glp, "/vol": vol_exp, "/vol_q36": vol_q36}
image = image_base.add_local_dir(REPO_LOCAL, REPO_REMOTE, copy=False, ignore=REPO_IGNORE)
app = modal.App("nla-glp", image=image)
COMMON = dict(volumes=VOLS, secrets=SECRETS, cpu=32, memory=256 * 1024, ephemeral_disk=600 * 1024)


def _run(args):
    import subprocess
    os.environ.setdefault("WANDB_DIR", "/root/wandb"); os.makedirs("/root/wandb", exist_ok=True)
    cmd = [sys.executable, f"{REPO_REMOTE}/scripts/glp_launch.py"] + args
    print("[modal] " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=REPO_REMOTE)
    vol_glp.commit()
    return rc


@app.function(gpu="B200:8", timeout=23 * 3600, **COMMON)
def pretrain(tag: str, config: str = "", sets: str = ""):
    args = ["--out-dir", f"/vol_glp/{tag}", "--wandb-name", tag]
    if config: args += ["--config", f"{REPO_REMOTE}/{config}"]
    if sets.strip(): args += ["--set"] + sets.split()
    return _run(args)


@app.function(gpu="B200:4", timeout=23 * 3600, **COMMON)
def pretrain_g4(tag: str, config: str = "", sets: str = ""):
    """Same as pretrain on a 4-GPU node (2 producers + 2 consumers) for when B200:8 capacity is scarce."""
    args = ["--out-dir", f"/vol_glp/{tag}", "--wandb-name", tag, "--set", "gpus.producers=2", "gpus.consumers=2"]
    if config: args[-3:-3] = ["--config", f"{REPO_REMOTE}/{config}"]
    if sets.strip(): args += sets.split()
    return _run(args)


@app.function(gpu="B200:3", timeout=2 * 3600, **COMMON)
def smoke(tag: str = "smoke_glp", sets: str = ""):
    args = ["--out-dir", f"/vol_glp/{tag}", "--wandb-name", tag, "--config", f"{REPO_REMOTE}/configs/glp/smoke.yaml"]
    if sets.strip(): args += ["--set"] + sets.split()
    return _run(args)


@app.function(gpu="B200", timeout=3 * 3600, **COMMON)
def eval_lm(tag: str, ckpt: str = "final", extra: str = ""):
    """Delta-LM-loss eval of a trained prior on the held-out docs (needs the FULL 27B)."""
    import subprocess
    from playground_app import resolve_base
    base = resolve_base("Qwen/Qwen3.6-27B", local_snapshot=True)
    out = f"/vol_glp/{tag}/eval_lm_{ckpt}{'_raw' if '--weights raw' in extra else ''}.json"
    cmd = [sys.executable, "-m", "nla.flow.eval_lm", "--base", base, "--ckpt", f"/vol_glp/{tag}/ckpts/{ckpt}", "--stats", f"/vol_glp/{tag}/rep_statistics.pt",
           "--heldout", f"/vol_glp/{tag}/heldout_acts.pt", "--out", out] + extra.split()
    rc = subprocess.call(cmd, cwd=REPO_REMOTE); vol_glp.commit(); return rc


@app.function(gpu="B200:2", timeout=12 * 3600, **COMMON)
def gen_onpolicy(n_prompts: int = 300000, extra: str = ""):
    """vLLM: Qwen3.6-27B responses to WildChat first-turn prompts -> /vol_glp/data/wildchat_onpolicy_*.parquet (+ wildchat_original.parquet)."""
    import subprocess
    from playground_app import resolve_base
    # same vLLM env as the RL path: FlashInfer / DeepGEMM JIT need nvcc, which the image lacks
    os.environ.update({"VLLM_ATTENTION_BACKEND": "FLASH_ATTN", "VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_USE_DEEP_GEMM": "0", "NLA_VLLM_GRAPHS": "0"})
    base = resolve_base("Qwen/Qwen3.6-27B", local_snapshot=True)
    cmd = [sys.executable, "-m", "nla.flow.gen_onpolicy", "--base", base, "--out-dir", "/vol_glp/data", "--n-prompts", str(n_prompts), "--tp", "2"] + extra.split()
    rc = subprocess.call(cmd, cwd=REPO_REMOTE); vol_glp.commit(); return rc


@app.function(gpu="B200", timeout=2 * 3600, **COMMON)
def bnoise(tag: str, ckpt: str = "init", extra: str = ""):
    """Gradient-noise-scale / critical-batch estimate at a checkpoint (or at init) of run <tag>."""
    import subprocess
    ck = "init" if ckpt == "init" else f"/vol_glp/{tag}/ckpts/{ckpt}"
    out = f"/vol_glp/{tag}/bnoise_{ckpt}.json"
    cmd = [sys.executable, "-m", "nla.flow.bnoise", "--ckpt", ck, "--stats", f"/vol_glp/{tag}/rep_statistics.pt", "--heldout", f"/vol_glp/{tag}/heldout_acts.pt", "--out", out] + extra.split()
    rc = subprocess.call(cmd, cwd=REPO_REMOTE); vol_glp.commit(); return rc


@app.function(gpu="B200", timeout=1800, **COMMON)
def bench_producer(attn_impl: str = "sdpa", tokens_per_batch: int = 32768, max_tokens: float = 4e6, extra: str = ""):
    """Time the activation producer alone (tok/s) under different settings; writes shards to local disk only."""
    import subprocess, json, time
    from playground_app import resolve_base
    base = resolve_base("Qwen/Qwen3.6-27B", local_snapshot=True)
    try:
        import flash_attn; print("[bench] flash_attn", flash_attn.__version__, flush=True)
    except Exception as e:
        print("[bench] flash_attn not importable:", e, flush=True)
    src = json.dumps([{"name": "fineweb", "kind": "hf_text", "dataset": "HuggingFaceFW/fineweb", "config": "sample-10BT", "weight": 1.0}])
    out = f"/tmp/bench_{attn_impl}_{tokens_per_batch}"; os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "-m", "nla.flow.produce", "--base", base, "--layer", "42", "--n-producers", "1", "--index", "0", "--shard-dir", "/tmp/shards_bench",
           "--out-dir", out, "--sources-json", src, "--attn-impl", attn_impl, "--tokens-per-batch", str(tokens_per_batch), "--max-tokens", str(max_tokens),
           "--stats-n", "1000", "--heldout-n", "1000", "--heldout-docs-full", "1", "--max-ready", "100000"] + extra.split()
    t = time.time(); rc = subprocess.call(cmd, cwd=REPO_REMOTE); dt = time.time() - t
    prog = json.load(open(f"{out}/progress_0.json"))
    print(f"[bench] attn={attn_impl} tokens_per_batch={tokens_per_batch}: {prog['tokens']/1e6:.1f}M tokens in {dt:.0f}s incl. model load -> {prog['tokens']/dt/1e3:.1f}k tok/s (wall)", flush=True)
    return rc


@app.function(gpu="B200", timeout=12 * 3600, **COMMON)
def gumbel_av(tag: str, extra: str = ""):
    """De-Diffusion-style end-to-end AV training through the frozen 8B MSE critic (Gumbel-softmax text)."""
    import subprocess
    out = f"/vol_glp/gumbel/{tag}"
    cmd = [sys.executable, "-m", "nla.flow.gumbel_av", "--base", "Qwen/Qwen3-8B", "--av-adapter", "/vol/ckpts/qwen3_8b/av_sft500k_lr1e4/iter_0007813",
           "--critic", "/vol/ckpts/qwen3_8b/ar_sft500k/iter_0007813", "--train-parquet", "/vol/data/qwen3_8b/av_sft_rl.parquet",
           "--eval-parquet", "/vol/data/qwen3_8b/av_sft_eval.parquet", "--out", out, "--tag", tag] + extra.split()
    rc = subprocess.call(cmd, cwd=REPO_REMOTE); vol_glp.commit(); return rc


@app.function(gpu="B200", timeout=3600, **COMMON)
def sample_diag(tag: str, ckpt: str, extra: str = ""):
    import subprocess
    out = f"/vol_glp/{tag}/sample_diag_{ckpt}.json"
    cmd = [sys.executable, "-m", "nla.flow.sample_diag", "--ckpt", f"/vol_glp/{tag}/ckpts/{ckpt}", "--stats", f"/vol_glp/{tag}/rep_statistics.pt", "--heldout", f"/vol_glp/{tag}/heldout_acts.pt", "--out", out] + extra.split()
    rc = subprocess.call(cmd, cwd=REPO_REMOTE); vol_glp.commit(); return rc


@app.function(gpu="B200", timeout=23 * 3600, **COMMON)
def train_cond(tag: str, prior_tag: str = "glp27b_main", prior_ckpt: str = "snap_000655M", extra: str = ""):
    """Stage 2: conditional adapter on (activation, explanation) pairs (27B SFT pairs), frozen prior + frozen 27B encoder."""
    import subprocess
    from playground_app import resolve_base
    base = resolve_base("Qwen/Qwen3.6-27B", local_snapshot=True)
    out = f"/vol_glp/cond/{tag}"
    cmd = [sys.executable, "-m", "nla.flow.train_cond", "--prior", f"/vol_glp/{prior_tag}/ckpts/{prior_ckpt}", "--stats", f"/vol_glp/{prior_tag}/rep_statistics.pt",
           "--base", base, "--train-parquet", "/vol_q36/data/sft/av_sft_train.parquet", "--val-parquet", "/vol_q36/data/sft/av_sft_val.parquet", "--out", out, "--tag", tag,
           "--mined-acts-parquet", "/vol_q36/data/rl/rl_shuf.parquet"] + extra.split()      # add --mined-dir /vol_glp/pairs/<tag> via extra to mix in on-policy pairs
    rc = subprocess.call(cmd, cwd=REPO_REMOTE); vol_glp.commit(); return rc


@app.function(gpu="B200", timeout=6 * 3600, **COMMON)
def mine_pairs(tag: str, av_merged: str, parquet: str = "/vol_q36/data/rl/rl_shuf.parquet", n_samples: int = 2, shard: int = 0, nshards: int = 8, extra: str = ""):
    """On-policy (activation, explanation) pairs: sample from a merged 27B AV through the vllm-lens injection path (same as RL rollouts)."""
    import subprocess
    from modal_nla_exp import _prep
    os.environ.update({"NLA_VLLM_GRAPHS": "0", "NLA_VLLM_EAGER": "1", "VLLM_ATTENTION_BACKEND": "FLASH_ATTN"})
    _prep(patch_lens=True)
    out = f"/vol_glp/pairs/{tag}"
    cmd = [sys.executable, f"{REPO_REMOTE}/scripts/mine_av_rollouts.py", "--av-ckpt", av_merged, "--parquet", parquet, "--sidecar", parquet, "--out-dir", out,
           "--n-samples", str(n_samples), "--shard", str(shard), "--nshards", str(nshards), "--vllm-gpu-mem", "0.85", "--vllm-max-len", "1024"] + extra.split()
    rc = subprocess.call(cmd, cwd=REPO_REMOTE); vol_glp.commit(); return rc


@app.function(gpu="B200", timeout=6 * 3600, **COMMON)
def ultra_extract(explained_glob: str = "/vol_glp/data/ultra_explained_t1/chunk_*.parquet", out_dir: str = "/vol_glp/data/ultra_L42", shard: int = 0, nshards: int = 1, extra: str = ""):
    """Layer-42 activations for Sonnet-5-explained UltraFineWeb prefixes -> extraction-schema shards (nla.flow.datagen_ultra extract)."""
    import subprocess
    from playground_app import resolve_base
    base = resolve_base("Qwen/Qwen3.6-27B", local_snapshot=True)
    cmd = [sys.executable, "-m", "nla.flow.datagen_ultra", "extract", "--explained-glob", explained_glob, "--out-dir", out_dir, "--base", base, "--shard", str(shard), "--nshards", str(nshards)] + extra.split()
    rc = subprocess.call(cmd, cwd=REPO_REMOTE); vol_glp.commit(); return rc


@app.function(timeout=2 * 3600, volumes=VOLS, secrets=SECRETS, cpu=16, memory=200 * 1024, ephemeral_disk=600 * 1024)   # Modal minimum is 512 GiB
def convert_wrapper(src: str, dst: str):
    """Text-only merged checkpoint -> vLLM wrapper layout (Qwen3_5ForConditionalGeneration)."""
    import subprocess
    from playground_app import resolve_base
    base = resolve_base("Qwen/Qwen3.6-27B", local_snapshot=True)
    rc = subprocess.call([sys.executable, f"{REPO_REMOTE}/scripts/convert_textonly_to_wrapper.py", "--src", src, "--dst", dst, "--base-config", f"{base}/config.json"], cwd=REPO_REMOTE)
    vol_exp.commit(); return rc


@app.function(gpu="B200", timeout=4 * 3600, **COMMON)
def eval_cond(tag: str, prior_tag: str = "glp27b_main", prior_ckpt: str = "snap_000655M", extra: str = ""):
    """Likelihood-level eval of a conditional adapter: exact log p(h|z) via ODE, PMI per pair, hedging/edit test vs the MSE critic."""
    import subprocess
    from playground_app import resolve_base
    base = resolve_base("Qwen/Qwen3.6-27B", local_snapshot=True)
    out = f"/vol_glp/cond/{tag}/eval_cond.json"
    cmd = [sys.executable, "-m", "nla.flow.eval_cond", "--prior", f"/vol_glp/{prior_tag}/ckpts/{prior_ckpt}", "--adapter", f"/vol_glp/cond/{tag}/adapter_latest.pt",
           "--stats", f"/vol_glp/{prior_tag}/rep_statistics.pt", "--base", base, "--val-parquet", "/vol_q36/data/sft/av_sft_val.parquet",
           "--critic", "/vol/ckpts/qwen36_27b/ar_sft_merged", "--out", out] + extra.split()
    rc = subprocess.call(cmd, cwd=REPO_REMOTE); vol_glp.commit(); return rc


@app.function(gpu="B200:8", timeout=23 * 3600, volumes=VOLS, secrets=SECRETS, cpu=64, memory=1024 * 1024, ephemeral_disk=600 * 1024)
def train_cond_ddp(tag: str, prior_tag: str = "glp27b_main", prior_ckpt: str = "snap_000655M", extra: str = "", nproc: int = 8):
    """Stage 2, replicated-adapter DDP over 8 B200 (torchrun; frozen prior; adapter + encoder LoRA grads all-reduced every step; rank-disjoint data)."""
    return _ddp(tag, prior_tag, prior_ckpt, extra, nproc)


@app.function(gpu="B200:2", timeout=3 * 3600, volumes=VOLS, secrets=SECRETS, cpu=32, memory=512 * 1024, ephemeral_disk=600 * 1024)
def train_cond_ddp2(tag: str, prior_tag: str = "glp27b_main", prior_ckpt: str = "snap_000655M", extra: str = ""):
    """2-rank smoke of train_cond_ddp (volume commits every 60 s)"""
    return _ddp(tag, prior_tag, prior_ckpt, extra, 2, commit_every=60)


def _ddp(tag, prior_tag, prior_ckpt, extra, nproc, commit_every=600):
    import subprocess
    from playground_app import resolve_base
    base = resolve_base("Qwen/Qwen3.6-27B", local_snapshot=True)
    out = f"/vol_glp/cond/{tag}"
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}", "-m", "nla.flow.train_cond", "--prior", f"/vol_glp/{prior_tag}/ckpts/{prior_ckpt}",
           "--stats", f"/vol_glp/{prior_tag}/rep_statistics.pt", "--base", base, "--train-parquet", "/vol_q36/data/sft/av_sft_train.parquet", "--val-parquet", "/vol_q36/data/sft/av_sft_val.parquet",
           "--out", out, "--tag", tag, "--mined-acts-parquet", "/vol_q36/data/rl/rl_shuf.parquet"] + extra.split()
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    os.makedirs(out, exist_ok=True); logf = open(os.path.join(out, "train.log"), "ab")
    import time as _t
    logf.write(f"[modal] container started {_t.strftime('%Y-%m-%d %H:%M:%S UTC', _t.gmtime())}, {nproc} ranks\n".encode()); logf.flush(); vol_glp.commit()   # watchers: started = train.log exists
    proc = subprocess.Popen(cmd, cwd=REPO_REMOTE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stop = _commit_loop(commit_every)                                            # snapshots / train.log visible to other containers mid-run
    for line in proc.stdout: sys.stdout.buffer.write(line); sys.stdout.flush(); logf.write(line); logf.flush()
    rc = proc.wait(); logf.close(); stop.set(); vol_glp.commit(); return rc


def _commit_loop(every):
    """background vol_glp.commit() every `every` seconds until the returned Event is set"""
    import threading
    ev = threading.Event()
    def loop():
        while not ev.wait(every):
            try: vol_glp.commit()
            except Exception as e: print(f"[modal] periodic commit failed: {str(e)[:120]}", flush=True)
    threading.Thread(target=loop, daemon=True).start(); return ev


@app.function(gpu="B200:4", timeout=23 * 3600, volumes=VOLS, secrets=SECRETS, cpu=32, memory=768 * 1024, ephemeral_disk=600 * 1024)
def train_cond_ddp4(tag: str, prior_tag: str = "glp27b_main", prior_ckpt: str = "snap_000655M", extra: str = ""):
    """train_cond_ddp on 4 B200 (fallback when an 8-GPU container stays pending; pass --batch x2 or --grad-accum 2 for the same global batch)"""
    return _ddp(tag, prior_tag, prior_ckpt, extra, 4)


@app.function(gpu="B200:8", timeout=23 * 3600, volumes=VOLS, secrets=SECRETS, cpu=64, memory=1024 * 1024, ephemeral_disk=600 * 1024)
def train_cond_g8(tag: str, prior_tag: str = "glp27b_main", prior_ckpt: str = "snap_000655M", extra: str = ""):
    """from-scratch / co-trained prior over 8 B200 (FSDP2 via --unfreeze-prior, torchrun); same launcher as train_cond_ddp with the prior unfrozen"""
    return _ddp(tag, prior_tag, prior_ckpt, "--unfreeze-prior " + extra, 8)


@app.function(gpu="B200:4", timeout=23 * 3600, **COMMON)
def train_cond_g4(tag: str, prior_tag: str = "glp27b_main", prior_ckpt: str = "snap_000655M", extra: str = ""):
    """Stage 2 with the prior CO-TRAINED (FSDP2 over 4 GPUs, torchrun); each rank holds its own encoder copy."""
    import subprocess
    from playground_app import resolve_base
    base = resolve_base("Qwen/Qwen3.6-27B", local_snapshot=True)
    out = f"/vol_glp/cond/{tag}"
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4", "-m", "nla.flow.train_cond", "--prior", f"/vol_glp/{prior_tag}/ckpts/{prior_ckpt}",
           "--stats", f"/vol_glp/{prior_tag}/rep_statistics.pt", "--base", base, "--train-parquet", "/vol_q36/data/sft/av_sft_train.parquet", "--val-parquet", "/vol_q36/data/sft/av_sft_val.parquet",
           "--out", out, "--tag", tag, "--mined-acts-parquet", "/vol_q36/data/rl/rl_shuf.parquet", "--unfreeze-prior"] + extra.split()
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    os.makedirs(out, exist_ok=True); logf = open(os.path.join(out, "train.log"), "ab")            # durable log on the volume (detached runs lose their stdout)
    proc = subprocess.Popen(cmd, cwd=REPO_REMOTE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stop = _commit_loop(600)                                                     # train.log / checkpoints visible mid-run
    for line in proc.stdout: sys.stdout.buffer.write(line); sys.stdout.flush(); logf.write(line); logf.flush()
    rc = proc.wait(); logf.close(); stop.set(); vol_glp.commit(); return rc


image_claims = image_base.pip_install("spacy==3.8.*", "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl").add_local_dir(
    REPO_LOCAL, REPO_REMOTE, copy=False, ignore=REPO_IGNORE)


def _gather(calls):
    """wait for every spawned call; a failed call is reported, not raised (raising would stop the app and cancel the others)"""
    out = []
    for c in calls:
        try: out.append(c.get())
        except Exception as e: out.append(f"ERR {type(e).__name__}: {str(e)[:120]}")
    return out


def _claims(args, commit=True):
    """python scripts/<args> from the repo, streamed; commits nla-glp so the next stage sees the files"""
    import subprocess
    cmd = [sys.executable] + args; print("[modal] " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=REPO_REMOTE)
    if commit: vol_glp.commit()
    return rc


@app.function(timeout=8 * 3600, volumes=VOLS, secrets=SECRETS, cpu=4, memory=16 * 1024)
def claims_docs(source: str, n_docs: int, root: str, tag: str = "v1", slice_: str = "0/1", extra: str = ""):
    """synthetic claims: stream one slice of one source of the corpus mix -> {root}/docs (scripts/claims_extract.py docs)"""
    vol_glp.reload()
    return _claims([f"{REPO_REMOTE}/scripts/claims_extract.py", "docs", "--source", source, "--n-docs", str(n_docs), "--root", root, "--tag", tag, "--slice", slice_] + extra.split())


@app.function(gpu="B200", timeout=8 * 3600, **COMMON)
def claims_anchors(root: str, shard: int = 0, nshards: int = 1, tag: str = "v1", extra: str = ""):
    """synthetic claims: anchors + Qwen3.6-27B state (L42 residual, top-10, entropy, J-lens top-20, vLLM greedy 16) + family-1 claims"""
    from modal_nla_exp import _prep
    from playground_app import resolve_base
    os.environ.update({"NLA_VLLM_EAGER": "1", "VLLM_ATTENTION_BACKEND": "FLASH_ATTN"}); _prep(patch_lens=True)
    vol_glp.reload(); base = resolve_base("Qwen/Qwen3.6-27B", local_snapshot=True)
    return _claims([f"{REPO_REMOTE}/scripts/claims_extract.py", "anchors", "--root", root, "--shard", str(shard), "--nshards", str(nshards), "--tag", tag, "--base", base] + extra.split())


@app.function(image=image_claims, timeout=6 * 3600, volumes=VOLS, secrets=SECRETS, cpu=32, memory=64 * 1024)
def claims_text(root: str, shard: int = 0, nshards: int = 1, extra: str = "", finalize_name: str = ""):
    """synthetic claims: family 2 (text-grounded, rule-based + spaCy) -> {root}/claims/text_*.parquet; finalize_name: then finalize that one
    streaming shard on CPU (one-claim shards: --min-claims 1)"""
    vol_glp.reload()
    rc = _claims([f"{REPO_REMOTE}/scripts/claims_text.py", "--root", root, "--shard", str(shard), "--nshards", str(nshards), "--procs", "30"] + extra.split())
    if finalize_name:
        if not finalize_name.startswith("v1_"):   # one-claim shards: rebuild family-1 claims first (write_internal dropped training anchors before e38e932+1)
            rc = rc or _claims([f"{REPO_REMOTE}/scripts/claims_extract.py", "internal", "--root", root, "--names", finalize_name])
        rc = rc or _claims([f"{REPO_REMOTE}/scripts/claims_finalize.py", "--root", root, "--names", finalize_name, "--min-claims", "1", "--stats-tag", finalize_name])
    return rc


@app.function(gpu="B200", timeout=8 * 3600, max_containers=12, **COMMON)
def claims_anchors_v2(root: str, shard: int, nshards: int, tag: str = "v2", extra: str = ""):
    """streaming one-claim extraction: anchors + state for docs shard i of n (at most 14 containers at once: + 8 training + 2 baselines = 24 B200), then (non-blocking) the text
    claims + per-shard finalize of this shard on a CPU container, so training can consume shards as they land"""
    from modal_nla_exp import _prep
    from playground_app import resolve_base
    os.environ.update({"NLA_VLLM_EAGER": "1", "VLLM_ATTENTION_BACKEND": "FLASH_ATTN"}); _prep(patch_lens=True)
    vol_glp.reload(); name = f"{tag}_{shard:03d}"
    if os.path.exists(f"{root}/anchors/anchors_{name}.parquet"):
        if not os.path.exists(f"{root}/claims/text_{name}.parquet"): claims_text.spawn(root, 0, 1, f"--names {name}", "")
        return 0                                                                   # resumable: finished shards are skipped
    base = resolve_base("Qwen/Qwen3.6-27B", local_snapshot=True)
    rc = _claims([f"{REPO_REMOTE}/scripts/claims_extract.py", "anchors", "--root", root, "--shard", str(shard), "--nshards", str(nshards), "--tag", tag, "--base", base,
                  "--one-claim", "--min-anchors", "2", "--max-anchors", "3", "--docs-glob", f"docs_*_{tag}_*.parquet"] + extra.split())
    if rc == 0: claims_text.spawn(root, 0, 1, f"--names {name}", "")                  # text claims now; finalize after the Gemma semantic claims
    return rc


@app.function(gpu="B200", timeout=8 * 3600, **COMMON)
def claims_compose(adapter: str, tag: str, extra: str = ""):
    """stage-1 eval: joint claim-set condition vs velocity composition (w = 1, 1/m) vs sum of single-claim PMIs, exact bits"""
    vol_glp.reload()
    return _claims([f"{REPO_REMOTE}/scripts/claims_compose_eval.py", "--adapter", adapter, "--tag", tag] + extra.split())


@app.function(gpu="B200", timeout=4 * 3600, **COMMON)
def claims_gates(adapter: str, tag: str, extra: str = ""):
    """stage-1 gates on the fixed stage-0 benchmark (paired detection, single-claim PMI, greedy frontier, set vs best single)"""
    vol_glp.reload()
    return _claims([f"{REPO_REMOTE}/scripts/claims_gates.py", "--adapter", adapter, "--tag", tag] + extra.split())


@app.function(gpu="B200", timeout=4 * 3600, **COMMON)
def claims_compose_variants(adapter: str, tag: str, extra: str = ""):
    """eval only: velocity-composition weights (mean / sum / w(t) linear / hard switch) and singles-minus-LM-redundancy on the 120-row benchmark"""
    vol_glp.reload()
    return _claims([f"{REPO_REMOTE}/scripts/claims_compose_variants.py", "--adapter", adapter, "--tag", tag] + extra.split())


@app.function(gpu="B200", timeout=3 * 3600, **COMMON)
def fit_whiten_unitnorm(extra: str = ""):
    """ZCA whitening on unit-norm L42 activations -> /vol_glp/whiten/l42_zca_unitnorm.pt (scripts/fit_whitening_unitnorm.py)"""
    vol_glp.reload()
    return _claims([f"{REPO_REMOTE}/scripts/fit_whitening_unitnorm.py"] + extra.split())


@app.function(gpu="B200", timeout=4 * 3600, **COMMON)
def claims_prebank(adapter: str, tag: str, extra: str = ""):
    """double-difference twin control against same-document activations before the detail's first mention (scripts/claims_prebank.py)"""
    vol_glp.reload()
    return _claims([f"{REPO_REMOTE}/scripts/claims_prebank.py", "--adapter", adapter, "--tag", tag] + extra.split())


@app.function(gpu="B200", timeout=4 * 3600, **COMMON)
def claims_redundancy(adapter: str, tag: str, extra: str = ""):
    """redundancy terms of the claims reward compared through the RL scorer (scripts/claims_redundancy_eval.py)"""
    vol_glp.reload()
    return _claims([f"{REPO_REMOTE}/scripts/claims_redundancy_eval.py", "--adapter", adapter, "--tag", tag] + extra.split())


@app.function(gpu="B200", timeout=4 * 3600, **COMMON)
def claims_gate_rl(adapter: str, tag: str, extra: str = ""):
    """120-row gates through the RL reward (FlowCritic.score_claims_composed singles_red; scripts/claims_gate_rl.py)"""
    vol_glp.reload()
    return _claims([f"{REPO_REMOTE}/scripts/claims_gate_rl.py", "--adapter", adapter, "--tag", tag] + extra.split())


@app.function(gpu="B200:2", timeout=2 * 3600, **COMMON)
def test_claims_reward(adapter: str, extra: str = ""):
    """GPU equivalence test: RL singles_red reward == the composition eval's singles_red (scripts/test_claims_reward_equiv.py; 2 GPUs)"""
    vol_glp.reload()
    return _claims([f"{REPO_REMOTE}/scripts/test_claims_reward_equiv.py", "--adapter", adapter] + extra.split())


@app.function(gpu="B200", timeout=4 * 3600, **COMMON)
def claims_g2eval(adapter: str, tag: str, extra: str = ""):
    """programmatic-twin benchmark + specificity-ladder calibration on the peer session's g2 positions (read only; scripts/claims_g2eval.py)"""
    vol_glp.reload()
    return _claims([f"{REPO_REMOTE}/scripts/claims_g2eval.py", "--adapter", adapter, "--tag", tag] + extra.split())


@app.function(gpu="B200", timeout=2 * 3600, **COMMON)
def claims_hubness(adapter: str, tag: str, extra: str = ""):
    """claim->activation hubness vs activation norm on same-template retrieval matrices (scripts/claims_hubness.py)"""
    vol_glp.reload()
    return _claims([f"{REPO_REMOTE}/scripts/claims_hubness.py", "--adapter", adapter, "--tag", tag] + extra.split())


@app.function(gpu="B200", timeout=4 * 3600, **COMMON)
def claims_controls(adapter: str, tag: str, extra: str = ""):
    """twin detection with a wrong-activation control, claim-only LM baseline, same-template retrieval (N = 16/64/256)"""
    vol_glp.reload()
    return _claims([f"{REPO_REMOTE}/scripts/claims_controls.py", "--adapter", adapter, "--tag", tag] + extra.split())


@app.function(gpu="B200", timeout=8 * 3600, max_containers=4, **COMMON)
def ws_score(adapter: str, critic_tag: str, parts: str):
    """verbalizer warm start: single-claim PMI of candidate bullets under a critic (scripts/claims_warmstart.py score)"""
    vol_glp.reload()
    return _claims([f"{REPO_REMOTE}/scripts/claims_warmstart.py", "score", "--adapter", adapter, "--critic-tag", critic_tag, "--parts", parts])


@app.function(gpu="B200", timeout=12 * 3600, max_containers=8, **COMMON)
def ws_margin(adapter: str, critic_tag: str, parts: str, extra: str = ""):
    """verbalizer warm start v2: contrastive margin of candidate bullets (own activation vs K same-template activations of other documents)"""
    vol_glp.reload()
    return _claims([f"{REPO_REMOTE}/scripts/claims_warmstart.py", "score_margin", "--adapter", adapter, "--critic-tag", critic_tag, "--parts", parts] + extra.split())


@app.function(timeout=2 * 3600, volumes=VOLS, secrets=SECRETS, cpu=8, memory=128 * 1024)
def ws_sft_subset(src: str, dst: str, n_train: int = 20000, n_val: int = 512):
    """random subset of a warm-start SFT set (nla-glp) -> nla-exp:<dst>/av_sft_{train,test}.parquet + sidecars, readable by modal_nla_exp.py --task sft"""
    import shutil, pyarrow as pa, pyarrow.parquet as pq
    vol_glp.reload(); os.makedirs(dst, exist_ok=True)
    cols = ["prompt", "response", "activation_vector", "activation_layer", "doc_id", "source"]
    for split, n, name in (("train", n_train, "av_sft_train"), ("val", n_val, "av_sft_test")):   # rows are pre-shuffled by the build: stream the first n
        pf = pq.ParquetFile(f"{src}/{split}.parquet"); tot = pf.metadata.num_rows; got = 0; w = None
        for rb in pf.iter_batches(batch_size=2048, columns=cols):
            tb = pa.Table.from_batches([rb]).slice(0, n - got)
            tb = tb.set_column(tb.schema.get_field_index("activation_vector"), "activation_vector", tb.column("activation_vector").cast(pa.list_(pa.float32(), 5120)))
            if w is None: w = pq.ParquetWriter(f"{dst}/{name}.parquet", tb.schema, compression="zstd")
            w.write_table(tb); got += tb.num_rows
            if got >= n: break
        if w is not None: w.close()
        shutil.copy2("/vol_q36/data/sft/av_sft_train.parquet.nla_meta.yaml", f"{dst}/{name}.parquet.nla_meta.yaml")
        print(f"[ws-sft-subset] {split}: {got} of {tot} rows -> {dst}/{name}.parquet", flush=True)
    vol_exp.commit(); return 0


@app.function(timeout=2 * 3600, volumes=VOLS, secrets=SECRETS, cpu=8, memory=128 * 1024)
def ws_margin_stats(critic_tag: str):
    """margin distributions by family / type + kept counts per threshold -> scores_margin_<critic>/margin_stats.json"""
    vol_glp.reload()
    return _claims([f"{REPO_REMOTE}/scripts/claims_warmstart.py", "margin_stats", "--critic-tag", critic_tag])


@app.function(timeout=4 * 3600, volumes=VOLS, secrets=SECRETS, cpu=16, memory=256 * 1024)
def ws_build(critic_tag: str, extra: str = ""):
    """verbalizer warm start: critic-filtered bullet-list SFT sets (scripts/claims_warmstart.py build)"""
    vol_glp.reload()
    return _claims([f"{REPO_REMOTE}/scripts/claims_warmstart.py", "build", "--critic-tag", critic_tag] + extra.split())


@app.function(gpu="B200", timeout=6 * 3600, **COMMON)
def claims_finalize(root: str, extra: str = ""):
    """synthetic claims: merge the three families per anchor, near-duplicate removal (sentence embeddings), stats -> {root}/final"""
    vol_glp.reload()
    return _claims([f"{REPO_REMOTE}/scripts/claims_finalize.py", "--root", root] + extra.split())


@app.local_entrypoint()
def main(task: str = "smoke", tag: str = "", config: str = "", sets: str = "", ckpt: str = "final", extra: str = "", n_prompts: int = 300000, attn_impl: str = "sdpa", tokens_per_batch: int = 32768, prior_tag: str = "glp27b_main", av_merged: str = "/vol/ckpts/qwen36_27b/av_sft_merged", nshards: int = 8, root: str = "/vol_glp/claims", n_docs: int = 0):
    """--sets "train.lr=1e-4 model.n_layers=12" ; --config configs/glp/<override>.yaml"""
    if task == "smoke":
        print("rc", smoke.remote(tag or "smoke_glp", sets))
    elif task == "pretrain":
        assert tag, "--tag required"
        print("rc", pretrain.remote(tag, config, sets))
    elif task == "pretrain_g4":
        assert tag, "--tag required"
        print("rc", pretrain_g4.remote(tag, config, sets))
    elif task == "eval_lm":
        print("rc", eval_lm.remote(tag, ckpt, extra))
    elif task == "bench_producer":
        print("rc", bench_producer.remote(attn_impl, tokens_per_batch, extra=extra))
    elif task == "gumbel_av":
        print("rc", gumbel_av.remote(tag or "gumbel_av", extra))
    elif task == "mine_pairs":   # all shards in parallel, one B200 each
        calls = [mine_pairs.spawn(tag or "sft_av", av_merged, shard=i, nshards=nshards, extra=extra) for i in range(nshards)]
        print("rc", _gather(calls))
    elif task == "eval_cond":
        print("rc", eval_cond.remote(tag, prior_tag, ckpt, extra))
    elif task == "ultra_extract":   # all shards in parallel, one B200 each; --tag = explained dir name, --sets = out dir
        calls = [ultra_extract.spawn(f"/vol_glp/data/{tag or 'ultra_explained_t1'}/chunk_*.parquet", sets or "/vol_glp/data/ultra_L42", shard=i, nshards=nshards, extra=extra) for i in range(nshards)]
        print("rc", _gather(calls))
    elif task == "train_cond_g8":   # 8-GPU FSDP (--unfreeze-prior added)
        print("rc", train_cond_g8.remote(tag or "cond_g8", prior_tag, ckpt, extra))
    elif task == "train_cond_g4":
        print("rc", train_cond_g4.remote(tag or "cond_cotrain", prior_tag, ckpt, extra))
    elif task == "train_cond_ddp4":
        print("rc", train_cond_ddp4.remote(tag or "cond_ddp4", prior_tag, ckpt, extra))
    elif task == "train_cond_ddp":   # 8 ranks on B200:8 (--nshards 2 -> the B200:2 smoke function)
        f_ = train_cond_ddp2 if nshards == 2 else train_cond_ddp
        print("rc", (f_.remote(tag or "cond_ddp", prior_tag, ckpt, extra) if nshards == 2 else f_.remote(tag or "cond_ddp", prior_tag, ckpt, extra, nproc=min(nshards, 8))))
    elif task == "train_cond":
        print("rc", train_cond.remote(tag or "cond_smoke", prior_tag, ckpt, extra))
    elif task == "sample_diag":
        print("rc", sample_diag.remote(tag, ckpt, extra))
    elif task == "bnoise":
        print("rc", bnoise.remote(tag, ckpt, extra))
    elif task == "claims_docs":   # one CPU container per source, in parallel; --n-docs = total docs over the mix
        K = {"ffw": 22, "code": 3, "chat": 4, "math": 3, "fiction": 6, "multi": 8} if n_docs >= 50000 else {}   # parallel slices per source at scale
        if n_docs >= 1000000: K = {"ffw": 66, "code": 8, "chat": 8, "math": 8, "fiction": 12, "multi": 24}
        todo = [(s_, i) for s_ in ("ffw", "code", "chat", "math", "fiction", "multi") for i in range(K.get(s_, 1))]   # = claims_extract.SOURCES
        if sets: todo = [(x.split(":")[0], int(x.split(":")[1])) for x in sets.split(",")]          # --sets "ffw:3,ffw:17": re-run only these slices
        calls = [claims_docs.spawn(s_, n_docs, root, tag or "v1", f"{i}/{K.get(s_, 1)}", extra) for s_, i in todo]
        print("rc", _gather(calls))
    elif task == "claims_anchors":   # all shards in parallel, one B200 each
        calls = [claims_anchors.spawn(root, i, nshards, tag or "v1", extra) for i in range(nshards)]
        print("rc", _gather(calls))
    elif task == "claims_anchors_v2":   # one call per shard, queued; Modal runs <= 14 at a time; returns when all are done
        calls = [claims_anchors_v2.spawn(root, i, nshards, tag or "v2", extra) for i in range(nshards)]
        print("rc", _gather(calls))
    elif task == "claims_finalize_shard":   # CPU finalize of one streaming shard (--tag = shard name, e.g. v2_017)
        old_share = tag.startswith("v2_") and int(tag[3:]) <= 22    # extracted under the .4/.4/.2 family shares: redo its text claims under the current draw
        print("rc", claims_text.remote(root, 0, 1, f"--names {tag}" if old_share else "--names __none__", tag))
    elif task == "claims_text":
        calls = [claims_text.spawn(root, i, nshards, extra) for i in range(nshards)]
        print("rc", _gather(calls))
    elif task == "claims_compose":   # --ckpt = adapter path, --tag = output tag
        print("rc", claims_compose.remote(ckpt, tag, extra))
    elif task == "claims_gates":   # --ckpt = adapter path, --tag = output tag
        print("rc", claims_gates.remote(ckpt, tag, extra))
    elif task == "claims_compose_variants":   # --ckpt = adapter path, --tag = output tag
        print("rc", claims_compose_variants.remote(ckpt, tag, extra))
    elif task == "fit_whiten_unitnorm":
        print("rc", fit_whiten_unitnorm.remote(extra))
    elif task == "claims_prebank":
        print("rc", claims_prebank.remote(ckpt, tag, extra))
    elif task == "claims_redundancy":
        print("rc", claims_redundancy.remote(ckpt, tag, extra))
    elif task == "claims_gate_rl":
        print("rc", claims_gate_rl.remote(ckpt, tag, extra))
    elif task == "test_claims_reward":
        print("rc", test_claims_reward.remote(ckpt, extra))
    elif task == "claims_g2eval":
        print("rc", claims_g2eval.remote(ckpt, tag, extra))
    elif task == "claims_hubness":
        print("rc", claims_hubness.remote(ckpt, tag, extra))
    elif task == "claims_controls":   # --ckpt = adapter path, --tag = output tag
        print("rc", claims_controls.remote(ckpt, tag, extra))
    elif task == "ws_score":   # --ckpt adapter, --tag critic tag, --sets comma list of parts (gold:<shard> / syn:<text shard>), split over --nshards containers
        ps = [x for x in sets.split(",") if x]; k = min(nshards, len(ps))
        print("rc", _gather([ws_score.spawn(ckpt, tag, ",".join(ps[i::k])) for i in range(k)]))
    elif task == "ws_margin":   # --ckpt adapter, --tag critic tag, --sets parts, --nshards containers (<= 8), --extra e.g. "--K 64"
        ps = [x for x in sets.split(",") if x]; k = min(nshards, len(ps), 8)
        print("rc", _gather([ws_margin.spawn(ckpt, tag, ",".join(ps[i::k]), extra) for i in range(k)]))
    elif task == "ws_sft_subset":   # --sets <src dir on nla-glp>, --root <dst dir on nla-exp>, --extra "<n_train> <n_val>"
        xs = extra.split(); print("rc", ws_sft_subset.remote(sets, root, int(xs[0]) if xs else 20000, int(xs[1]) if len(xs) > 1 else 512))
    elif task == "ws_margin_stats":
        print("rc", ws_margin_stats.remote(tag))
    elif task == "ws_build":
        print("rc", ws_build.remote(tag, extra))
    elif task == "claims_finalize":
        print("rc", claims_finalize.remote(root, extra))
    elif task == "gen_onpolicy":
        print("rc", gen_onpolicy.remote(n_prompts, extra))
