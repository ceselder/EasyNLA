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
           "--base", base, "--train-parquet", "/vol_q36/data/sft/av_sft_train.parquet", "--val-parquet", "/vol_q36/data/sft/av_sft_val.parquet", "--out", out, "--tag", tag] + extra.split()
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


@app.function(timeout=2 * 3600, volumes=VOLS, secrets=SECRETS, cpu=16, memory=200 * 1024, ephemeral_disk=300 * 1024)
def convert_wrapper(src: str, dst: str):
    """Text-only merged checkpoint -> vLLM wrapper layout (Qwen3_5ForConditionalGeneration)."""
    import subprocess
    from playground_app import resolve_base
    base = resolve_base("Qwen/Qwen3.6-27B", local_snapshot=True)
    rc = subprocess.call([sys.executable, f"{REPO_REMOTE}/scripts/convert_textonly_to_wrapper.py", "--src", src, "--dst", dst, "--base-config", f"{base}/config.json"], cwd=REPO_REMOTE)
    vol_exp.commit(); return rc


@app.local_entrypoint()
def main(task: str = "smoke", tag: str = "", config: str = "", sets: str = "", ckpt: str = "final", extra: str = "", n_prompts: int = 300000, attn_impl: str = "sdpa", tokens_per_batch: int = 32768, prior_tag: str = "glp27b_main", av_merged: str = "/vol/ckpts/qwen36_27b/av_sft_merged", nshards: int = 8):
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
        print("rc", [c.get() for c in calls])
    elif task == "train_cond":
        print("rc", train_cond.remote(tag or "cond_smoke", prior_tag, ckpt, extra))
    elif task == "sample_diag":
        print("rc", sample_diag.remote(tag, ckpt, extra))
    elif task == "bnoise":
        print("rc", bnoise.remote(tag, ckpt, extra))
    elif task == "gen_onpolicy":
        print("rc", gen_onpolicy.remote(n_prompts, extra))
