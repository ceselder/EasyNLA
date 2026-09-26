"""Modal app nla-claims-gemma: Gemma-4 semantic claims for the compositional-NLA synthetic data (scripts/claims_gemma.py).
  modal run scripts/modal_claims_gemma.py --task bench                    # one B200:4 container, all GPU layouts on the same prompts
  modal run scripts/modal_claims_gemma.py --task gen --layout a_dp4 --names v1_001,v1_002,...   # B200:4 containers (<= 2), shards split between them
  modal run scripts/modal_claims_gemma.py --task gen1 --names ...        # fallback: 1-GPU containers (<= 8)
Outputs under /vol_glp/claims/gemma (own dir); reads the text shards in /vol_glp/claims/text; weights from the shared HF cache /vol_glp/hf."""
import os, sys
import modal
for _p in (os.path.dirname(os.path.abspath(__file__)), os.path.join(os.environ.get("PYTHONPATH", "/root/easyNLA").split(":")[0], "scripts")):
    if os.path.isdir(_p) and _p not in sys.path: sys.path.insert(0, _p)
from modal_nla_exp import SECRETS, REPO_LOCAL, REPO_REMOTE, REPO_IGNORE  # noqa: E402

vol_glp = modal.Volume.from_name("nla-glp")
vol_q36 = modal.Volume.from_name("nla-qwen36-ema")
image = (modal.Image.from_registry("vllm/vllm-openai:v0.29.0", setup_dockerfile_commands=["RUN ln -sf $(which python3) /usr/local/bin/python"]).entrypoint([])
         .run_commands("pip install --no-cache-dir pyarrow aiohttp 'huggingface_hub[hf_xet]'")
         .env({"HF_HOME": "/vol_glp/hf", "HF_XET_HIGH_PERFORMANCE": "1", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": REPO_REMOTE,
               "VLLM_ALLOW_INSECURE_SERIALIZATION": "1"})
         .add_local_dir(REPO_LOCAL, REPO_REMOTE, copy=False, ignore=REPO_IGNORE))
app = modal.App("nla-claims-gemma", image=image)
ROOT = "/vol_glp/claims"; OUT = f"{ROOT}/gemma"
vol_exp = modal.Volume.from_name("nla-exp")   # read-only use here: mined verbalizer samples (best-of-N distillation)
KW = dict(volumes={"/vol_glp": vol_glp, "/vol_q36": vol_q36, "/vol": vol_exp}, secrets=SECRETS, cpu=32, memory=256 * 1024)


@app.function(gpu="B200:4", timeout=3 * 3600, **KW)
def bench(layouts: str = "", n: int = 20000):
    sys.path.insert(0, f"{REPO_REMOTE}/scripts"); import claims_gemma as cg
    vol_glp.reload()
    r = cg.bench(ROOT, f"{OUT}/bench", n=n, layouts=[x for x in layouts.split(",") if x] or None); vol_glp.commit(); return r


@app.function(gpu="B200:4", timeout=12 * 3600, max_containers=2, **KW)
def gen(names: str, layout: str = "a_dp4"):
    sys.path.insert(0, f"{REPO_REMOTE}/scripts"); import claims_gemma as cg
    vol_glp.reload()
    return cg.gen(ROOT, OUT, [x for x in names.split(",") if x], layout=layout, commit=vol_glp.commit)


@app.function(gpu="B200", timeout=12 * 3600, max_containers=8, **KW)
def gen1(names: str):
    sys.path.insert(0, f"{REPO_REMOTE}/scripts"); import claims_gemma as cg
    vol_glp.reload()
    return cg.gen(ROOT, OUT, [x for x in names.split(",") if x], layout="single", commit=vol_glp.commit)


@app.function(gpu="B200:4", timeout=23 * 3600, max_containers=2, **KW)
def stream(k: int, K: int, layout: str = "a_dp4", pattern: str = "text_v*_*.jsonl.gz", exclude: str = ""):
    """streaming generator on a 4 x B200 container (see claims_gemma.gen_stream)"""
    sys.path.insert(0, f"{REPO_REMOTE}/scripts"); import claims_gemma as cg
    vol_glp.reload()
    return cg.gen_stream(ROOT, OUT, k, K, pattern=pattern, layout=layout, commit=vol_glp.commit, reload=vol_glp.reload, exclude=set(exclude.split(",")))


@app.function(gpu="B200", timeout=23 * 3600, max_containers=8, **KW)
def stream1(k: int, K: int, pattern: str = "text_v*_*.jsonl.gz", exclude: str = ""):
    """streaming generator on a 1 x B200 container (fallback when 4-GPU containers do not schedule)"""
    sys.path.insert(0, f"{REPO_REMOTE}/scripts"); import claims_gemma as cg
    vol_glp.reload()
    return cg.gen_stream(ROOT, OUT, k, K, pattern=pattern, layout="single", commit=vol_glp.commit, reload=vol_glp.reload, exclude=set(exclude.split(",")))


@app.function(gpu="B200", timeout=12 * 3600, max_containers=8, **KW)
def ws_split(files: str):
    """verbalizer warm start: Gemma splits gold Opus explanations into atomic bullets (scripts/claims_warmstart.run_split)"""
    sys.path.insert(0, f"{REPO_REMOTE}/scripts"); import claims_warmstart as cw
    vol_glp.reload()
    return cw.run_split(cw.WS, [x for x in files.split(",") if x], commit=vol_glp.commit)


@app.function(gpu="B200", timeout=12 * 3600, max_containers=8, **KW)
def ws_multi(names: str):
    """verbalizer warm start: 3 aspects x 2 quote-checked claims for every held-out synthetic anchor of the given text shards"""
    sys.path.insert(0, f"{REPO_REMOTE}/scripts"); import claims_warmstart as cw
    vol_glp.reload()
    return cw.run_multi(ROOT, cw.WS, [x for x in names.split(",") if x], commit=vol_glp.commit)


@app.function(gpu="B200", timeout=12 * 3600, max_containers=8, **KW)
def bon_verify(tag: str, samples_dir: str, shard: int, nshards: int):
    """best-of-N distillation: Gemma verification + hedged rewrites of mined verbalizer samples (scripts/bon_distill.py verify)"""
    import subprocess
    vol_glp.reload()
    rc = subprocess.call([sys.executable, f"{REPO_REMOTE}/scripts/bon_distill.py", "verify", "--tag", tag, "--samples-dir", samples_dir, "--shard", str(shard), "--nshards", str(nshards)], cwd=REPO_REMOTE)
    vol_glp.commit(); return rc


@app.local_entrypoint()
def main(task: str = "bench", layouts: str = "", n: int = 20000, names: str = "", layout: str = "a_dp4", containers: int = 2, exclude: str = ""):
    if task == "bon_verify":   # --names "<tag>|<samples dir>", --containers N
        tg, sd = names.split("|"); print("rc", [c.get() for c in [bon_verify.spawn(tg, sd, i, containers) for i in range(containers)]])
    elif task == "bench":
        print(bench.remote(layouts, n))
    elif task in ("gen", "gen1"):
        ns = [x for x in names.split(",") if x]; k = containers if task == "gen" else min(8, len(ns))
        groups = [",".join(ns[i::k]) for i in range(k) if ns[i::k]]
        f = gen if task == "gen" else gen1
        calls = [f.spawn(g, layout) if task == "gen" else f.spawn(g) for g in groups]
        out = []
        for c in calls:
            try: out.append(c.get())
            except Exception as e: out.append(f"ERR {type(e).__name__}: {str(e)[:200]}")
        print(out)
    elif task in ("ws_split", "ws_multi"):   # --names = comma list (gold shard paths for ws_split, text shard names for ws_multi), split over --containers
        ns = [x for x in names.split(",") if x]; k = min(containers, len(ns)); f = ws_split if task == "ws_split" else ws_multi
        calls = [f.spawn(",".join(ns[i::k])) for i in range(k)]
        out = []
        for c in calls:
            try: out.append(c.get())
            except Exception as e: out.append(f"ERR {type(e).__name__}: {str(e)[:200]}")
        print(out)
    elif task in ("stream", "stream1"):   # K = --containers streaming generators; returns when all have exited (idle or STOP)
        f = stream if task == "stream" else stream1
        calls = [f.spawn(k, containers, layout, exclude=exclude) if task == "stream" else f.spawn(k, containers, exclude=exclude) for k in range(containers)]
        out = []
        for c in calls:
            try: out.append(c.get())
            except Exception as e: out.append(f"ERR {type(e).__name__}: {str(e)[:200]}")
        print(out)
