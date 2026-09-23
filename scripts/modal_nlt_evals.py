"""Modal app `nlt-evals` (volume `nlt`; owner redteam): GPU pieces of the eval suite.

  modal run --detach scripts/modal_nlt_evals.py --task manifest --extra "--ckpt /vol/critic/text_v0/ckpt_latest.pt --manifest /vol/evals/manifest_lensL1.parquet --out /vol/evals/scored_lensL1.parquet"
  modal run --detach scripts/modal_nlt_evals.py --task causal   --extra "--out /vol/evals/causal_val.parquet --n-pairs 4096"
  modal run          scripts/modal_nlt_evals.py --task text     --extra "--z /vol/z/lensdiff_v1/val/L1.parquet --out /vol/evals/text_lensL1.json"   (CPU)
  modal run          scripts/modal_nlt_evals.py --task build-manifest --extra "--z /vol/z/lensdiff_v1/val/L1.parquet --out /vol/evals/manifest_lensL1.parquet"  (CPU)
  modal run          scripts/modal_nlt_evals.py --task cat --path evals/scored_lensL1.parquet.summary.json
Data dir default /vol/data/qwen3_8b (infra). HF: shared cache /vol/hf_cache, xet disabled (429s).
"""
import os, sys
import modal
for _p in (os.path.dirname(os.path.abspath(__file__)),):
    if _p not in sys.path: sys.path.insert(0, _p)
from modal_nla_exp import image_base, REPO_LOCAL, REPO_REMOTE, REPO_IGNORE  # noqa: E402

vol = modal.Volume.from_name("nlt", create_if_missing=True)
SECRETS = [modal.Secret.from_name("nla-exp-secrets")]
DATA = os.environ.get("NLT_DATA", "/vol/data/qwen3_8b")
IGNORE = list(REPO_IGNORE) + ["*.log", "**/logs/**", "logs", "*.npy", "*.pt", "*.jsonl", "*.out", "wandb"]
image = image_base.env({"HF_HOME": "/vol/hf_cache", "HF_HUB_DISABLE_XET": "1"}).add_local_dir(REPO_LOCAL, REPO_REMOTE, copy=False, ignore=IGNORE)
app = modal.App(os.environ.get("NLT_APP", "nlt-evals"), image=image)
GPU = os.environ.get("NLT_GPU", "B200")
COMMON = dict(volumes={"/vol": vol}, secrets=SECRETS, cpu=16, memory=128 * 1024, ephemeral_disk=512 * 1024)


def _run(cmd):
    import subprocess
    os.makedirs("/vol/evals", exist_ok=True); vol.reload()
    print("[modal] " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=REPO_REMOTE); vol.commit(); return rc


@app.function(gpu=GPU, timeout=6 * 3600, **COMMON)
def manifest(extra: str = "", data: str = DATA):
    return _run([sys.executable, "-m", "nlt.evals.score_manifest", "--data-dir", data] + extra.split())


@app.function(gpu=GPU, timeout=6 * 3600, **COMMON)
def causal(extra: str = "", data: str = DATA):
    return _run([sys.executable, "-m", "nlt.evals.causal", "--data-dir", data] + extra.split())


@app.function(timeout=2 * 3600, volumes={"/vol": vol}, secrets=SECRETS, cpu=8, memory=32 * 1024)
def text(extra: str = "", data: str = DATA):
    import glob
    meta = sorted(glob.glob(f"{data}/val/meta_*.parquet")); docs = sorted(glob.glob(f"{data}/val/docs_*.parquet"))
    cmd = [sys.executable, "-m", "nlt.evals.run_text_evals", "--pairs", f"{data}/pairs_val.parquet"] + extra.split()
    if meta and docs and "--meta" not in extra: cmd += ["--meta", meta[0], "--docs", docs[0]]       # TODO multi-shard: concatenate
    return _run(cmd)


@app.function(timeout=2 * 3600, volumes={"/vol": vol}, secrets=SECRETS, cpu=8, memory=32 * 1024)
def build_manifest(extra: str = "", data: str = DATA):
    import glob
    meta = sorted(glob.glob(f"{data}/val/meta_*.parquet")); docs = sorted(glob.glob(f"{data}/val/docs_*.parquet"))
    cmd = [sys.executable, "-m", "nlt.evals.controls", "--pairs", f"{data}/pairs_val.parquet"] + extra.split()
    if meta and docs and "--meta" not in extra: cmd += ["--meta", meta[0], "--docs", docs[0]]
    return _run(cmd)


@app.function(timeout=1800, volumes={"/vol": vol}, cpu=2, memory=8 * 1024)
def cat(path: str):
    vol.reload(); print(open(f"/vol/{path}").read())


@app.function(timeout=1800, volumes={"/vol": vol}, cpu=2, memory=8 * 1024)
def ls(path: str = ""):
    import subprocess; vol.reload(); print(subprocess.run(["ls", "-la", f"/vol/{path}"], capture_output=True, text=True).stdout)


@app.local_entrypoint()
def main(task: str = "text", extra: str = "", data: str = DATA, path: str = ""):
    if task == "manifest": print("rc", manifest.remote(extra, data))
    elif task == "causal": print("rc", causal.remote(extra, data))
    elif task == "text": print("rc", text.remote(extra, data))
    elif task == "build-manifest": print("rc", build_manifest.remote(extra, data))
    elif task == "cat": cat.remote(path)
    elif task == "ls": ls.remote(path)
    else: raise SystemExit(task)
    print("done.")
