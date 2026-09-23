"""Modal app `nlt-trunk` (volume `nlt`): the TRUNK critic (nlt/trunk): training, exact bits, manifests, speed test.

  NLT_GPU=H100 modal run --detach scripts/modal_nlt_trunk.py --task train --tag trunk_smoke --extra "--prior /vol/critic/none_v1_pooled/ckpt_final.pt --text-parquet ... --steps 1500"
  NLT_GPU=H100 modal run --detach scripts/modal_nlt_trunk.py --task bits  --tag trunk_smoke --extra "--ckpt /vol/trunk/trunk_smoke/ckpt_final.pt --text-parquet ... --n 512"
  modal run scripts/modal_nlt_trunk.py --task script --path nlt/trunk/speed.py --extra "..."
Outputs: /vol/trunk/<tag>/ (ckpt_*.pt, eval_latest.json), /vol/results/bits_<tag>.json. Wandb octahedral-systems/nlt-qwen3-8b. HF cache /vol/hf_cache (Qwen3-8B is there).
"""
import os, sys
import modal
for _p in (os.path.dirname(os.path.abspath(__file__)), os.path.join(os.environ.get("PYTHONPATH", "/root/easyNLA").split(":")[0], "scripts")):
    if os.path.isdir(_p) and _p not in sys.path: sys.path.insert(0, _p)
from modal_nla_exp import image_base, REPO_LOCAL, REPO_REMOTE, REPO_IGNORE  # noqa: E402

vol = modal.Volume.from_name("nlt", create_if_missing=True)
SECRETS = [modal.Secret.from_name("nla-exp-secrets")]
DATA = os.environ.get("NLT_DATA", "/vol/data/qwen3_8b")
IGNORE = list(REPO_IGNORE) + ["*.log", "**/logs/**", "logs", "*.npy", "*.pt", "*.jsonl", "*.out", "wandb"]
image = image_base.add_local_dir(REPO_LOCAL, REPO_REMOTE, copy=False, ignore=IGNORE)
app = modal.App(os.environ.get("NLT_APP", "nlt-trunk"), image=image)
GPU = os.environ.get("NLT_GPU", "H100")
COMMON = dict(volumes={"/vol": vol}, secrets=SECRETS, cpu=16, memory=200 * 1024, ephemeral_disk=512 * 1024)


def _run(cmd):
    import subprocess
    os.environ["HF_HOME"] = "/vol/hf_cache"; os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ.setdefault("WANDB_DIR", "/root/wandb"); os.makedirs("/root/wandb", exist_ok=True)
    vol.reload()
    print("[modal] " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=REPO_REMOTE)
    vol.commit(); return rc


@app.function(gpu=GPU, timeout=23 * 3600, **COMMON)
def train(tag: str, extra: str = "", data: str = DATA):
    return _run([sys.executable, "-m", "nlt.trunk.train", "--data-dir", data, "--out", f"/vol/trunk/{tag}", "--tag", tag] + extra.split())


@app.function(gpu=GPU, timeout=23 * 3600, **COMMON)
def chain(tag: str, extra: str = "", bits_extra: str = "", data: str = DATA):
    """train, then exact bits of ckpt_final.pt in the SAME container (saves a cold start + store load); bits_extra = eval_bits args without --ckpt"""
    rc = _run([sys.executable, "-m", "nlt.trunk.train", "--data-dir", data, "--out", f"/vol/trunk/{tag}", "--tag", tag] + extra.split())
    if rc != 0: return rc
    return _run([sys.executable, "-m", "nlt.trunk.eval_bits", "--data-dir", data, "--ckpt", f"/vol/trunk/{tag}/ckpt_final.pt", "--out", f"/vol/results/bits_{tag}.json"] + bits_extra.split())


@app.function(gpu=GPU, timeout=8 * 3600, **COMMON)
def bits(tag: str, extra: str = "", data: str = DATA):
    return _run([sys.executable, "-m", "nlt.trunk.eval_bits", "--data-dir", data, "--out", f"/vol/results/bits_{tag}.json"] + extra.split())


@app.function(gpu=GPU, timeout=8 * 3600, **COMMON)
def manifest(tag: str, extra: str = "", data: str = DATA):
    """redteam control manifest -> exact log p per row with the trunk critic (nlt.trunk.score_manifest); extra carries --ckpt and --manifest"""
    return _run([sys.executable, "-m", "nlt.trunk.score_manifest", "--data-dir", data, "--out", f"/vol/evals/scored_{tag}.parquet"] + extra.split())


@app.function(gpu=GPU, timeout=8 * 3600, **COMMON)
def script(path: str, extra: str = ""):
    return _run([sys.executable, f"{REPO_REMOTE}/{path}"] + extra.split())


@app.function(timeout=1800, volumes={"/vol": vol}, cpu=2, memory=8 * 1024)
def cat(path: str):
    vol.reload(); print(open(f"/vol/{path}").read())


@app.local_entrypoint()
def main(task: str = "train", tag: str = "dev", extra: str = "", data: str = DATA, path: str = "", bits_extra: str = ""):
    if task == "train": print("rc", train.remote(tag, extra, data))
    elif task == "chain": print("rc", chain.remote(tag, extra, bits_extra, data))
    elif task == "bits": print("rc", bits.remote(tag, extra, data))
    elif task == "manifest": print("rc", manifest.remote(tag, extra, data))
    elif task == "script": print("rc", script.remote(path, extra))
    elif task == "cat": cat.remote(path)
    else: raise SystemExit(task)
    print("done.")
