"""Modal app `nlt-critic` (volume `nlt`): transcoder critic training, MSE baselines and the bits evaluation.

  modal run --detach scripts/modal_nlt_critic.py --task train --tag none_v0 --extra "--cond none --steps 5000"
  modal run --detach scripts/modal_nlt_critic.py --task train --tag depth_v0 --extra "--cond depth --steps 5000"
  modal run --detach scripts/modal_nlt_critic.py --task mse --tag mse_mlp --extra "--arch mlp"
  modal run --detach scripts/modal_nlt_critic.py --task bits --tag none_v0 --extra "--ckpts none:/vol/critic/none_v0/ckpt_latest.pt,..."
Outputs: /vol/critic/<tag>/ (ckpt_latest.pt, eval_latest.json), /vol/results/<name>.json. Wandb octahedral-systems/nlt-qwen3-8b.
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
app = modal.App(os.environ.get("NLT_APP", "nlt-critic"), image=image)
GPU = os.environ.get("NLT_GPU", "B200")
COMMON = dict(volumes={"/vol": vol}, secrets=SECRETS, cpu=16, memory=200 * 1024, ephemeral_disk=512 * 1024)


def _run(cmd):
    import subprocess
    os.environ["HF_HOME"] = "/root/hf_home"; os.makedirs("/root/hf_home", exist_ok=True)
    os.environ.setdefault("WANDB_DIR", "/root/wandb"); os.makedirs("/root/wandb", exist_ok=True)
    vol.reload()
    print("[modal] " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=REPO_REMOTE)
    vol.commit(); return rc


@app.function(gpu=GPU, timeout=23 * 3600, **COMMON)
def train(tag: str, extra: str = "", data: str = DATA):
    return _run([sys.executable, "-m", "nlt.critic.train", "--data-dir", data, "--out", f"/vol/critic/{tag}", "--tag", tag] + extra.split())


@app.function(gpu=GPU, timeout=6 * 3600, **COMMON)
def mse(tag: str, extra: str = "", data: str = DATA):
    return _run([sys.executable, "-m", "nlt.critic.mse_baseline", "--data-dir", data, "--out", f"/vol/critic/{tag}", "--tag", tag] + extra.split())


@app.function(gpu=GPU, timeout=6 * 3600, **COMMON)
def bits(tag: str, extra: str = "", data: str = DATA):
    return _run([sys.executable, "-m", "nlt.eval_bits.run", "--data-dir", data, "--out", f"/vol/results/bits_{tag}.json", "--tag", tag] + extra.split())


@app.function(gpu=GPU, timeout=6 * 3600, **COMMON)
def manifest(tag: str, extra: str = "", data: str = DATA):
    """redteam control manifest -> exact log p per row (nlt.eval_bits.score_manifest); extra must carry --ckpt and --manifest"""
    return _run([sys.executable, "-m", "nlt.eval_bits.score_manifest", "--data-dir", data, "--out", f"/vol/evals/scored_{tag}.parquet"] + extra.split())


@app.function(gpu=GPU, timeout=6 * 3600, **COMMON)
def winners(tag: str, extra: str = "", data: str = DATA):
    """critic-selected best-of-K per train pair (nlt.eval_bits.select_winners); extra carries --ckpt and --text-parquet"""
    return _run([sys.executable, "-m", "nlt.eval_bits.select_winners", "--data-dir", data, "--out", f"/vol/z/winners_{tag}/train.parquet"] + extra.split())


@app.function(gpu=GPU, timeout=6 * 3600, **COMMON)
def script(path: str, extra: str = ""):
    """run any repo script on a GPU with the volume mounted: --task script --path scripts/foo.py --extra '...'"""
    return _run([sys.executable, f"{REPO_REMOTE}/{path}"] + extra.split())


@app.function(timeout=1800, volumes={"/vol": vol}, cpu=2, memory=8 * 1024)
def cat(path: str):
    vol.reload(); print(open(f"/vol/{path}").read())


@app.local_entrypoint()
def main(task: str = "train", tag: str = "dev", extra: str = "", data: str = DATA, path: str = ""):
    if task == "train": print("rc", train.remote(tag, extra, data))
    elif task == "mse": print("rc", mse.remote(tag, extra, data))
    elif task == "bits": print("rc", bits.remote(tag, extra, data))
    elif task == "manifest": print("rc", manifest.remote(tag, extra, data))
    elif task == "winners": print("rc", winners.remote(tag, extra, data))
    elif task == "script": print("rc", script.remote(path, extra))
    elif task == "cat": cat.remote(path)
    else: raise SystemExit(task)
    print("done.")
