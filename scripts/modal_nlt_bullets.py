"""Modal app `nlt-bullets` (volume `nlt`): the bullet-list NLA reconstructor R(h_i, text) -> delta, its evaluation (FVE gain, bits,
controls, crux LOO) and the h_i-only baselines. Outputs under /vol/bullets/.

  modal run --detach scripts/modal_nlt_bullets.py --task train --tag R_bullets --extra "--text /vol/z/bullets-sonnet-v1/train/part_*.parquet --val-text /vol/z/bullets-sonnet-v1/val/part_*.parquet"
  modal run --detach scripts/modal_nlt_bullets.py --task eval --tag R_bullets_on_bullets --extra "--ckpt /vol/bullets/R_bullets/ckpt_final.pt --text ... --flip ..."
  modal run --detach scripts/modal_nlt_bullets.py --task baselines --tag baselines --extra "--pair-ids /vol/bullets/R_bullets/train_pairs.json"
  modal run scripts/modal_nlt_bullets.py --task ls --path bullets/R_bullets
"""
import os, sys
import modal
for _p in (os.path.dirname(os.path.abspath(__file__)), os.path.join(os.environ.get("PYTHONPATH", "/root/easyNLA").split(":")[0], "scripts")):
    if os.path.isdir(_p) and _p not in sys.path: sys.path.insert(0, _p)
from modal_nla_exp import image_base, REPO_LOCAL, REPO_REMOTE, REPO_IGNORE  # noqa: E402

vol = modal.Volume.from_name("nlt", create_if_missing=True)
SECRETS = [modal.Secret.from_name("nla-exp-secrets")]
DATA = os.environ.get("NLT_DATA", "/vol/data/qwen3_8b")
IGNORE = list(REPO_IGNORE) + ["*.log", "**/logs/**", "logs", "*.npy", "*.pt", "*.jsonl", "*.out", "wandb", "*.safetensors"]
image = image_base.add_local_dir(REPO_LOCAL, REPO_REMOTE, copy=False, ignore=IGNORE)
app = modal.App(os.environ.get("NLT_APP", "nlt-bullets"), image=image)
GPU = os.environ.get("NLT_GPU", "H100")
COMMON = dict(volumes={"/vol": vol}, secrets=SECRETS, cpu=8, memory=96 * 1024)


def _run(cmd):
    import subprocess
    os.environ["HF_HOME"] = "/vol/hf_cache"; os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.setdefault("WANDB_DIR", "/root/wandb"); os.makedirs("/root/wandb", exist_ok=True)
    try: vol.reload()
    except Exception as e: print(f"[modal] vol.reload skipped: {e}", flush=True)
    print("[modal] " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=REPO_REMOTE)
    try: vol.commit()
    except Exception as e: print(f"[modal] vol.commit: {e}", flush=True)
    return rc


@app.function(gpu=GPU, timeout=6 * 3600, **COMMON)
def train(tag: str, extra: str = "", data: str = DATA):
    return _run([sys.executable, "-m", "nlt.bullets.train", "--data-dir", data, "--out", f"/vol/bullets/{tag}", "--tag", tag] + extra.split())


@app.function(gpu=GPU, timeout=6 * 3600, **COMMON)
def evaluate(tag: str, extra: str = "", data: str = DATA):
    return _run([sys.executable, "-m", "nlt.bullets.evaluate", "--data-dir", data, "--out", f"/vol/bullets/eval/{tag}"] + extra.split())


@app.function(gpu=GPU, timeout=6 * 3600, **COMMON)
def baselines(tag: str, extra: str = "", data: str = DATA):
    return _run([sys.executable, "-m", "nlt.bullets.baselines", "--data-dir", data, "--out", f"/vol/bullets/{tag}"] + extra.split())


@app.function(gpu=GPU, timeout=6 * 3600, **COMMON)
def chain(tag: str, extra: str = "", data: str = DATA):
    """train then evaluate in ONE container: extra = '<train args> ;; <eval args without --ckpt>' (eval uses the fresh ckpt)"""
    tr_args, ev_args = [s.strip() for s in extra.split(";;")]
    rc = _run([sys.executable, "-m", "nlt.bullets.train", "--data-dir", data, "--out", f"/vol/bullets/{tag}", "--tag", tag] + tr_args.split())
    if rc != 0: return rc
    return _run([sys.executable, "-m", "nlt.bullets.evaluate", "--data-dir", data, "--out", f"/vol/bullets/eval/{tag}", "--ckpt", f"/vol/bullets/{tag}/ckpt_final.pt"] + ev_args.split())


@app.function(timeout=1800, volumes={"/vol": vol}, cpu=2, memory=8 * 1024)
def cat(path: str):
    vol.reload(); print(open(f"/vol/{path}").read())


@app.function(timeout=1800, volumes={"/vol": vol}, cpu=2, memory=8 * 1024)
def ls(path: str):
    import glob
    vol.reload()
    for f in sorted(glob.glob(f"/vol/{path}/*")): print(f, os.path.getsize(f) if os.path.isfile(f) else "<dir>")


@app.local_entrypoint()
def main(task: str = "train", tag: str = "dev", extra: str = "", data: str = DATA, path: str = ""):
    if task == "train": print("rc", train.remote(tag, extra, data))
    elif task == "eval": print("rc", evaluate.remote(tag, extra, data))
    elif task == "baselines": print("rc", baselines.remote(tag, extra, data))
    elif task == "chain": print("rc", chain.remote(tag, extra, data))
    elif task == "cat": cat.remote(path)
    elif task == "ls": ls.remote(path)
    else: raise SystemExit(task)
    print("done.")
