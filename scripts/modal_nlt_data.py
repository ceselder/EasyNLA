"""Modal app `nlt-data` (volume `nlt`): multi-layer Qwen3-8B activation data for the natural-language transcoder.

  modal run scripts/modal_nlt_data.py --task smoke                       # 1 GPU, tiny targets, end-to-end check
  modal run --detach scripts/modal_nlt_data.py --task extract --n-producers 4 --target-train 80000 --target-val 3000
  modal run --detach scripts/modal_nlt_data.py --task finalize            # global stats + fixed pair lists (CPU)
  modal run scripts/modal_nlt_data.py --task ls --path data/qwen3_8b/train

Volume layout: /vol/data/qwen3_8b/{train,val}/{acts,meta,docs}_PP_NNNN.{npy,parquet}, /vol/data/qwen3_8b/stats.pt, pairs_{train,val}.parquet.
"""
import os, sys
import modal
for _p in (os.path.dirname(os.path.abspath(__file__)), os.path.join(os.environ.get("PYTHONPATH", "/root/easyNLA").split(":")[0], "scripts")):
    if os.path.isdir(_p) and _p not in sys.path: sys.path.insert(0, _p)
from modal_nla_exp import image_base, REPO_LOCAL, REPO_REMOTE, REPO_IGNORE  # noqa: E402  (the validated B200 stack; no app of theirs is touched)

VOL_NAME = "nlt"
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
SECRETS = [modal.Secret.from_name("nla-exp-secrets")]
DATA = "/vol/data/qwen3_8b"
IGNORE = list(REPO_IGNORE) + ["*.log", "**/logs/**", "logs", "*.npy", "*.pt", "*.jsonl", "*.out", "wandb"]   # other agents write logs inside the repo
image = image_base.add_local_dir(REPO_LOCAL, REPO_REMOTE, copy=False, ignore=IGNORE)
app = modal.App("nlt-data", image=image)
GPU = os.environ.get("NLT_GPU", "B200")


def _local_hf():
    """model weights on the container's local disk (no shared-volume HF cache -> no partial-snapshot races between producers)"""
    os.environ["HF_HOME"] = "/root/hf_home"; os.makedirs("/root/hf_home", exist_ok=True)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


def _run(cmd, cwd=REPO_REMOTE):
    import subprocess
    print("[modal] " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=cwd)
    vol.commit()
    return rc


@app.function(gpu=GPU, timeout=4 * 3600, volumes={"/vol": vol}, secrets=SECRETS, cpu=8, memory=96 * 1024, ephemeral_disk=256 * 1024)
def extract(index: int, n_producers: int, target_train: int, target_val: int, extra: str = "", out: str = DATA):
    _local_hf()
    from huggingface_hub import snapshot_download
    snapshot_download("Qwen/Qwen3-8B", token=os.environ.get("HF_TOKEN"), allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"])
    cmd = [sys.executable, "-m", "nlt.data.extract", "--out-dir", out, "--index", str(index), "--n-producers", str(n_producers),
           "--target-train", str(target_train), "--target-val", str(target_val)] + extra.split()
    return _run(cmd)


@app.function(timeout=3 * 3600, volumes={"/vol": vol}, secrets=SECRETS, cpu=16, memory=160 * 1024)
def finalize(extra: str = "", data: str = DATA):
    cmd = [sys.executable, "-m", "nlt.data.finalize", "--data-dir", data] + extra.split()
    return _run(cmd)


@app.function(timeout=600, volumes={"/vol": vol}, cpu=2, memory=4 * 1024)
def ls(path: str = "data/qwen3_8b"):
    import subprocess
    vol.reload()
    print(subprocess.run(["bash", "-c", f"ls -la /vol/{path} | head -80; du -sh /vol/{path} 2>/dev/null"], capture_output=True, text=True).stdout)


@app.function(timeout=1800, volumes={"/vol": vol}, cpu=2, memory=8 * 1024)
def cat(path: str):
    vol.reload(); print(open(f"/vol/{path}").read())


@app.local_entrypoint()
def main(task: str = "smoke", n_producers: int = 4, target_train: int = 80000, target_val: int = 3000, extra: str = "", path: str = "data/qwen3_8b", out: str = DATA):
    if task == "smoke":
        rc = extract.remote(0, 1, 2000, 200, "--docs-per-batch 16 --shard-size 1024 --max-minutes 20 " + extra, out=f"{DATA}_smoke")
        print("smoke extract rc", rc)
        print("finalize rc", finalize.remote("--max-stats-pos 4000 " + extra, data=f"{DATA}_smoke"))
    elif task == "extract":
        rcs = list(extract.starmap([(i, n_producers, target_train, target_val, extra, out) for i in range(n_producers)]))
        print("extract rcs", rcs)
    elif task == "finalize":
        print("finalize rc", finalize.remote(extra, data=out))
    elif task == "ls":
        ls.remote(path)
    elif task == "cat":
        cat.remote(path)
    else:
        raise SystemExit(f"unknown task {task}")
    print("done.")
