"""Modal app `nlt-prior` (volume `nlt`): the DALL-E-2-style diffusion-prior critic -- training, exact bits, manifest scoring.

  modal run --detach scripts/modal_nlt_prior.py --task train --tag smoke_depth --extra "--text-synth depth --steps 400 --batch 512"
  modal run --detach scripts/modal_nlt_prior.py --task bits --tag <tag> --extra "--ckpts prior:/vol/prior/<tag>/ckpt_final.pt --text-parquet ... --n 512 --ode-steps 64"
  modal run --detach scripts/modal_nlt_prior.py --task manifest --tag <tag>_twin_v0 --extra "--ckpt /vol/prior/<tag>/ckpt_final.pt --manifest /vol/evals/manifest_twinnext2_....parquet"
Outputs: /vol/prior/<tag>/ (checkpoints, eval_latest.json), /vol/results/bits_prior_<tag>.json, /vol/evals/scored_prior_<tag>.parquet. GPU via NLT_GPU (default H100).
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
app = modal.App(os.environ.get("NLT_APP", "nlt-prior"), image=image)
GPU = os.environ.get("NLT_GPU", "H100")
COMMON = dict(volumes={"/vol": vol}, secrets=SECRETS, cpu=16, memory=int(os.environ.get("NLT_MEM_GB", "200")) * 1024, ephemeral_disk=512 * 1024)


def _run(cmd):
    import subprocess
    os.environ["HF_HOME"] = "/root/hf_home"; os.makedirs("/root/hf_home", exist_ok=True)
    os.environ.setdefault("WANDB_DIR", "/root/wandb"); os.makedirs("/root/wandb", exist_ok=True)
    vol.reload(); print("[modal] " + " ".join(cmd), flush=True)
    every = int(os.environ.get("NLT_COMMIT_EVERY", "600"))
    if every > 0:
        import threading, time as _t
        def _bg():
            while True:
                _t.sleep(every)
                try: vol.commit(); print("[modal] bg volume commit ok", flush=True)
                except Exception as e: print(f"[modal] bg volume commit failed: {e}", flush=True)
        threading.Thread(target=_bg, daemon=True).start()
    rc = subprocess.call(cmd, cwd=REPO_REMOTE)
    vol.commit(); return rc


@app.function(gpu=GPU, timeout=23 * 3600, **COMMON)
def train(tag: str, extra: str = "", data: str = DATA):
    return _run([sys.executable, "-m", "nlt.prior.train", "--data-dir", data, "--out", f"/vol/prior/{tag}", "--tag", f"prior_{tag}"] + extra.split())


@app.function(gpu=GPU, timeout=8 * 3600, **COMMON)
def bits(tag: str, extra: str = "", data: str = DATA):
    return _run([sys.executable, "-m", "nlt.prior.eval_bits", "--data-dir", data, "--out", f"/vol/results/bits_prior_{tag}.json", "--tag", tag] + extra.split())


@app.function(gpu=GPU, timeout=6 * 3600, **COMMON)
def manifest(tag: str, extra: str = "", data: str = DATA):
    return _run([sys.executable, "-m", "nlt.prior.score_manifest", "--data-dir", data, "--out", f"/vol/evals/scored_prior_{tag}.parquet"] + extra.split())


@app.function(gpu=GPU, timeout=6 * 3600, **COMMON)
def script(path: str, extra: str = ""):
    return _run([sys.executable, f"{REPO_REMOTE}/{path}"] + extra.split())


@app.function(timeout=1800, volumes={"/vol": vol}, cpu=2, memory=8 * 1024)
def cat(path: str):
    vol.reload(); print(open(f"/vol/{path}").read())


@app.local_entrypoint()
def main(task: str = "train", tag: str = "dev", extra: str = "", data: str = DATA, path: str = ""):
    if task == "train": print("rc", train.remote(tag, extra, data))
    elif task == "bits": print("rc", bits.remote(tag, extra, data))
    elif task == "manifest": print("rc", manifest.remote(tag, extra, data))
    elif task == "script": print("rc", script.remote(path, extra))
    elif task == "cat": cat.remote(path)
    else: raise SystemExit(task)
    print("done.")
