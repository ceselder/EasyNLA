"""Modal app for pretraining the activation flow prior (GLP-style) on Qwen3.6-27B layer-42 activations.
  modal run scripts/modal_glp.py --task smoke                      # B200:3, tiny model, ~20 min end-to-end check
  modal run --detach scripts/modal_glp.py --task pretrain --tag glp27b_d6   # B200:8, 5 producers + 3 DDP consumers, 2B activations
"""
import os, sys
import modal
for _p in (os.path.dirname(os.path.abspath(__file__)), os.path.join(os.environ.get("PYTHONPATH", "/root/easyNLA").split(":")[0], "scripts")):
    if os.path.isdir(_p) and _p not in sys.path: sys.path.insert(0, _p)
from modal_nla_exp import image_base, SECRETS, REPO_LOCAL, REPO_REMOTE, REPO_IGNORE  # noqa: E402

vol_glp = modal.Volume.from_name("nla-glp", create_if_missing=True)
VOLS = {"/vol_glp": vol_glp}
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
def pretrain(tag: str, n_producers: int = 5, n_consumers: int = 3, total_samples: float = 2e9, extra: str = ""):
    return _run(["--out-dir", f"/vol_glp/{tag}", "--n-producers", str(n_producers), "--n-consumers", str(n_consumers),
                 "--total-samples", str(total_samples), "--wandb-name", tag] + extra.split())


@app.function(gpu="B200:3", timeout=2 * 3600, **COMMON)
def smoke(tag: str = "smoke_glp", extra: str = ""):
    return _run(["--out-dir", f"/vol_glp/{tag}", "--n-producers", "2", "--n-consumers", "1", "--d-model", "2048", "--d-mlp", "4096", "--n-layers", "3",
                 "--batch", "2048", "--total-samples", "3e7", "--max-tokens-per-producer", "1.5e7", "--max-hours", "1.0", "--wandb-name", tag,
                 "--producer-extra", "--stats-n 200000 --heldout-n 16384 --heldout-docs-full 16 --shard-size 8192",
                 "--trainer-extra", "--ckpt-every 200 --eval-every 100 --eval-n 8192 --snapshot-every-samples 1e7 --stream-timeout 300"] + extra.split())


@app.local_entrypoint()
def main(task: str = "smoke", tag: str = "", n_producers: int = 5, n_consumers: int = 3, total_samples: float = 2e9, extra: str = ""):
    if task == "smoke":
        print("rc", smoke.remote(tag or "smoke_glp", extra))
    elif task == "pretrain":
        assert tag, "--tag required"
        print("rc", pretrain.remote(tag, n_producers, n_consumers, total_samples, extra))
