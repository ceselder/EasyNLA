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
def pretrain(tag: str, config: str = "", sets: str = ""):
    args = ["--out-dir", f"/vol_glp/{tag}", "--wandb-name", tag]
    if config: args += ["--config", f"{REPO_REMOTE}/{config}"]
    if sets.strip(): args += ["--set"] + sets.split()
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
    out = f"/vol_glp/{tag}/eval_lm_{ckpt}.json"
    cmd = [sys.executable, "-m", "nla.flow.eval_lm", "--base", base, "--ckpt", f"/vol_glp/{tag}/ckpts/{ckpt}", "--stats", f"/vol_glp/{tag}/rep_statistics.pt",
           "--heldout", f"/vol_glp/{tag}/heldout_acts.pt", "--out", out] + extra.split()
    rc = subprocess.call(cmd, cwd=REPO_REMOTE); vol_glp.commit(); return rc


@app.local_entrypoint()
def main(task: str = "smoke", tag: str = "", config: str = "", sets: str = "", ckpt: str = "final", extra: str = ""):
    """--sets "train.lr=1e-4 model.n_layers=12" ; --config configs/glp/<override>.yaml"""
    if task == "smoke":
        print("rc", smoke.remote(tag or "smoke_glp", sets))
    elif task == "pretrain":
        assert tag, "--tag required"
        print("rc", pretrain.remote(tag, config, sets))
    elif task == "eval_lm":
        print("rc", eval_lm.remote(tag, ckpt, extra))
