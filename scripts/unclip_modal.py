"""Modal app for the unCLIP DECODER p(h | e) (agent A). Workspace safety-sahan, app nla-unclip-dec, volumes as modal_glp.py.
  modal run --detach scripts/unclip_modal.py --task train_stream --tag dec_main --extra "--batch 8192 --lr 2e-4 --prior-lr 2e-5"   # B200:8 = 4 producers + 4 FSDP consumers
  modal run --detach scripts/unclip_modal.py --task train_static4 --tag dec_smoke --extra "--parquet-glob ... --max-steps 300"    # B200:4, static shards (smokes)
  modal run scripts/unclip_modal.py --task eval --tag dec_main/snap_000100M --extra "--tests fm,exact,recon"                        # B200:2 (LM for the downstream KL)
Outputs: /vol_glp/unclip/decoder/<tag>/{train.log, latest/, snap_XXXXXXM/{adapter_latest.pt, prior_cotrained_latest.pt, eval.json}, eval_*.json}."""
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
app = modal.App("nla-unclip-dec", image=image)
DEC = "/vol_glp/unclip/decoder"


def _commit_loop(every):
    import threading
    ev = threading.Event()
    def loop():
        while not ev.wait(every):
            try: vol_glp.commit()
            except Exception as e: print(f"[modal] periodic commit failed: {str(e)[:120]}", flush=True)
    threading.Thread(target=loop, daemon=True).start(); return ev


def _stream(cmd, log_path, env=None, commit_every=600):
    """run cmd, tee stdout to a durable log on the volume, commit the volume periodically (snapshots visible mid-run)"""
    import subprocess, time as _t
    os.makedirs(os.path.dirname(log_path), exist_ok=True); logf = open(log_path, "ab")
    logf.write(f"[modal] container started {_t.strftime('%Y-%m-%d %H:%M:%S UTC', _t.gmtime())}: {' '.join(cmd)[:400]}\n".encode()); logf.flush(); vol_glp.commit()
    proc = subprocess.Popen(cmd, cwd=REPO_REMOTE, env=env or dict(os.environ), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stop = _commit_loop(commit_every); tail = []
    for line in proc.stdout:
        sys.stdout.buffer.write(line); sys.stdout.flush(); logf.write(line); logf.flush(); tail = (tail + [line.decode(errors="replace").rstrip()])[-40:]
    rc = proc.wait(); logf.close(); stop.set(); vol_glp.commit()
    if rc != 0: print(f"[modal] exited {rc}; tail:\n" + "\n".join(tail), flush=True)
    return rc


def _env():
    e = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", WANDB_DIR="/root/wandb"); os.makedirs("/root/wandb", exist_ok=True); return e


@app.function(gpu="B200:8", timeout=23 * 3600, volumes=VOLS, secrets=SECRETS, cpu=64, memory=1024 * 1024, ephemeral_disk=600 * 1024)
def train_stream(tag: str, extra: str = "", n_prod: int = 4, n_cons: int = 4, prior: str = "/vol_glp/glp27b_main/ckpts/snap_001966M", launch_extra: str = ""):
    """streaming producers + FSDP consumers on one node (scripts/unclip_launch.py)"""
    out = f"{DEC}/{tag}"
    cmd = [sys.executable, f"{REPO_REMOTE}/scripts/unclip_launch.py", "--out-dir", out, "--n-producers", str(n_prod), "--n-consumers", str(n_cons), "--prior", prior, "--tag", tag, "--wandb-name", tag, "--trainer-extra", extra] + launch_extra.split()
    return _stream(cmd, f"{out}/train.log", _env())


@app.function(gpu="B200:4", timeout=23 * 3600, volumes=VOLS, secrets=SECRETS, cpu=32, memory=768 * 1024, ephemeral_disk=600 * 1024)
def train_stream4(tag: str, extra: str = "", n_prod: int = 2, n_cons: int = 2, prior: str = "/vol_glp/glp27b_main/ckpts/snap_001966M", launch_extra: str = ""):
    """B200:4 fallback when 8-GPU containers stay pending: 2 producers + 2 FSDP consumers (use --batch 8192 --grad-accum 2 for the same global batch)"""
    out = f"{DEC}/{tag}"
    cmd = [sys.executable, f"{REPO_REMOTE}/scripts/unclip_launch.py", "--out-dir", out, "--n-producers", str(n_prod), "--n-consumers", str(n_cons), "--prior", prior, "--tag", tag, "--wandb-name", tag, "--trainer-extra", extra] + launch_extra.split()
    return _stream(cmd, f"{out}/train.log", _env())


def _static(tag, extra, nproc, prior):
    out = f"{DEC}/{tag}"
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}", "-m", "nla.unclip.train_dec", "--prior", prior, "--stats", "/vol_glp/glp27b_main/rep_statistics.pt",
           "--out", out, "--tag", tag, "--wandb-name", tag] + extra.split()
    return _stream(cmd, f"{out}/train.log", _env(), commit_every=300)


@app.function(gpu="B200:4", timeout=23 * 3600, volumes=VOLS, secrets=SECRETS, cpu=32, memory=768 * 1024, ephemeral_disk=600 * 1024)
def train_static4(tag: str, extra: str = "", prior: str = "/vol_glp/glp27b_main/ckpts/snap_001966M"):
    return _static(tag, extra, 4, prior)


@app.function(gpu="B200:8", timeout=23 * 3600, volumes=VOLS, secrets=SECRETS, cpu=64, memory=1024 * 1024, ephemeral_disk=600 * 1024)
def train_static8(tag: str, extra: str = "", prior: str = "/vol_glp/glp27b_main/ckpts/snap_001966M"):
    return _static(tag, extra, 8, prior)


@app.function(gpu="B200:2", timeout=8 * 3600, volumes=VOLS, secrets=SECRETS, cpu=32, memory=512 * 1024, ephemeral_disk=600 * 1024)
def eval_dec(snap: str, extra: str = ""):
    """decoder evals (scripts/unclip_eval_dec.py) on <DEC>/<snap>; the downstream-KL test needs the 27B LM on the second GPU"""
    vol_glp.reload()
    cmd = [sys.executable, f"{REPO_REMOTE}/scripts/unclip_eval_dec.py", "--snap", f"{DEC}/{snap}"] + extra.split()
    return _stream(cmd, f"{DEC}/{snap}/eval_dec.log", _env(), commit_every=120)


@app.function(gpu="B200", timeout=8 * 3600, volumes=VOLS, secrets=SECRETS, cpu=16, memory=256 * 1024, ephemeral_disk=600 * 1024)
def eval_dec1(snap: str, extra: str = ""):
    """single-GPU decoder evals (no LM tests)"""
    vol_glp.reload()
    cmd = [sys.executable, f"{REPO_REMOTE}/scripts/unclip_eval_dec.py", "--snap", f"{DEC}/{snap}"] + extra.split()
    return _stream(cmd, f"{DEC}/{snap}/eval_dec.log", _env(), commit_every=120)


@app.function(gpu="B200", timeout=4 * 3600, volumes=VOLS, secrets=SECRETS, cpu=16, memory=256 * 1024, ephemeral_disk=600 * 1024)
def shell1(cmd: str):
    import subprocess
    vol_glp.reload(); rc = subprocess.call(["bash", "-lc", cmd], cwd=REPO_REMOTE, env=_env()); vol_glp.commit(); return rc


@app.local_entrypoint()
def main(task: str = "train_stream", tag: str = "", extra: str = "", prior: str = "/vol_glp/glp27b_main/ckpts/snap_001966M", n_prod: int = 4, n_cons: int = 4, launch_extra: str = "", cmd: str = ""):
    if task == "train_stream": print("rc", train_stream.remote(tag, extra, n_prod, n_cons, prior, launch_extra))
    elif task == "train_stream4": print("rc", train_stream4.remote(tag, extra, n_prod, n_cons, prior, launch_extra))
    elif task == "train_static4": print("rc", train_static4.remote(tag, extra, prior))
    elif task == "train_static8": print("rc", train_static8.remote(tag, extra, prior))
    elif task == "eval": print("rc", eval_dec.remote(tag, extra))
    elif task == "eval1": print("rc", eval_dec1.remote(tag, extra))
    elif task == "shell": print("rc", shell1.remote(cmd))
    else: raise SystemExit(task)
