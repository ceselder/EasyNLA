"""Modal app for the unCLIP prior p(e | z) (nla/unclip). Deploy once, then spawn (robust to the local client dying):
  modal deploy scripts/unclip_prior_modal.py
  python scripts/unclip_prior_launch.py spawn train4 <tag> "<extra args>"       # or via scripts/unclip_prior_launch.sh (poll + resume)
Tasks: probe (CPU), train4 (B200:4 torchrun), train1 (B200:1 smoke / small runs), selfcheck (B200:1, scripts/unclip_prior_selfcheck.py).
Outputs live under /vol_glp/unclip/prior/<tag>/ (volume nla-glp)."""
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
app = modal.App("nla-unclip-prior", image=image)
ROOT = "/vol_glp/unclip/prior"


@app.function(volumes=VOLS, timeout=1800, cpu=4.0, memory=32 * 1024, secrets=SECRETS)
def probe():
    import json, glob
    for d in ("/vol_glp/unclip", "/vol_glp/unclip/prior", "/vol_glp/clip/clipQ_opus_frozen_plain/latest"):
        print(d, sorted(os.listdir(d)) if os.path.isdir(d) else "MISSING", flush=True)
    ej = "/vol_glp/unclip/encoder.json"
    print("encoder.json:", json.load(open(ej)) if os.path.exists(ej) else "MISSING", flush=True)
    for g in ("/vol_q36/data/acts_qwen36_L42/shard_*.parquet", "/vol_glp/scale/g1/shards/shard_*.parquet", "/vol_glp/scale/g2/shards/shard_*.parquet"):
        print(g, len(glob.glob(g)), "files", flush=True)


@app.function(volumes=VOLS, timeout=3600, cpu=8.0, memory=64 * 1024, secrets=SECRETS)
def count_rows(globs: str):
    """non-val rows per glob (is_val column only) -> json on stdout; sizes the Gemma phase"""
    import json
    from nla.flow.train_cond import count_shard_rows
    out = {}
    for g in globs.split(","):
        c = count_shard_rows(g.strip()); out[g.strip()] = {"files": len(c), "rows": sum(n for _, n in c), "per_file": [n for _, n in c][:5]}
        print(g.strip(), out[g.strip()], flush=True)
    return json.dumps(out)


def _commit_loop(every):
    import threading
    ev = threading.Event()
    def loop():
        while not ev.wait(every):
            try: vol_glp.commit()
            except Exception as e: print(f"[modal] periodic commit failed: {str(e)[:120]}", flush=True)
    threading.Thread(target=loop, daemon=True).start(); return ev


def _run(cmd, logpath, commit_every=300, cwd=REPO_REMOTE):
    """stream a subprocess to stdout + a durable log on the volume, periodic commits so snapshots are visible mid-run"""
    import subprocess, time as _t
    os.makedirs(os.path.dirname(logpath), exist_ok=True)
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", WANDB_DIR="/root/wandb", TOKENIZERS_PARALLELISM="false"); os.makedirs("/root/wandb", exist_ok=True)
    logf = open(logpath, "ab"); logf.write(f"[modal] container started {_t.strftime('%Y-%m-%d %H:%M:%S UTC', _t.gmtime())}: {' '.join(cmd)}\n".encode()); logf.flush(); vol_glp.commit()
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stop = _commit_loop(commit_every); tail = []
    for line in proc.stdout:
        sys.stdout.buffer.write(line); sys.stdout.flush(); logf.write(line); logf.flush(); tail = (tail + [line.decode(errors="replace").rstrip()])[-80:]
    rc = proc.wait(); logf.close(); stop.set(); vol_glp.commit()
    if rc != 0: raise SystemExit(f"exited {rc}; tail:\n" + "\n".join(tail))
    return rc


def _train(tag, extra, nproc):
    from playground_app import resolve_base
    base = resolve_base("Qwen/Qwen3.6-27B", local_snapshot=True)
    out = f"{ROOT}/{tag}"; os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}", "-m", "nla.unclip.train_prior", "--base", base, "--out", out, "--tag", tag] + extra.split()
    _run(cmd, os.path.join(out, "train.log")); return out


@app.function(gpu="B200:8", timeout=23 * 3600, volumes=VOLS, secrets=SECRETS, cpu=64, memory=1024 * 1024, ephemeral_disk=600 * 1024)
def train8(tag: str, extra: str = ""):
    """8 ranks (the data-scaling runs: frozen trunk, all Gemma renderings)"""
    return _train(tag, extra, 8)


@app.function(gpu="B200:4", timeout=23 * 3600, volumes=VOLS, secrets=SECRETS, cpu=32, memory=768 * 1024, ephemeral_disk=600 * 1024)
def train4(tag: str, extra: str = ""):
    return _train(tag, extra, 4)


@app.function(gpu="B200:3", timeout=23 * 3600, volumes=VOLS, secrets=SECRETS, cpu=32, memory=640 * 1024, ephemeral_disk=600 * 1024)
def train3(tag: str, extra: str = ""):
    """3 ranks: leaves one B200 of the prior's 4-GPU budget free for self-checks / API tests on early snapshots while training runs"""
    return _train(tag, extra, 3)


@app.function(gpu="B200:2", timeout=23 * 3600, volumes=VOLS, secrets=SECRETS, cpu=32, memory=512 * 1024, ephemeral_disk=600 * 1024)
def train2(tag: str, extra: str = ""):
    return _train(tag, extra, 2)


@app.function(gpu="B200", timeout=12 * 3600, volumes=VOLS, secrets=SECRETS, cpu=16, memory=256 * 1024, ephemeral_disk=600 * 1024)
def train1(tag: str, extra: str = ""):
    return _train(tag, extra, 1)


@app.function(gpu="B200", timeout=8 * 3600, volumes=VOLS, secrets=SECRETS, cpu=16, memory=256 * 1024, ephemeral_disk=600 * 1024)
def selfcheck(prior_dir: str, extra: str = ""):
    """scripts/unclip_prior_selfcheck.py on one B200: held-out FM per t, exact PMI, retrieval, wrong-detail, number edits, API timing"""
    from playground_app import resolve_base
    base = resolve_base("Qwen/Qwen3.6-27B", local_snapshot=True)
    cmd = [sys.executable, f"{REPO_REMOTE}/scripts/unclip_prior_selfcheck.py", "--prior-dir", prior_dir, "--base", base] + extra.split()
    _run(cmd, os.path.join(prior_dir, "selfcheck.log"), commit_every=120); return prior_dir


@app.function(gpu="B200", timeout=2 * 3600, volumes=VOLS, secrets=SECRETS, cpu=16, memory=256 * 1024, ephemeral_disk=600 * 1024)
def shell(cmd: str):
    """run an arbitrary python -m / script line from the repo on one B200 (debug)"""
    import shlex
    _run([sys.executable] + shlex.split(cmd), f"{ROOT}/_shell.log", commit_every=120); return 0


@app.local_entrypoint()
def main(task: str = "probe", tag: str = "", extra: str = "", prior_dir: str = ""):
    if task == "probe": probe.remote()
    elif task == "count": print(count_rows.remote(extra or "/vol_q36/data/acts_qwen36_L42/shard_*.parquet,/vol_glp/scale/g1/shards/shard_*.parquet,/vol_glp/scale/g2/shards/shard_*.parquet"))
    elif task == "train8": print(train8.remote(tag, extra))
    elif task == "train4": print(train4.remote(tag, extra))
    elif task == "train3": print(train3.remote(tag, extra))
    elif task == "train2": print(train2.remote(tag, extra))
    elif task == "train1": print(train1.remote(tag, extra))
    elif task == "selfcheck": print(selfcheck.remote(prior_dir, extra))
    elif task == "shell": print(shell.remote(extra))
