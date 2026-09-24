"""Modal app for the CLIP-style (in-batch contrastive) activation <-> explanation critic (nla/contrastive).
Deploy once, then spawn (robust to the local client dying):
  modal deploy scripts/clip_modal.py
  python -c "import modal; print(modal.Function.from_name('nla-clip','train').spawn('clipQ_opus', '<extra>').object_id)"
Tasks: probe (CPU: shard schemas / doc multiplicity), train (B200:8 torchrun), train4 (B200:4 fallback), evaluate (1 B200, scripts/clip_eval.py).
"""
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
app = modal.App("nla-clip", image=image)


@app.function(volumes=VOLS, timeout=3600, cpu=8.0, memory=64 * 1024, secrets=SECRETS)
def probe(globs: str = "/vol_q36/data/acts_qwen36_L42/shard_*.parquet,/vol_glp/scale/g1/shards/shard_*.parquet,/vol_glp/scale/g2/shards/shard_*.parquet"):
    import glob, collections, pyarrow.parquet as pq
    for g in globs.split(","):
        fs = sorted(glob.glob(g)); print(f"== {g}: {len(fs)} files", flush=True)
        if not fs: continue
        pf = pq.ParquetFile(fs[0]); print("  schema:", pf.schema_arrow, flush=True); print("  rows in first file:", pf.metadata.num_rows, flush=True)
        cols = [c for c in ("doc_id", "is_val", "position", "pos") if c in pf.schema_arrow.names]
        t = pq.read_table(fs[0], columns=cols) if cols else None
        if t is not None and "doc_id" in cols:
            c = collections.Counter(t.column("doc_id").to_pylist()); m = collections.Counter(c.values())
            print(f"  doc multiplicity in first file: {sorted(m.items())[:12]} (docs {len(c)})", flush=True)
        if t is not None and "is_val" in cols: print("  is_val true:", sum(bool(x) for x in t.column("is_val").to_pylist()), flush=True)
        ex = pq.read_table(fs[0], columns=[c for c in ("explanation",) if c in pf.schema_arrow.names]).slice(0, 2).to_pylist()
        print("  example:", str(ex)[:600], flush=True)
    for p_ in ("/vol_q36/data/sft/av_sft_val.parquet", "/vol_q36/data/sft/av_sft_val_clean1.parquet", "/vol_q36/data/sft/av_sft_train.parquet"):
        if os.path.exists(p_):
            pf = pq.ParquetFile(p_); print(f"== {p_}: rows {pf.metadata.num_rows}; cols {pf.schema_arrow.names}", flush=True)
            if "doc_id" in pf.schema_arrow.names:
                c = collections.Counter(pq.read_table(p_, columns=["doc_id"]).column(0).to_pylist()); print("   doc multiplicity:", sorted(collections.Counter(c.values()).items())[:10], flush=True)
    for p_ in ("/vol/ckpts/qwen36_27b/ar_sft_merged", "/vol_glp/glp27b_main/rep_statistics.pt"):
        print(p_, "exists" if os.path.exists(p_) else "MISSING", sorted(os.listdir(p_))[:20] if os.path.isdir(p_) else "", flush=True)


def _commit_loop(every):
    import threading
    ev = threading.Event()
    def loop():
        while not ev.wait(every):
            try: vol_glp.commit()
            except Exception as e: print(f"[modal] periodic commit failed: {str(e)[:120]}", flush=True)
    threading.Thread(target=loop, daemon=True).start(); return ev


def _torchrun(tag, extra, nproc, module="nla.contrastive.train_clip", commit_every=600):
    import subprocess, time as _t
    from playground_app import resolve_base
    base = resolve_base("Qwen/Qwen3.6-27B", local_snapshot=True)
    out = f"/vol_glp/clip/{tag}"; os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}", "-m", module, "--base", base, "--out", out, "--tag", tag] + extra.split()
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", WANDB_DIR="/root/wandb"); os.makedirs("/root/wandb", exist_ok=True)
    logf = open(os.path.join(out, "train.log"), "ab")
    logf.write(f"[modal] container started {_t.strftime('%Y-%m-%d %H:%M:%S UTC', _t.gmtime())}, {nproc} ranks\n".encode()); logf.flush(); vol_glp.commit()
    proc = subprocess.Popen(cmd, cwd=REPO_REMOTE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stop = _commit_loop(commit_every); tail = []
    for line in proc.stdout:
        sys.stdout.buffer.write(line); sys.stdout.flush(); logf.write(line); logf.flush(); tail = (tail + [line.decode(errors="replace").rstrip()])[-60:]
    rc = proc.wait(); logf.close(); stop.set(); vol_glp.commit()
    if rc != 0: raise SystemExit(f"train exited {rc}; tail:\n" + "\n".join(tail))
    return out


@app.function(gpu="B200:8", timeout=23 * 3600, volumes=VOLS, secrets=SECRETS, cpu=64, memory=1024 * 1024, ephemeral_disk=600 * 1024)
def train(tag: str, extra: str = "", nproc: int = 8):
    return _torchrun(tag, extra, nproc)


@app.function(gpu="B200:4", timeout=23 * 3600, volumes=VOLS, secrets=SECRETS, cpu=32, memory=768 * 1024, ephemeral_disk=600 * 1024)
def train4(tag: str, extra: str = ""):
    return _torchrun(tag, extra, 4)


@app.function(gpu="B200:2", timeout=3 * 3600, volumes=VOLS, secrets=SECRETS, cpu=32, memory=512 * 1024, ephemeral_disk=600 * 1024)
def train2(tag: str, extra: str = ""):
    return _torchrun(tag, extra, 2, commit_every=120)


@app.function(gpu="B200", timeout=6 * 3600, volumes=VOLS, secrets=SECRETS, cpu=16, memory=256 * 1024, ephemeral_disk=600 * 1024)
def train1(tag: str, extra: str = ""):
    return _torchrun(tag, extra, 1, commit_every=120)


@app.function(gpu="B200", timeout=6 * 3600, volumes=VOLS, secrets=SECRETS, cpu=16, memory=256 * 1024, ephemeral_disk=600 * 1024)
def evaluate(ckpt: str, extra: str = ""):
    import subprocess
    from playground_app import resolve_base
    base = resolve_base("Qwen/Qwen3.6-27B", local_snapshot=True)
    cmd = [sys.executable, f"{REPO_REMOTE}/scripts/clip_eval.py", "--ckpt", ckpt, "--base", base] + extra.split()
    proc = subprocess.Popen(cmd, cwd=REPO_REMOTE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT); tail = []
    for line in proc.stdout: sys.stdout.buffer.write(line); sys.stdout.flush(); tail = (tail + [line.decode(errors="replace").rstrip()])[-60:]
    rc = proc.wait(); vol_glp.commit()
    if rc != 0: raise SystemExit(f"eval exited {rc}; tail:\n" + "\n".join(tail))
    return ckpt


@app.function(gpu="B200", timeout=6 * 3600, volumes=VOLS, secrets=SECRETS, cpu=16, memory=256 * 1024, ephemeral_disk=600 * 1024)
def script(name: str, extra: str = ""):
    """run scripts/<name> on one B200 with --base = the local Qwen3.6-27B snapshot (pooling diagnostic etc.)"""
    import subprocess
    from playground_app import resolve_base
    base = resolve_base("Qwen/Qwen3.6-27B", local_snapshot=True)
    proc = subprocess.Popen([sys.executable, f"{REPO_REMOTE}/scripts/{name}", "--base", base] + extra.split(), cwd=REPO_REMOTE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT); tail = []
    for line in proc.stdout: sys.stdout.buffer.write(line); sys.stdout.flush(); tail = (tail + [line.decode(errors="replace").rstrip()])[-60:]
    rc = proc.wait(); vol_glp.commit()
    if rc != 0: raise SystemExit(f"{name} exited {rc}; tail:\n" + "\n".join(tail))
    return name


@app.local_entrypoint()
def main(task: str = "probe", tag: str = "", extra: str = ""):
    if task == "probe": probe.remote()
    elif task == "train": train.remote(tag, extra)
    elif task == "train2": train2.remote(tag, extra)
    elif task == "train1": train1.remote(tag, extra)
    elif task == "evaluate": evaluate.remote(tag, extra)
