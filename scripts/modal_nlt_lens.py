"""Modal app `nlt-lens` (volume `nlt`): lens toolkit jobs for the natural-language transcoder.

Every task is `python -m nlt.lens.<module> <args>` on one B200 (the repo is mounted at container start,
so code edits need no image rebuild). Launch detached:

  setsid nohup modal run --detach scripts/modal_nlt_lens.py --task smoke                     < /dev/null &
  setsid nohup modal run --detach scripts/modal_nlt_lens.py --task jlens --n-parts 8 --n-prompts 48 < /dev/null &
  setsid nohup modal run --detach scripts/modal_nlt_lens.py --task tuned --args "--steps 400"  < /dev/null &
  setsid nohup modal run --detach scripts/modal_nlt_lens.py --task run --module eval_lenses     < /dev/null &
  setsid nohup modal run --detach scripts/modal_nlt_lens.py --task extract --n-parts 4 --args "--n-pos 60000" < /dev/null &

Rules (PLAN.md §3): only app nlt-* and volume nlt; secret nla-exp-secrets (HF_TOKEN, WANDB_API_KEY); <= 8 GPUs.
"""
from __future__ import annotations

import os
import shlex
import sys

import modal

for _p in (os.path.dirname(os.path.abspath(__file__)), os.path.join(os.environ.get("PYTHONPATH", "/root/easyNLA").split(":")[0], "scripts")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)     # local: scripts/; in the container: /root/easyNLA/scripts
from modal_nla_exp import REPO_LOCAL, REPO_REMOTE, image_base  # noqa: E402  (validated B200 stack)

VOL_NAME = "nlt"
VOL_MNT = "/nlt"
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)


MOUNT_IGNORE = ["**/__pycache__/**", "**/*.pyc", "**/*.log", "**/logs/**", "**/*.parquet", "**/*.pt", "**/*.safetensors",
                "**/*.jsonl", "**/*.png", "**/*.pdf", "**/*.npy", "**/wandb/**", "**/data/**", "**/results/**"]
image = image_base.env({"HF_HOME": f"{VOL_MNT}/hf_cache", "HF_HUB_DISABLE_XET": "1", "NLT_VOL": VOL_MNT,
                        "PYTHONPATH": f"{REPO_REMOTE}:{REPO_REMOTE}/scripts", "TOKENIZERS_PARALLELISM": "false"})
# mount only what the lens jobs need (other agents write logs/data inside the repo while we snapshot)
for _sub in ("nla", "utils", "scripts", "nlt/lens"):
    image = image.add_local_dir(f"{REPO_LOCAL}/{_sub}", f"{REPO_REMOTE}/{_sub}", copy=False, ignore=MOUNT_IGNORE)
image = image.add_local_file(f"{REPO_LOCAL}/nlt/__init__.py", f"{REPO_REMOTE}/nlt/__init__.py", copy=False)
app = modal.App("nlt-lens", image=image)
SECRETS = [modal.Secret.from_name("nla-exp-secrets")]
COMMON = dict(volumes={VOL_MNT: vol}, secrets=SECRETS, cpu=8, memory=96 * 1024)


def _run(module: str, args: str, log_name: str | None = None) -> int:
    import subprocess
    import time
    os.chdir(REPO_REMOTE)
    try:
        vol.reload()
    except Exception as e:
        print(f"[prep] vol.reload skipped: {e}", flush=True)
    os.makedirs(f"{VOL_MNT}/logs", exist_ok=True)
    cmd = [sys.executable, "-m", f"nlt.lens.{module}"] + shlex.split(args)
    print("CMD:", " ".join(shlex.quote(c) for c in cmd), flush=True)
    log_path = f"{VOL_MNT}/logs/{log_name or module}_{time.strftime('%H%M%S')}.log"
    with open(log_path, "w") as lf:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
        for line in p.stdout:
            print(line, end="", flush=True)
            lf.write(line); lf.flush()
        rc = p.wait()
    vol.commit()
    print(f"[modal] {module} exited {rc}; log {log_path}", flush=True)
    return rc


@app.function(gpu="B200", timeout=6 * 3600, **COMMON)
def gpu1(module: str, args: str = "", log_name: str | None = None) -> int:
    return _run(module, args, log_name)


@app.function(gpu="H100", timeout=6 * 3600, **COMMON)
def gpu1_h100(module: str, args: str = "", log_name: str | None = None) -> int:
    return _run(module, args, log_name)


@app.function(timeout=2 * 3600, volumes={VOL_MNT: vol}, secrets=SECRETS, cpu=8, memory=64 * 1024)
def cpu(module: str, args: str = "", log_name: str | None = None) -> int:
    return _run(module, args, log_name)


@app.function(gpu="B200", timeout=30 * 60, **COMMON)
def smoke() -> str:
    import subprocess
    import torch
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout)
    print("torch", torch.__version__, "cuda", torch.cuda.device_count())
    os.makedirs(f"{VOL_MNT}/lens", exist_ok=True)
    rc = _run("fit_jlens", "--part 0 --n-parts 1 --n-prompts 1 --chunk 64 --out /nlt/lens/jlens_smoke", "smoke_jlens")
    return f"smoke rc={rc} vol={os.listdir(VOL_MNT)}"


@app.local_entrypoint()
def main(task: str = "smoke", module: str = "", args: str = "", n_parts: int = 1, n_prompts: int = 48, tag: str = "", gpu: str = "b200"):
    fn = gpu1_h100 if gpu.lower() == "h100" else gpu1
    if task == "smoke":
        print(smoke.remote())
    elif task == "jlens":
        rcs = list(gpu1.starmap([("fit_jlens", f"--part {p} --n-parts {n_parts} --n-prompts {n_prompts} {args}", f"jlens_part{p:02d}") for p in range(n_parts)]))
        print("part rcs", rcs)
        if all(rc == 0 for rc in rcs):
            print("reduce rc", cpu.remote("fit_jlens", "--reduce", "jlens_reduce"))
    elif task == "tuned":
        print("rc", gpu1.remote("fit_tuned", args, "tuned"))
    elif task == "extract":
        rcs = list(gpu1.starmap([("extract_acts", f"--part {p} --n-parts {n_parts} {args}", f"extract_part{p:02d}") for p in range(n_parts)]))
        print("part rcs", rcs)
    elif task == "run":            # any module on one GPU (--gpu h100 for light jobs)
        print("rc", fn.remote(module, args, tag or module))
    elif task == "runparts":       # same module, --part p --n-parts N appended, one GPU per part
        rcs = list(fn.starmap([(module, f"{args} --part {p} --n-parts {n_parts}", f"{tag or module}_part{p:02d}") for p in range(n_parts)]))
        print("part rcs", rcs)
    elif task == "cpu":
        print("rc", cpu.remote(module, args, tag or module))
    else:
        raise SystemExit(f"unknown task {task}")
