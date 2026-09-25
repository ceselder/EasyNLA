"""Modal app `nlt-q36` (volume `nlt`, dir /vol/q36): the Qwen3.6-27B natural-language transcoder built on the oracle lens.

Read-only mounts (never written): /vol_go = grad-olens (olens checkpoints ckpt/ar_ivrl, reconstructor ckpt/h2hpfx_ar_r512, acts_v2),
/vol_ol1 = olens-1layer (harvest_v5 contexts, jlens, frozen head), /vol_data = olens-new-arch (27B weights in hf_cache).

Two images: `hf` (torch 2.8.0 / transformers 5.5.4 / peft 0.19.1, the grad-olens modal_app.py pins) and `vllm` (vllm 0.21.0 + the
vllm-metamodels fork, the grad-olens modal_vllm.py pins). The whole ~/nlt repo is mounted at /root/easyNLA; q36 scripts run from
/root/easyNLA/nlt/q36 with PYTHONPATH covering both.

  modal run --detach scripts/modal_nlt_q36.py --task hf   --gpus 1 --script extract_layers.py --args "..."
  modal run --detach scripts/modal_nlt_q36.py --task vllm --gpus 1 --script rollout_vllm.py  --args "..."
  modal run --detach scripts/modal_nlt_q36.py --task vllm-many --gpus 1 --script rollout_vllm.py --args "... ;; ..."
  modal run --detach scripts/modal_nlt_q36.py --task pyrun --code "print(1)"
"""
import os
import subprocess

import modal

APP_NAME = os.environ.get("NLT_Q36_APP", "nlt-q36")
GPU_TYPE = os.environ.get("NLT_Q36_GPU", "B200")                  # ONE type per launch (Modal 1.5.4 rejects fallback lists); e.g. NLT_Q36_GPU=H100 to route around a B200 queue
GPU_LIST = [g.strip() for g in GPU_TYPE.split(",") if g.strip()]
def gpu_spec(n: int):
    return [f"{g}:{n}" for g in GPU_LIST] if len(GPU_LIST) > 1 else f"{GPU_LIST[0]}:{n}"
REPO_LOCAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_REMOTE = "/root/easyNLA"
Q36 = f"{REPO_REMOTE}/nlt/q36"
IGNORE = [".git", ".venv", "__pycache__", "*.pyc", "*.parquet", "*.log", "**/logs/**", "logs", "*.npy", "*.pt", "*.jsonl", "*.out", "wandb"]
ENV = {"HF_HOME": "/vol_data/hf_cache", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "PYTHONPATH": f"{Q36}:{REPO_REMOTE}"}

image_hf = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch==2.8.0", "transformers==5.5.4", "peft==0.19.1", "accelerate", "safetensors", "sentencepiece", "pyarrow", "numpy", "pandas",
                 "wandb", "einops", "scipy", "pyyaml", "huggingface_hub[hf_transfer]", "flash-linear-attention", "matplotlib")
    .env(ENV)
    .add_local_dir(REPO_LOCAL, REPO_REMOTE, copy=False, ignore=IGNORE)
)
image_vllm = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system --python $(which python) "
        "'vllm==0.21.0' 'vllm-lens @ git+https://github.com/ceselder/vllm-metamodels@6a0a0e56a81d4c234346575238ee5fe8f580f332' 'transformers==5.5.4' "
        "peft==0.19.1 wandb accelerate pyarrow pandas numpy 'huggingface_hub[hf_transfer]' safetensors sentencepiece "
        "protobuf pyyaml tqdm flash-linear-attention scipy einops"
    )
    .env({**ENV, "VLLM_ATTENTION_BACKEND": "FLASH_ATTN", "VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_USE_DEEP_GEMM": "0",
          "VLLM_DEEP_GEMM_WARMUP": "skip", "VLLM_ALLOW_INSECURE_SERIALIZATION": "1", "VLLM_LENS_CUDA_GRAPHS": "1"})
    .add_local_dir(REPO_LOCAL, REPO_REMOTE, copy=False, ignore=IGNORE)
)

app = modal.App(APP_NAME)
vol = modal.Volume.from_name("nlt", create_if_missing=True)
VOLS = {"/vol": vol, "/vol_go": modal.Volume.from_name("grad-olens").read_only(), "/vol_ol1": modal.Volume.from_name("olens-1layer").read_only(),
        "/vol_data": modal.Volume.from_name("olens-new-arch").read_only()}
SECRETS = [modal.Secret.from_name("nla-exp-secrets")]


def _run(cmd, tag):
    import threading, time as _t
    os.makedirs("/vol/q36/logs", exist_ok=True)
    logf = f"/vol/q36/logs/{tag}_{_t.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.log"
    print("[run]", cmd, "| tee", logf, flush=True)
    def _bg():
        while True:
            _t.sleep(600)
            try: vol.commit()
            except Exception as e: print(f"[run] bg commit failed: {e}", flush=True)
    threading.Thread(target=_bg, daemon=True).start()
    p = subprocess.run(f"set -o pipefail; {cmd} 2>&1 | tee {logf}", shell=True, executable="/bin/bash", cwd=Q36, env=dict(os.environ))
    vol.commit()
    print(f"[run] exit {p.returncode}", flush=True)
    return p.returncode


@app.function(image=image_hf, gpu=gpu_spec(1), volumes=VOLS, timeout=23 * 60 * 60, secrets=SECRETS, cpu=16, memory=128 * 1024)
def run_hf(script: str, args: str = "", nproc: int = 1):
    cmd = (f"torchrun --nproc_per_node {nproc} {Q36}/{script} {args}" if nproc > 1 else f"python {Q36}/{script} {args}")
    return _run(cmd, script.replace(".py", ""))


@app.function(image=image_hf, gpu=gpu_spec(1), volumes=VOLS, timeout=23 * 60 * 60, secrets=SECRETS, cpu=16, memory=200 * 1024)
def run_mod(module: str, args: str = ""):
    """python -m <module> from the repo root (critic trainer / bits runner)"""
    return _run(f"cd {REPO_REMOTE} && python -m {module} {args}", module.split(".")[-1])


@app.function(image=image_vllm, gpu=gpu_spec(1), volumes=VOLS, timeout=23 * 60 * 60, secrets=SECRETS, cpu=16, memory=128 * 1024)
def run_vllm(script: str, args: str = ""):
    return _run(f"python {Q36}/{script} {args}", script.replace(".py", ""))


@app.function(image=image_hf, volumes=VOLS, timeout=6 * 60 * 60, cpu=8.0, memory=96 * 1024, secrets=SECRETS)
def run_cpu(script: str, args: str = ""):
    return _run(f"python {Q36}/{script} {args}", script.replace(".py", ""))


@app.function(image=image_hf, volumes=VOLS, timeout=2 * 60 * 60, cpu=4.0, memory=32 * 1024, secrets=SECRETS)
def pyrun(code: str):
    import textwrap
    open("/tmp/snippet.py", "w").write(textwrap.dedent(code))
    return _run("python /tmp/snippet.py", "pyrun")


@app.local_entrypoint()
def main(task: str, script: str = "", args: str = "", code: str = "", gpus: int = 1, nproc: int = 0, module: str = ""):
    if task in ("hf", "hf-many"):
        fn = run_hf.with_options(gpu=gpu_spec(gpus))
        for a in ([args] if task == "hf" else [x.strip() for x in args.split(";;") if x.strip()]):
            h = fn.spawn(script, a, nproc or gpus); print(f"SPAWNED {h.object_id} :: {a[:90]}", flush=True)
    elif task in ("vllm", "vllm-many"):
        fn = run_vllm.with_options(gpu=gpu_spec(gpus))
        for a in ([args] if task == "vllm" else [x.strip() for x in args.split(";;") if x.strip()]):
            h = fn.spawn(script, a); print(f"SPAWNED {h.object_id} :: {a[:90]}", flush=True)
    elif task in ("mod", "mod-many"):
        fn = run_mod.with_options(gpu=gpu_spec(gpus))
        for a in ([args] if task == "mod" else [x.strip() for x in args.split(";;") if x.strip()]):
            h = fn.spawn(module, a); print(f"SPAWNED {h.object_id} :: {a[:90]}", flush=True)
    elif task == "cpu":
        h = run_cpu.spawn(script, args); print(f"SPAWNED {h.object_id} :: {args[:90]}", flush=True)
    elif task == "pyrun":
        h = pyrun.spawn(code); print(f"SPAWNED {h.object_id}", flush=True)
    else:
        raise SystemExit(f"unknown task {task}")
