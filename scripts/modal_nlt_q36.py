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
    .pip_install("triton>=3.7.1")            # fla refuses the gated-DeltaNet BACKWARD on Hopper (H100/H200) with triton 3.4-3.7.0 (fla #640); torch 2.8 + triton 3.7.1 validated on the H100 playground box
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


@app.function(image=image_hf, gpu=gpu_spec(1), volumes=VOLS, timeout=23 * 60 * 60, secrets=SECRETS, cpu=16, memory=128 * 1024)
def run_evalq(idle_min: int = 20, worker: str = "w0"):
    """EVAL QUEUE WORKER (orchestrator 09:12: one long-lived app instead of an app per eval; the workspace has a 100-ephemeral-app limit shared with other users).
    Loops over /vol/q36/evalq/<prio>_<time>_<label>.json specs {label, args, script?}: claims the first (priority order = filename order) by renaming it, runs the script
    (default eval_bits.py) with _run (log tee'd to /vol/q36/logs, volume committed at the end), marks it .done<rc>.json, repeats; exits after idle_min minutes without work."""
    import glob, json, time as _t
    q = "/vol/q36/evalq"; os.makedirs(q, exist_ok=True); idle = 0.0; n = 0; t_up = _t.time()
    print(f"[evalq] worker {worker} up; queue {q}", flush=True)
    while idle < idle_min * 60:
        try: vol.reload()
        except Exception as e: print(f"[evalq] reload failed: {e}", flush=True)
        # GRACEFUL RECYCLE: a worker runs the code snapshot of its launch; when the local code changes (eval_bits / critic_data), `touch` /vol/q36/evalq/.code_epoch and every worker exits
        # BETWEEN jobs (no job is interrupted) - the eval_workers.sh manager respawns fresh workers. (11:20: the v5 train probe needs eval_bits --pair-ids-file, which the 10:43 workers do not have.)
        ep = f"{q}/.code_epoch"
        if os.path.exists(ep) and os.path.getmtime(ep) > t_up: print(f"[evalq] worker {worker}: code epoch newer than my launch -> exit after {n} jobs for a fresh worker", flush=True); return
        specs = sorted(f for f in glob.glob(f"{q}/*.json") if ".running" not in f and ".done" not in f)
        if not specs: _t.sleep(60); idle += 60; continue
        spec = specs[0]; claimed = spec[:-5] + f".running.{worker}.json"
        try: os.rename(spec, claimed); vol.commit()
        except Exception as e: print(f"[evalq] claim of {os.path.basename(spec)} failed ({e}); retrying", flush=True); _t.sleep(5); continue
        # CLAIM RACE (12:20: two workers renamed the same spec inside their own volume views and both ran it): after the commit, reload and let the lexicographically first claimant keep it
        try: vol.reload()
        except Exception: pass
        others = sorted(glob.glob(spec[:-5] + ".running.*.json"))
        if len(others) > 1 and others[0] != claimed:
            print(f"[evalq] {worker} lost the claim race for {os.path.basename(spec)} to {os.path.basename(others[0])}; releasing mine", flush=True)
            try: os.remove(claimed); vol.commit()
            except Exception as e: print(f"[evalq] release failed: {e}", flush=True)
            _t.sleep(3); continue
        job = json.load(open(claimed)); idle = 0.0; n += 1
        print(f"[evalq] {worker} job {n}: {job['label']} :: {job['args'][:160]}", flush=True)
        import re as _re; m_out = _re.search(r"--out (\S+)", job["args"]); outp = m_out.group(1) if m_out else None
        if outp and outp.endswith(".json") and os.path.exists(outp) and '"elapsed_min"' in open(outp).read():          # idempotent: a complete result already exists (duplicate spec) -> skip
            print(f"[evalq] {job['label']}: result {outp} already complete -> skipped", flush=True); rc = 0
        else:
            rc = _run(f"python {Q36}/{job.get('script', 'eval_bits.py')} {job['args']}", f"evalq_{job['label']}")
        try: os.rename(claimed, spec[:-5] + f".done{rc}.json"); vol.commit()
        except Exception as e: print(f"[evalq] done-mark failed: {e}", flush=True)
        print(f"[evalq] {worker} finished {job['label']} rc {rc}", flush=True)
    print(f"[evalq] worker {worker} idle {idle_min} min after {n} jobs -> exit", flush=True)


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
    elif task == "evalq":
        fn = run_evalq.with_options(gpu=gpu_spec(1)); h = fn.spawn(int(args or 20), module or "w0"); print(f"SPAWNED {h.object_id} :: evalq worker {module or 'w0'} idle {args or 20} min", flush=True)
    else:
        raise SystemExit(f"unknown task {task}")
