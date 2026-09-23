"""Modal app `nlt-rl` (volume `nlt`): two-marker verbalizer checks, SFT, the exact-bits RL trainer and the step-0 test.

  modal run --detach scripts/modal_nlt_rl.py --task check --tag ao   --extra "--init ao --n-pairs 16 --group 4"
  modal run --detach scripts/modal_nlt_rl.py --task sft   --tag v0   --extra "--text /vol/z/pool_v1/train.parquet --init ao"
  modal run --detach scripts/modal_nlt_rl.py --task step0 --tag ao   --extra "--init ao --n-pairs 512 --group 8 --critic /vol/critic/text_v1/ckpt_latest.pt"
  NLT_RL_GPUS=2 modal run --detach scripts/modal_nlt_rl.py --task rl --tag smoke --extra "--stub-critic --steps 5"
Outputs under /vol/rl/<task>/<tag>/. HF cache shared with the team at /vol/hf_cache (HF_HUB_DISABLE_XET=1: the xet path burns the
org's API quota). Wandb octahedral-systems/nlt-qwen3-8b.
"""
import os, sys
import modal
for _p in (os.path.dirname(os.path.abspath(__file__)), os.path.join(os.environ.get("PYTHONPATH", "/root/easyNLA").split(":")[0], "scripts")):
    if os.path.isdir(_p) and _p not in sys.path: sys.path.insert(0, _p)
from modal_nla_exp import image_base, REPO_LOCAL, REPO_REMOTE, REPO_IGNORE  # noqa: E402

vol = modal.Volume.from_name("nlt", create_if_missing=True)
SECRETS = [modal.Secret.from_name("nla-exp-secrets")]
DATA = os.environ.get("NLT_DATA", "/vol/data/qwen3_8b")
IGNORE = list(REPO_IGNORE) + ["*.log", "**/logs/**", "logs", "*.npy", "*.pt", "*.jsonl", "*.out", "wandb", "*.safetensors"]
image = image_base.add_local_dir(REPO_LOCAL, REPO_REMOTE, copy=False, ignore=IGNORE)
app = modal.App(os.environ.get("NLT_APP", "nlt-rl"), image=image)
GPU = os.environ.get("NLT_GPU", "B200")
NGPU = int(os.environ.get("NLT_RL_GPUS", "1"))
COMMON = dict(volumes={"/vol": vol}, secrets=SECRETS, cpu=16, memory=256 * 1024, ephemeral_disk=512 * 1024)


def _prep():
    """per-container: shared HF cache, idempotent vllm-lens patch (site-packages is ephemeral), decode CUDA graphs on the fork."""
    import subprocess
    os.chdir(REPO_REMOTE)
    os.environ["HF_HOME"] = "/vol/hf_cache"; os.environ["HF_HUB_DISABLE_XET"] = "1"; os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
    os.environ.setdefault("WANDB_DIR", "/root/wandb"); os.makedirs("/root/wandb", exist_ok=True)
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    try: vol.reload()
    except Exception as e: print(f"[prep] vol.reload skipped: {e}", flush=True)
    import importlib.util
    spec = importlib.util.find_spec("vllm_lens._worker_ext")
    is_fork = bool(spec and spec.origin and "_apply_layer_vectorized" in open(spec.origin).read())
    patcher = "utils/patch_vllm_metamodel.py" if is_fork else "utils/patch_vllm_lens.py"
    r = subprocess.run([sys.executable, patcher], capture_output=True, text=True)
    print(f"[prep] {patcher} rc={r.returncode}: {(r.stdout + r.stderr).strip()[-300:]}", flush=True)
    if r.returncode != 0: raise SystemExit("vllm-lens patch failed")
    if is_fork:
        os.environ["NLA_ALLOW_STALE_LENS"] = "1"
        if os.environ.get("NLA_VLLM_GRAPHS", "1") == "1":
            os.environ["VLLM_LENS_CUDA_GRAPHS"] = "1"; os.environ["NLA_VLLM_EAGER"] = "0"
    print(f"[prep] fork={is_fork} eager={os.environ.get('NLA_VLLM_EAGER', '1')}", flush=True)


def _run(cmd):
    import subprocess
    _prep()
    print("[modal] " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=REPO_REMOTE)
    try: vol.commit()
    except Exception as e: print(f"[modal] vol.commit: {e}", flush=True)
    return rc


@app.function(gpu=f"{GPU}:1", timeout=4 * 3600, **COMMON)
def check(tag: str, extra: str = "", data: str = DATA):
    return _run([sys.executable, "-m", "nlt.verbalizer.check_injection", "--data-dir", data, "--out", f"/vol/rl/check/{tag}.json"] + extra.split())


@app.function(gpu=f"{GPU}:{NGPU}", timeout=12 * 3600, **COMMON)
def sft(tag: str, extra: str = "", data: str = DATA):
    return _run([sys.executable, "-m", "nlt.verbalizer.sft", "--data-dir", data, "--out", f"/vol/rl/sft/{tag}", "--tag", tag] + extra.split())


@app.function(gpu=f"{GPU}:{NGPU}", timeout=23 * 3600, **COMMON)
def rl(tag: str, extra: str = "", data: str = DATA):
    return _run([sys.executable, "-m", "nlt.rl.train", "--data-dir", data, "--out", f"/vol/rl/runs/{tag}", "--tag", tag] + extra.split())


@app.function(gpu=f"{GPU}:{NGPU}", timeout=8 * 3600, **COMMON)
def step0(tag: str, extra: str = "", data: str = DATA):
    return _run([sys.executable, "-m", "nlt.rl.step0", "--data-dir", data, "--out", f"/vol/rl/step0/{tag}.json", "--tag", tag] + extra.split())


@app.function(gpu=f"{GPU}:1", timeout=4 * 3600, **COMMON)
def dump(tag: str, extra: str = "", data: str = DATA):
    """rollouts of a policy on the fixed eval pairs in the board #31 text format -> /vol/z/<tag>/val/part_0000000_0004096.parquet"""
    return _run([sys.executable, "-m", "nlt.rl.dump_rollouts", "--data-dir", data, "--source", tag, "--out", f"/vol/z/{tag}/val/part_0000000_0004096.parquet"] + extra.split())


@app.function(timeout=1800, volumes={"/vol": vol}, cpu=2, memory=8 * 1024)
def cat(path: str):
    vol.reload(); print(open(f"/vol/{path}").read())


@app.local_entrypoint()
def main(task: str = "check", tag: str = "dev", extra: str = "", data: str = DATA, path: str = ""):
    fn = {"check": check, "sft": sft, "rl": rl, "step0": step0, "dump": dump}.get(task)
    if task == "cat": cat.remote(path); return
    if fn is None: raise SystemExit(f"unknown task {task}")
    print("rc", fn.remote(tag, extra, data)); print("done.")
