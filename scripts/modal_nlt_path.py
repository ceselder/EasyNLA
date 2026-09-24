"""Modal app `nlt-path` (volume `nlt`): PATH verbalizer (DECISIONS v1.26) -- write extraction, controlled SFT, HF-generate dumps.

  modal run --detach scripts/modal_nlt_path.py --task extract --tag v0b_train --extra "--split train --rows /vol/z/v0b_mix/train/rows.parquet"
  modal run --detach scripts/modal_nlt_path.py --task extract --tag val      --extra "--split val --rows /vol/z/v0b_mix/val/rows.parquet --pairs-head 4096"
  modal run --detach scripts/modal_nlt_path.py --task sft  --tag v0b_path_d --extra "--path-mode delta --text ... --val-text ... --init lora:/vol/rl/sft/v0_ao_tsv1/lora"
  modal run --detach scripts/modal_nlt_path.py --task dump --tag v0b_path_d_0 --extra "--path-mode delta --init lora:/vol/rl/sft/v0b_path_d/lora"
Outputs: /vol/path/qwen3_8b/<split>/ (extract), /vol/rl/sft/<tag>/ (sft), /vol/z/<tag>/val/part_0000000_0004096.parquet (dump).
No vLLM anywhere in this app (plain HF hooks + generate). HF cache /vol/hf_cache (XET off). GPU: NLT_GPU (default H100).
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
app = modal.App(os.environ.get("NLT_APP", "nlt-path"), image=image)
GPU = os.environ.get("NLT_GPU", "H100")
COMMON = dict(volumes={"/vol": vol}, secrets=SECRETS, cpu=16, memory=256 * 1024, ephemeral_disk=512 * 1024)


def _run(cmd):
    import subprocess
    os.chdir(REPO_REMOTE)
    os.environ["HF_HOME"] = "/vol/hf_cache"; os.environ["HF_HUB_DISABLE_XET"] = "1"; os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
    os.environ.setdefault("WANDB_DIR", "/root/wandb"); os.makedirs("/root/wandb", exist_ok=True)
    try: vol.reload()
    except Exception as e: print(f"[prep] vol.reload skipped: {e}", flush=True)
    print("[modal] " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=REPO_REMOTE)
    try: vol.commit()
    except Exception as e: print(f"[modal] vol.commit: {e}", flush=True)
    print(f"[modal] rc={rc}", flush=True)
    return rc


@app.function(gpu=f"{GPU}:1", timeout=4 * 3600, **COMMON)
def extract(tag: str, extra: str = "", data: str = DATA):
    return _run([sys.executable, "-m", "nlt.path.extract", "--data-dir", data, "--out-dir", "/vol/path/qwen3_8b", "--tag", tag] + extra.split())


@app.function(gpu=f"{GPU}:1", timeout=8 * 3600, **COMMON)
def sft(tag: str, extra: str = "", data: str = DATA):
    return _run([sys.executable, "-m", "nlt.path.sft", "--data-dir", data, "--out", f"/vol/rl/sft/{tag}", "--tag", tag] + extra.split())


@app.function(gpu=f"{GPU}:1", timeout=4 * 3600, **COMMON)
def dump(tag: str, extra: str = "", data: str = DATA):
    return _run([sys.executable, "-m", "nlt.path.dump", "--data-dir", data, "--source", tag, "--out", f"/vol/z/{tag}/val/part_0000000_0004096.parquet"] + extra.split())


@app.function(timeout=3600, volumes={"/vol": vol}, secrets=SECRETS, cpu=8, memory=64 * 1024)
def facts(tag: str, extra: str = "", data: str = DATA):
    """CPU: path-dependent SFT targets from the extracted writes (nlt.path.facts) -> /vol/z/<tag>/<split>/rows.parquet"""
    return _run([sys.executable, "-m", "nlt.path.facts", "--data-dir", data, "--path-dir", "/vol/path/qwen3_8b"] + extra.split())


@app.function(timeout=1800, volumes={"/vol": vol}, cpu=2, memory=8 * 1024)
def cat(path: str):
    vol.reload(); print(open(f"/vol/{path}").read())


@app.local_entrypoint()
def main(task: str = "extract", tag: str = "dev", extra: str = "", data: str = DATA, path: str = ""):
    if task == "cat": cat.remote(path); return
    fn = {"extract": extract, "sft": sft, "dump": dump, "facts": facts}.get(task)
    if fn is None: raise SystemExit(f"unknown task {task}")
    rc = fn.remote(tag, extra, data); print(f"rc {rc}")
    if rc != 0: raise SystemExit(f"task {task} {tag} failed rc={rc}")
    print("done.")
