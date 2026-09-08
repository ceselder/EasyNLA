"""Upload checkpoint dirs from the Modal volumes to a HuggingFace repo (private), and inspect run metadata.
  modal run scripts/modal_hf_upload.py::inspect
  modal run scripts/modal_hf_upload.py::upload --repo ceselder/<name> --spec "<vol_dir>=<path_in_repo>,..."
  modal run scripts/modal_hf_upload.py::upload_files --repo ceselder/<name> --local-dir ./hf_readme   (README/scripts from this box)
"""
import os, sys
import modal
for _p in (os.path.dirname(os.path.abspath(__file__)), os.path.join(os.environ.get("PYTHONPATH", "/root/easyNLA").split(":")[0], "scripts")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
from modal_nla_exp import image_base, VOLS, SECRETS, REPO_LOCAL, REPO_REMOTE, REPO_IGNORE  # noqa: E402

image = image_base.add_local_dir(REPO_LOCAL, REPO_REMOTE, copy=False, ignore=REPO_IGNORE)
app = modal.App("nla-hf-upload", image=image)


@app.function(volumes=VOLS, secrets=SECRETS, timeout=600, cpu=2)
def inspect(parquet: str = "/vol_q36/data/sft/av_sft_val.parquet", dirs: str = ""):
    import json, glob
    import pyarrow.parquet as pq
    t = pq.ParquetFile(parquet).read_row_group(0, columns=["prompt", "response"]).slice(0, 1)
    print("PROMPT ROW 0:", json.dumps(t.column("prompt").to_pylist()[0], ensure_ascii=False)[:1500])
    print("RESPONSE ROW 0:", t.column("response").to_pylist()[0][:600])
    for d in dirs.split(","):
        if not d:
            continue
        tot = sum(os.path.getsize(f) for f in glob.glob(f"{d}/**/*", recursive=True) if os.path.isfile(f))
        print(f"DIR {d}: {tot/1e9:.2f} GB, files: {sorted(os.listdir(d))}")
        for fn in ("adapter_config.json", "run_config.yaml", "nla_meta.yaml", "ar_meta.json", "saved_at_step.txt"):
            p = os.path.join(d, fn)
            if os.path.exists(p):
                print(f"--- {p}\n{open(p).read()[:2500]}")


@app.function(volumes=VOLS, secrets=SECRETS, timeout=4 * 3600, cpu=4)
def upload(repo: str, spec: str, private: bool = True):
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo, private=private, exist_ok=True)
    for item in spec.split(","):
        src, dst = item.split("=")
        print(f"[upload] {src} -> {repo}/{dst}", flush=True)
        api.upload_folder(repo_id=repo, folder_path=src, path_in_repo=dst, commit_message=f"add {dst}",
                          ignore_patterns=["reference/**", "optim*", "*.tmp", "*.old/**", "README.md"])   # reference/ = frozen SFT adapter copy (KL term); PEFT README has an invalid base_model path
        print(f"[upload] done {dst}", flush=True)


@app.function(secrets=SECRETS, timeout=1800, cpu=2)
def upload_files(repo: str, files: dict):
    """files: {path_in_repo: text content} (small text files rendered on this box)."""
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    for dst, content in files.items():
        api.upload_file(repo_id=repo, path_in_repo=dst, path_or_fileobj=content.encode(), commit_message=f"add {dst}")
        print(f"[upload] {dst} ({len(content)} bytes)", flush=True)


@app.local_entrypoint()
def main(task: str = "inspect", repo: str = "", spec: str = "", dirs: str = "", local_dir: str = ""):
    if task == "inspect":
        inspect.remote(dirs=dirs)
    elif task == "upload":
        upload.remote(repo, spec)
    elif task == "run_script":
        run_script.remote(open(local_dir).read(), spec, baseline_parquet=dirs)
    elif task == "upload_files":
        files = {}
        for root, _, fns in os.walk(local_dir):
            for fn in fns:
                p = os.path.join(root, fn); files[os.path.relpath(p, local_dir)] = open(p).read()
        upload_files.remote(repo, files)


@app.function(gpu="B200", volumes=VOLS, secrets=SECRETS, timeout=3600, cpu=8)
def run_script(script: str, args: str, baseline_parquet: str = ""):
    """Run a standalone script (text passed in) on a B200 with the volumes mounted; optionally print the predict-mean
    baseline MSE of a validation parquet (for the FVE-equiv constant in the README)."""
    import subprocess, sys
    if baseline_parquet:
        import numpy as np, pyarrow.parquet as pq, torch
        from nla.schema import compute_predict_mean_baselines
        t = pq.ParquetFile(baseline_parquet).read(columns=["activation_vector"]).slice(0, 1024)
        ac = np.asarray(t.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(t.num_rows, -1)
        b = compute_predict_mean_baselines(torch.tensor(ac), float(ac.shape[1]) ** 0.5)
        print(f"BASELINE predict-mean MSE (1024 val rows, mse_scale sqrt(d)): meannorm {b[0]:.4f} raw {b[1]:.4f}", flush=True)
    open("/root/script.py", "w").write(script)
    subprocess.run([sys.executable, "/root/script.py"] + args.split("|"), check=False)
