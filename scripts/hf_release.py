"""HF release of tonight's NLT artifacts as PRIVATE repos under `ceselder` (Modal app `nlt-packager`, volume `nlt`, HF token from the
Modal secret `nla-exp-secrets`; nothing is ever made public here).

  modal run scripts/hf_release.py --task plan                       # list every file the manifest would upload, with sizes (no network to HF)
  modal run scripts/hf_release.py --task upload --dry-run 1         # create nothing, print the upload plan incl. generated cards
  modal run scripts/hf_release.py --task upload [--only nlt-qwen3-8b-lenses]

Manifest: scripts/hf_release_manifest.json (repo_id -> type, files/folders on the volume, dataset parquet builders, card template).
Cards: scripts/hf_cards/<name>.md with {placeholders} filled from the manifest's `card_vars` and from the report's data/*.json (mounted at /report_data).
Datasets always ship parquet (user rule). Repos are created with private=True and exist_ok=True; re-running only re-uploads changed files.
"""
import json, os, sys
import modal

for _p in (os.path.dirname(os.path.abspath(__file__)), os.path.join(os.environ.get("PYTHONPATH", "/root/easyNLA").split(":")[0], "scripts")):
    if os.path.isdir(_p) and _p not in sys.path: sys.path.insert(0, _p)

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DATA_LOCAL = os.path.expanduser("~/shared/reports/natural-language-transcoder/data")
vol = modal.Volume.from_name("nlt", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub[hf_xet]>=0.34", "pyarrow", "pandas", "numpy")
         .add_local_dir(HERE, "/root/scripts", copy=False, ignore=["__pycache__", "*.pyc"])
         .add_local_dir(REPORT_DATA_LOCAL, "/report_data", copy=False, ignore=["*.png", "*.pdf"]))
app = modal.App(os.environ.get("NLT_APP", "nlt-packager"), image=image)
SECRETS = [modal.Secret.from_name("nla-exp-secrets")]
OWNER = "ceselder"


def _load_manifest():
    return json.load(open("/root/scripts/hf_release_manifest.json"))


def _fill(template: str, vars_: dict) -> str:
    out = template
    for k, v in vars_.items(): out = out.replace("{" + k + "}", str(v))
    return out


def _card_vars(repo: dict) -> dict:
    """numbers for the cards: manifest card_vars + a few read from the report jsons when present"""
    v = dict(repo.get("card_vars", {}))
    try:
        j = json.load(open("/report_data/lens_quality.json"))
        v.setdefault("lens_quality_note", json.dumps({k: j[k] for k in list(j)[:3]})[:400])
    except Exception: pass
    try:
        j = json.load(open("/report_data/trunk_results.json")); v.setdefault("trunk_verdict", j.get("verdict", ""))
    except Exception: pass
    return v


def _build_dataset_files(repo: dict, work: str) -> list:
    """dataset builders -> list of (local_path, path_in_repo); every dataset gets parquet"""
    import glob, shutil
    import pyarrow.parquet as pq, pandas as pd
    out = []
    for b in repo.get("builders", []):
        kind = b["kind"]
        if kind == "copy_parquet":                      # copy parquet files as they are (globs on the volume)
            for pat in b["src"]:
                for f in sorted(glob.glob(pat)):
                    dst = os.path.join(work, b["dst"], os.path.basename(f)); os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(f, dst); out.append((dst, os.path.relpath(dst, work)))
        elif kind == "fixed_pairs_subset":              # rows of text parquets restricted to the first N pairs of pairs_val (the fixed eval set)
            pairs = pq.read_table(b["pairs"]).to_pandas().iloc[: b.get("n", 4096)]
            pids = set(pairs["pair_id"].tolist())
            dst = os.path.join(work, b["dst"], "pairs_val_fixed.parquet"); os.makedirs(os.path.dirname(dst), exist_ok=True); pairs.to_parquet(dst, index=False); out.append((dst, os.path.relpath(dst, work)))
            for pat in b.get("src", []):
                for f in sorted(glob.glob(pat)):
                    df = pq.read_table(f).to_pandas(); df = df[df["pair_id"].isin(pids)]
                    if len(df) == 0: continue
                    name = b.get("rename", {}).get(os.path.basename(f), os.path.basename(os.path.dirname(os.path.dirname(f))) + "_" + os.path.basename(f))
                    d2 = os.path.join(work, b["dst"], name); df.to_parquet(d2, index=False); out.append((d2, os.path.relpath(d2, work)))
        elif kind == "copy_tree":                        # any files (parquet/json) under a volume dir, flattened by relative path
            for pat in b["src"]:
                for f in sorted(glob.glob(pat, recursive=True)):
                    if os.path.isdir(f): continue
                    rel = os.path.relpath(f, b["root"]) if b.get("root") else os.path.basename(f)
                    dst = os.path.join(work, b["dst"], rel); os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(f, dst); out.append((dst, os.path.relpath(dst, work)))
        elif kind == "report_data":                      # the report's data/*.json (mounted)
            for f in sorted(glob.glob("/report_data/*.json")):
                dst = os.path.join(work, b["dst"], os.path.basename(f)); os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(f, dst); out.append((dst, os.path.relpath(dst, work)))
        else: raise ValueError(kind)
    return out


def _plan_repo(repo: dict, work: str):
    """-> list of (local_path, path_in_repo) for files + folders + built datasets (no card)"""
    import glob
    items = []
    for f in repo.get("files", []):
        for src in sorted(glob.glob(f["src"])):
            items.append((src, os.path.join(f.get("dst", ""), os.path.basename(src)) if f.get("dst") is not None else os.path.basename(src)))
    for fo in repo.get("folders", []):
        for root, _, fs in os.walk(fo["src"]):
            for fn in fs:
                p = os.path.join(root, fn); items.append((p, os.path.join(fo["dst"], os.path.relpath(p, fo["src"]))))
    items += _build_dataset_files(repo, work)
    return items


@app.function(timeout=6 * 3600, volumes={"/vol": vol}, secrets=SECRETS, cpu=4, memory=32 * 1024, ephemeral_disk=256 * 1024)
def run(task: str = "plan", only: str = "", dry_run: int = 1):
    import shutil, tempfile
    vol.reload(); man = _load_manifest(); total = 0
    for repo in man["repos"]:
        if only and repo["name"] != only: continue
        repo_id = f"{OWNER}/{repo['name']}"; work = tempfile.mkdtemp(prefix="hfrel_")
        items = _plan_repo(repo, work)
        card_t = open(os.path.join("/root/scripts/hf_cards", repo["card"])).read(); card = _fill(card_t, _card_vars(repo))
        sizes = [(os.path.getsize(p) if os.path.exists(p) else -1) for p, _ in items]; total += sum(s for s in sizes if s > 0)
        print(f"\n=== {repo_id} ({repo['type']}, private): {len(items)} files, {sum(s for s in sizes if s > 0)/1e9:.2f} GB", flush=True)
        for (p, d), s in zip(items, sizes): print(f"   {s/1e6:9.1f} MB  {d}   <- {p}" + ("  MISSING" if s < 0 else ""), flush=True)
        print("   --- README.md (first 600 chars) ---\n" + card[:600].replace("\n", "\n   "), flush=True)
        if task == "upload" and not dry_run:
            from huggingface_hub import HfApi
            api = HfApi(token=os.environ["HF_TOKEN"])
            api.create_repo(repo_id, repo_type=repo["type"], private=True, exist_ok=True)
            open(os.path.join(work, "README.md"), "w").write(card)
            api.upload_file(path_or_fileobj=os.path.join(work, "README.md"), path_in_repo="README.md", repo_id=repo_id, repo_type=repo["type"])
            for (p, d), s in zip(items, sizes):
                if s < 0: print("   skip missing", p, flush=True); continue
                api.upload_file(path_or_fileobj=p, path_in_repo=d, repo_id=repo_id, repo_type=repo["type"]); print("   uploaded", d, flush=True)
            info = api.repo_info(repo_id, repo_type=repo["type"]); assert info.private, f"{repo_id} is not private!"
            print(f"   DONE {repo_id} private={info.private}", flush=True)
        shutil.rmtree(work, ignore_errors=True)
    print(f"\nTOTAL {total/1e9:.2f} GB across repos", flush=True)


@app.local_entrypoint()
def main(task: str = "plan", only: str = "", dry_run: int = 1):
    run.remote(task, only, dry_run)
