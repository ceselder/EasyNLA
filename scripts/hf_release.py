"""HF release of tonight's NLT artifacts as PRIVATE repos under `ceselder` (Modal app `nlt-packager`, volume `nlt`, HF token from the
Modal secret `nla-exp-secrets`; nothing is ever made public here).

  modal run scripts/hf_release.py --task plan                       # list every file the manifest would upload, with sizes (no network to HF)
  modal run scripts/hf_release.py --task upload --dry-run 1         # create nothing, print the upload plan incl. generated cards
  modal run scripts/hf_release.py --task upload [--only nlt-qwen3-8b-lenses]

Manifest: scripts/hf_release_manifest.json (repo_id -> type, files/folders on the volume, dataset parquet builders, card template).
Cards: scripts/hf_cards/<name>.md with {placeholders} filled from the manifest's `card_vars` and from the report's data/*.json (mounted at /report_data).
Datasets always ship parquet (user rule). Repos are created with private=True and exist_ok=True; re-running only re-uploads changed files.
"""
import json, os, re, sys
import modal

for _p in (os.path.dirname(os.path.abspath(__file__)), os.path.join(os.environ.get("PYTHONPATH", "/root/easyNLA").split(":")[0], "scripts")):
    if os.path.isdir(_p) and _p not in sys.path: sys.path.insert(0, _p)

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DATA_LOCAL = os.path.expanduser("~/shared/reports/natural-language-transcoder/data")
vol = modal.Volume.from_name("nlt", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub[hf_xet]>=0.34", "pyarrow", "pandas", "numpy")
         .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")                      # to strip optimizer states from critic checkpoints
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


def _build_dataset_files(repo: dict, work: str, plan_only: bool = False) -> list:
    """builders -> list of (local_path, path_in_repo); every dataset gets parquet. plan_only skips heavy builders (reports the source instead)."""
    import glob, shutil
    import pyarrow.parquet as pq, pandas as pd
    out = []
    for b in repo.get("builders", []):
        kind = b["kind"]
        if kind == "strip_ckpt":                        # critic checkpoint without optimizer states (22-39 GB -> 4-13 GB); keeps model/args/config/step/d_enc
            for src in b["src"]:
                if src.startswith("BEST:"):             # the run's exact-selected checkpoint, named inside <dir>/BEST.txt (a path; fallback: 'BEST:<file>|<fallback ckpt>' or ckpt_latest.pt, with a note)
                    bf, _, fb = src[5:].partition("|"); d_ = os.path.dirname(bf)
                    toks = [t for t in re.split(r"[\s,;:'\"]+", open(bf).read()) if t.endswith(".pt")] if os.path.exists(bf) else []
                    if toks:
                        named = toks[0]; src = named if os.path.isabs(named) else os.path.join(d_, named)
                        if not os.path.exists(src): print(f"   NOTE {bf} names {named} which does not exist -> skipped", flush=True); src = src + ".MISSING"
                    else:
                        src = fb or os.path.join(d_, "ckpt_latest.pt"); print(f"   NOTE {bf} missing -> using {os.path.basename(src)} (manifest fallback)", flush=True)
                dst = os.path.join(work, b["dst"], os.path.basename(os.path.dirname(src)), os.path.basename(src)); os.makedirs(os.path.dirname(dst), exist_ok=True)
                if plan_only or not os.path.exists(src):
                    out.append((src if os.path.exists(src) else src + ".MISSING", os.path.relpath(dst, work))); continue
                import torch
                ck = torch.load(src, map_location="cpu", mmap=True, weights_only=False)
                torch.save({k: ck[k] for k in ck if k != "opt"}, dst); del ck; out.append((dst, os.path.relpath(dst, work)))
                for extra in b.get("sidecars", []):
                    e = os.path.join(os.path.dirname(src), extra)
                    if os.path.exists(e): d2 = os.path.join(os.path.dirname(dst), extra); shutil.copy(e, d2); out.append((d2, os.path.relpath(d2, work)))
        elif kind == "copy_parquet":                      # copy parquet files as they are (globs on the volume)
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


def _plan_repo(repo: dict, work: str, plan_only: bool = False):
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
    for fg in repo.get("folder_globs", []):            # late additions whose exact run names are not known in advance: <run>/lora -> dst_fmt.format(run=<run dir name>)
        for src in sorted(glob.glob(fg["src_glob"])):
            run = os.path.basename(os.path.dirname(src.rstrip("/"))); dst = fg["dst_fmt"].format(run=run)
            for root, _, fs in os.walk(src):
                for fn in fs:
                    p = os.path.join(root, fn); items.append((p, os.path.join(dst, os.path.relpath(p, src))))
    for fg in repo.get("file_globs", []):
        for src in sorted(glob.glob(fg["src_glob"])):
            run = os.path.basename(os.path.dirname(src)); items.append((src, os.path.join(fg["dst_fmt"].format(run=run), os.path.basename(src))))
    seen = set(); items = [it for it in items if not (it[1] in seen or seen.add(it[1]))]
    items += _build_dataset_files(repo, work, plan_only=plan_only)
    return items


@app.function(timeout=6 * 3600, volumes={"/vol": vol}, secrets=SECRETS, cpu=8, memory=160 * 1024, ephemeral_disk=512 * 1024)
def run(task: str = "plan", only: str = "", dry_run: int = 1):
    import shutil, tempfile
    vol.reload(); man = _load_manifest(); total = 0
    for repo in man["repos"]:
        if only and repo["name"] != only: continue
        repo_id = f"{OWNER}/{repo['name']}"; work = tempfile.mkdtemp(prefix="hfrel_")
        items = _plan_repo(repo, work, plan_only=(task == "plan" or bool(dry_run)))
        card_t = open(os.path.join("/root/scripts/hf_cards", repo["card"])).read(); card = _fill(card_t, _card_vars(repo))
        sizes = [(os.path.getsize(p) if os.path.exists(p) else -1) for p, _ in items]; total += sum(s for s in sizes if s > 0)
        for (p, d), s_ in zip(items, sizes):
            if p.endswith(".pt") and s_ > 0 and "/vol/critic/" in p: print(f"   (ckpt {d}: {s_/1e9:.1f} GB on disk incl. optimizer; shipped WITHOUT optimizer states)", flush=True)
        print(f"\n=== {repo_id} ({repo['type']}, private): {len(items)} files, {sum(s for s in sizes if s > 0)/1e9:.2f} GB", flush=True)
        for (p, d), s in zip(items, sizes): print(f"   {s/1e6:9.1f} MB  {d}   <- {p}" + ("  MISSING" if s < 0 else ""), flush=True)
        print("   --- README.md (first 600 chars) ---\n" + card[:600].replace("\n", "\n   "), flush=True)
        if task == "upload" and not dry_run:
            from huggingface_hub import HfApi
            api = HfApi(token=os.environ["HF_TOKEN"])
            api.create_repo(repo_id, repo_type=repo["type"], private=True, exist_ok=True)
            info0 = api.repo_info(repo_id, repo_type=repo["type"])
            if not info0.private:                                  # HARD RULE (board #557): never upload into a public repo; never flip visibility from here
                print(f"   REFUSING {repo_id}: it exists and is PUBLIC -- make it private first", flush=True); continue
            open(os.path.join(work, "README.md"), "w").write(card)
            api.upload_file(path_or_fileobj=os.path.join(work, "README.md"), path_in_repo="README.md", repo_id=repo_id, repo_type=repo["type"])
            n_ok = 0
            for (p, d), s in zip(items, sizes):
                if s < 0: print("   skip missing", p, flush=True); continue
                try:
                    api.upload_file(path_or_fileobj=p, path_in_repo=d, repo_id=repo_id, repo_type=repo["type"]); n_ok += 1; print(f"   uploaded {d} ({s/1e6:.0f} MB)", flush=True)
                except Exception as e:                                  # quota / transient: report and continue with the next file (priority order in the manifest)
                    print(f"   FAILED {d}: {str(e)[:300]}", flush=True)
            print(f"   {n_ok} files uploaded", flush=True)
            info = api.repo_info(repo_id, repo_type=repo["type"]); assert info.private, f"{repo_id} is not private!"
            print(f"   DONE {repo_id} private={info.private}", flush=True)
        shutil.rmtree(work, ignore_errors=True)
    print(f"\nTOTAL {total/1e9:.2f} GB across repos", flush=True)


@app.function(timeout=3600, volumes={"/vol": vol}, cpu=8, memory=160 * 1024, ephemeral_disk=512 * 1024)
def strip_test(src: str = "/vol/critic/text_union_pooled_n/ckpt_best.pt"):
    """exercise the strip_ckpt path on one checkpoint: load (mmap), drop 'opt', save to /tmp, report sizes + keys"""
    import time, torch
    vol.reload(); t0 = time.time()
    ck = torch.load(src, map_location="cpu", mmap=True, weights_only=False)
    keys = list(ck.keys()); slim = {k: ck[k] for k in ck if k != "opt"}
    torch.save(slim, "/tmp/strip_test.pt"); del ck, slim
    print(f"{src}: keys {keys}; on disk {os.path.getsize(src)/1e9:.2f} GB -> stripped {os.path.getsize('/tmp/strip_test.pt')/1e9:.2f} GB in {time.time()-t0:.0f}s", flush=True)
    ck2 = torch.load("/tmp/strip_test.pt", map_location="cpu", weights_only=False); print("reload ok:", list(ck2.keys()), "step", ck2.get("step"), flush=True)


@app.local_entrypoint()
def main(task: str = "plan", only: str = "", dry_run: int = 1, src: str = ""):
    if task == "strip-test": strip_test.remote(src or "/vol/critic/text_union_pooled_n/ckpt_best.pt")
    else: run.remote(task, only, dry_run)
