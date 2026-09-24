#!/usr/bin/env python3
"""Upload NLT verbalizer adapters (SFT + RL checkpoints) from the Modal volume `nlt` to a HF model repo.

Runs LOCALLY (needs `modal` CLI + a stored HF token in ~/.cache/huggingface/token; no key literals).
Each source becomes a subfolder of the repo:  <repo>/<name>/{adapter_config.json, adapter_model.safetensors, meta.json?}

  python3 scripts/hf_upload_ckpts.py --repo ceselder/nlt-qwen3-8b-verbalizer \
      v0_ao_tsv1=/rl/sft/v0_ao_tsv1/lora v0b_mix=/rl/sft/v0b_mix/lora ref_v1_step40=/rl/runs/ref_v1/step_00040

For an RL step dir the `lora/` subdir is uploaded plus `meta.json`; pass --with-critic to include the co-trained listener `critic.pt`.
"""
import argparse, glob, json, os, shutil, subprocess, sys, time

CARD = """---
base_model: Qwen/Qwen3-8B
library_name: peft
tags: [interpretability, natural-language-transcoder, activation-verbalizer, lora, qwen3]
---
# Natural-language transcoder verbalizer for Qwen3-8B (LoRA adapters)

Adapters that make Qwen3-8B verbalize what its own forward pass did between residual-stream layers i and j.
Two residual vectors (h_i, h_j at one token position) are norm-matched ADDED to the output of decoder block 1 at the two
marker tokens (` ?`, token id 937) of a constant prompt (no layer label is given):

    " ? \\n ? \\n<question>"   (chat-templated, thinking off; markers at positions 3 and 5)

The adapter is rsLoRA r=64, alpha=16 on q/k/v/o/gate/up/down of every block, initialised from the activation-oracle LoRA.
`disable_adapter()` therefore gives the untouched base model.

| subfolder | what |
|---|---|
{rows}

Each subfolder has a PEFT `adapter_config.json` + `adapter_model.safetensors`; RL checkpoints also carry `meta.json`
(step, reward config, listener critic path). See the report for the recipe, critics and numbers.

Uploaded {when} by the `rl` agent of the NLT overnight team.
"""

def sh(*cmd, **kw):
    r = subprocess.run(cmd, text=True, capture_output=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed rc={r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.stdout

def fetch(remote: str, local: str, with_critic: bool) -> str:
    """Download a lora dir (or an RL step dir) from volume `nlt` into `local`; return the folder to upload."""
    shutil.rmtree(local, ignore_errors=True); os.makedirs(local, exist_ok=True)
    for attempt in range(3):
        try:
            sh("modal", "volume", "get", "nlt", remote.rstrip("/"), local, "--force"); break
        except RuntimeError as e:
            if attempt == 2: raise
            print(f"  retry volume get ({e.args[0][:200]}...)", flush=True); time.sleep(20)
    cfgs = glob.glob(os.path.join(local, "**", "adapter_config.json"), recursive=True)
    if not cfgs: raise RuntimeError(f"no adapter_config.json under {local} after downloading {remote}")
    lora_dir = os.path.dirname(cfgs[0])
    up = os.path.join(local, "_upload"); os.makedirs(up, exist_ok=True)
    for f in ("adapter_config.json", "adapter_model.safetensors", "README.md"):
        p = os.path.join(lora_dir, f)
        if os.path.exists(p): shutil.copy(p, up)
    step_dir = os.path.dirname(lora_dir) if os.path.basename(lora_dir) == "lora" else None
    for f in (["meta.json"] + (["critic.pt"] if with_critic else [])):
        cands = glob.glob(os.path.join(local, "**", f), recursive=True)
        if cands: shutil.copy(cands[0], up)
    sz = sum(os.path.getsize(os.path.join(up, f)) for f in os.listdir(up)) / 1e6
    print(f"  fetched {remote} -> {up} ({sz:.0f} MB; files {sorted(os.listdir(up))})", flush=True)
    return up

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="+", help="name=/vol/path (path relative to the volume root, e.g. /rl/sft/v0b_mix/lora)")
    ap.add_argument("--repo", default="ceselder/nlt-qwen3-8b-verbalizer")
    ap.add_argument("--desc", action="append", default=[], help="name=description for the model card table")
    ap.add_argument("--with-critic", action="store_true")
    ap.add_argument("--work", default="/tmp/hf_up")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(a.repo, repo_type="model", exist_ok=True, private=a.private)
    descs = dict(d.split("=", 1) for d in a.desc)
    done = []
    for it in a.items:
        name, remote = it.split("=", 1)
        print(f"[{name}] {remote}", flush=True)
        up = fetch(remote, os.path.join(a.work, name), a.with_critic)
        api.upload_folder(folder_path=up, repo_id=a.repo, path_in_repo=name, repo_type="model",
                          commit_message=f"add {name} ({remote})")
        done.append((name, remote)); print(f"  uploaded -> https://huggingface.co/{a.repo}/tree/main/{name}", flush=True)
        shutil.rmtree(os.path.join(a.work, name), ignore_errors=True)
    # model card: merge with existing table rows if any
    rows = []
    try:
        old = api.hf_hub_download(a.repo, "README.md", repo_type="model", cache_dir=os.path.join(a.work, "_card"))
        rows = [l for l in open(old).read().splitlines() if l.startswith("| `")]
    except Exception: pass
    names_done = {n for n, _ in done}
    rows = [r for r in rows if r.split("`")[1] not in names_done]
    for n, r in done: rows.append(f"| `{n}` | {descs.get(n, r)} |")
    card = CARD.format(rows="\n".join(rows), when=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()))
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=a.repo, repo_type="model",
                    commit_message="model card")
    print(f"card updated: https://huggingface.co/{a.repo}", flush=True)

if __name__ == "__main__":
    main()
