"""In-container launcher for the unCLIP decoder: N producer GPUs stream FineWeb/WildChat layer-42 activations into a local shard queue
(nla.flow.produce, the path that fed the prior its 2B samples) while M consumer GPUs train nla.unclip.train_dec (FSDP2). The prior's
standardisation statistics + held-out set are REUSED (copied into --out-dir so producer 0 does not recompute them): the encoder f and the prior
both live in that model space. Producers resume from --out-dir/progress_i.json, the trainer from --out-dir/latest (DCP), so a relaunch continues.
usage (inside scripts/unclip_modal.py train_stream): python scripts/unclip_launch.py --out-dir /vol_glp/unclip/decoder/<tag> --trainer-extra "..." """
import argparse, json, os, shutil, subprocess, sys, time
for _p in (os.path.dirname(os.path.abspath(__file__)), os.path.join(os.environ.get("PYTHONPATH", "/root/easyNLA").split(":")[0], "scripts")):
    if os.path.isdir(_p) and _p not in sys.path: sys.path.insert(0, _p)
from playground_app import resolve_base  # noqa: E402

SOURCES = [{"name": "fineweb", "kind": "hf_text", "dataset": "HuggingFaceFW/fineweb", "config": "sample-10BT", "weight": 0.85},
           {"name": "wildchat_onpolicy", "kind": "chat_parquet", "pattern": "/vol_glp/data/wildchat_onpolicy_*.parquet", "weight": 0.10},
           {"name": "wildchat_original", "kind": "chat_parquet", "pattern": "/vol_glp/data/wildchat_original.parquet", "weight": 0.05}]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True); p.add_argument("--shard-dir", default="/shards"); p.add_argument("--n-producers", type=int, default=4); p.add_argument("--n-consumers", type=int, default=4)
    p.add_argument("--prior", default="/vol_glp/glp27b_main/ckpts/snap_001966M"); p.add_argument("--stats", default="/vol_glp/glp27b_main/rep_statistics.pt"); p.add_argument("--heldout", default="/vol_glp/glp27b_main/heldout_acts.pt")
    p.add_argument("--encoder-json", default="/vol_glp/unclip/encoder.json"); p.add_argument("--base-id", default="Qwen/Qwen3.6-27B"); p.add_argument("--layer", type=int, default=42)
    p.add_argument("--seed", type=int, default=1, help="stream shuffle seed (the prior used 0)"); p.add_argument("--tokens-per-batch", type=int, default=32768); p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--shard-size", type=int, default=16384); p.add_argument("--max-ready", type=int, default=48); p.add_argument("--sources-json", default=json.dumps(SOURCES))
    p.add_argument("--trainer-extra", default="", help="extra args for nla.unclip.train_dec"); p.add_argument("--wandb-name", default=None); p.add_argument("--tag", default="unclip_dec")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True); logs = os.path.join(a.out_dir, "logs"); os.makedirs(logs, exist_ok=True)
    stop_file = os.path.join(a.shard_dir, "STOP"); os.makedirs(a.shard_dir, exist_ok=True)
    if os.path.exists(stop_file): os.unlink(stop_file)
    for src, name in ((a.stats, "rep_statistics.pt"), (a.heldout, "heldout_acts.pt")):   # producer 0 skips stats/held-out when these exist in ITS out-dir
        dst = os.path.join(a.out_dir, name)
        if not os.path.exists(dst): shutil.copy2(src, dst)
    base = resolve_base(a.base_id, local_snapshot=True); print(f"[launch] base -> {base}", flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    prods = {}
    def start_prod(i):
        cmd = [sys.executable, "-m", "nla.flow.produce", "--base", base, "--layer", str(a.layer), "--n-producers", str(a.n_producers), "--index", str(i), "--shard-dir", a.shard_dir, "--out-dir", a.out_dir,
               "--stop-file", stop_file, "--sources-json", a.sources_json, "--attn-impl", "sdpa", "--max-len", str(a.max_len), "--min-len", "16", "--tokens-per-batch", str(a.tokens_per_batch),
               "--buffer-docs", "512", "--seed", str(a.seed), "--shard-size", str(a.shard_size), "--max-ready", str(a.max_ready), "--stats-n", "1000", "--heldout-n", "1000", "--heldout-docs-full", "1"]
        lf = open(os.path.join(logs, f"prod_{i}.log"), "a")
        prods[i] = (subprocess.Popen(cmd, env=dict(env, CUDA_VISIBLE_DEVICES=str(i)), stdout=lf, stderr=subprocess.STDOUT), lf, 0)
        print(f"[launch] producer {i} started (gpu {i})", flush=True)
    for i in range(a.n_producers): start_prod(i)
    gpus = ",".join(str(i) for i in range(a.n_producers, a.n_producers + a.n_consumers))
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={a.n_consumers}", "-m", "nla.unclip.train_dec", "--shard-dir", a.shard_dir, "--stats", a.stats, "--heldout", a.heldout,
           "--encoder-json", a.encoder_json, "--prior", a.prior, "--out", a.out_dir, "--tag", a.tag, "--stop-file", stop_file] + (["--wandb-name", a.wandb_name] if a.wandb_name else []) + a.trainer_extra.split()
    print("[launch] trainer:", " ".join(cmd), flush=True)
    tr = subprocess.Popen(cmd, env=dict(env, CUDA_VISIBLE_DEVICES=gpus, OMP_NUM_THREADS="8", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"))
    while tr.poll() is None:
        time.sleep(30)
        for i, (pr, lf, n) in list(prods.items()):
            if pr.poll() is not None and not os.path.exists(stop_file):
                lf.close(); print(f"[launch] producer {i} exited (code {pr.returncode}); restarts so far {n}", flush=True); os.system(f"tail -n 8 {os.path.join(logs, f'prod_{i}.log')}")
                if pr.returncode != 0 and n < 5: start_prod(i); prods[i] = (prods[i][0], prods[i][1], n + 1)
                else: del prods[i]
        if not prods and not os.path.exists(stop_file):
            print("[launch] all producers finished; trainer stops when the queue drains", flush=True); open(stop_file, "w").write("producers done"); break
    tr.wait(); print(f"[launch] trainer exited with {tr.returncode}", flush=True); open(stop_file, "w").write("trainer exited")
    for i, (pr, lf, n) in prods.items():
        try: pr.wait(timeout=90)
        except subprocess.TimeoutExpired:
            pr.terminate()
            try: pr.wait(timeout=30)
            except subprocess.TimeoutExpired: pr.kill()
    print("[launch] done", flush=True); sys.exit(tr.returncode)


if __name__ == "__main__":
    main()
