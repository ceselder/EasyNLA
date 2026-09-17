"""In-container launcher: N producer GPUs stream activations into a local shard dir, M consumer GPUs train the flow (DDP).
Handles: base snapshot download, waiting for stats/held-out, producer restarts (resume via progress files), clean stop."""
import argparse, json, os, subprocess, sys, time
for _p in (os.path.dirname(os.path.abspath(__file__)), os.path.join(os.environ.get("PYTHONPATH", "/root/easyNLA").split(":")[0], "scripts")):
    if os.path.isdir(_p) and _p not in sys.path: sys.path.insert(0, _p)
from playground_app import resolve_base  # noqa: E402  (HF snapshot -> local disk with shard verification)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-id", default="Qwen/Qwen3.6-27B"); p.add_argument("--layer", type=int, default=42)
    p.add_argument("--n-producers", type=int, default=5); p.add_argument("--n-consumers", type=int, default=3)
    p.add_argument("--out-dir", required=True, help="volume dir: stats, heldout, progress, ckpts, logs"); p.add_argument("--shard-dir", default="/shards")
    p.add_argument("--d-model", type=int, default=10240); p.add_argument("--d-mlp", type=int, default=20480); p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--batch", type=int, default=4096); p.add_argument("--lr", type=float, default=5e-5); p.add_argument("--total-samples", type=float, default=2e9)
    p.add_argument("--max-tokens-per-producer", type=float, default=float("inf")); p.add_argument("--max-hours", type=float, default=22.3)
    p.add_argument("--producer-extra", default=""); p.add_argument("--trainer-extra", default=""); p.add_argument("--wandb-name", default=None)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True); logs = os.path.join(a.out_dir, "logs"); os.makedirs(logs, exist_ok=True)
    stop_file = os.path.join(a.shard_dir, "STOP"); os.makedirs(a.shard_dir, exist_ok=True)
    if os.path.exists(stop_file): os.unlink(stop_file)
    base = resolve_base(a.base_id, local_snapshot=True); print(f"[launch] base -> {base}", flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    prods = {}
    def start_prod(i):
        cmd = [sys.executable, "-m", "nla.flow.produce", "--base", base, "--layer", str(a.layer), "--n-producers", str(a.n_producers), "--index", str(i),
               "--shard-dir", a.shard_dir, "--out-dir", a.out_dir, "--max-tokens", str(a.max_tokens_per_producer), "--stop-file", stop_file] + a.producer_extra.split()
        lf = open(os.path.join(logs, f"prod_{i}.log"), "a")
        prods[i] = (subprocess.Popen(cmd, env=dict(env, CUDA_VISIBLE_DEVICES=str(i)), stdout=lf, stderr=subprocess.STDOUT), lf, 0)
        print(f"[launch] producer {i} started (gpu {i})", flush=True)
    for i in range(a.n_producers): start_prod(i)
    stats = os.path.join(a.out_dir, "rep_statistics.pt"); held = os.path.join(a.out_dir, "heldout_acts.pt")
    t0 = time.time()
    while not (os.path.exists(stats) and os.path.exists(held)):
        time.sleep(15)
        for i, (pr, lf, n) in list(prods.items()):
            if pr.poll() is not None:
                print(f"[launch] producer {i} exited early (code {pr.returncode}) before stats; tail:", flush=True)
                os.system(f"tail -n 25 {os.path.join(logs, f'prod_{i}.log')}"); sys.exit(1)
        if time.time() - t0 > 3 * 3600: print("[launch] stats/heldout not produced within 3 h", flush=True); sys.exit(1)
    print(f"[launch] stats + heldout ready after {(time.time()-t0)/60:.1f} min", flush=True)
    gpus = ",".join(str(i) for i in range(a.n_producers, a.n_producers + a.n_consumers))
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={a.n_consumers}", "-m", "nla.flow.train",
           "--shard-dir", a.shard_dir, "--stats", stats, "--heldout", held, "--ckpt-dir", os.path.join(a.out_dir, "ckpts"),
           "--d-model", str(a.d_model), "--d-mlp", str(a.d_mlp), "--n-layers", str(a.n_layers), "--batch", str(a.batch), "--lr", str(a.lr),
           "--total-samples", str(a.total_samples), "--max-hours", str(a.max_hours), "--stop-file", stop_file] + (["--wandb-name", a.wandb_name] if a.wandb_name else []) + a.trainer_extra.split()
    print("[launch] trainer:", " ".join(cmd), flush=True)
    tr = subprocess.Popen(cmd, env=dict(env, CUDA_VISIBLE_DEVICES=gpus, OMP_NUM_THREADS="8"))
    while tr.poll() is None:
        time.sleep(30)
        for i, (pr, lf, n) in list(prods.items()):
            if pr.poll() is not None and not os.path.exists(stop_file):
                lf.close(); print(f"[launch] producer {i} exited (code {pr.returncode}); restarts so far {n}", flush=True)
                os.system(f"tail -n 8 {os.path.join(logs, f'prod_{i}.log')}")
                if pr.returncode != 0 and n < 3:
                    start_prod(i); prods[i] = (prods[i][0], prods[i][1], n + 1)
                else:
                    del prods[i]
        if not prods and not os.path.exists(stop_file):
            print("[launch] all producers finished; trainer will stop when the shard queue drains", flush=True)
            open(stop_file, "w").write("producers done")   # trainer stops when ready queue empty + stop file
            break
    tr.wait(); print(f"[launch] trainer exited with {tr.returncode}", flush=True)
    open(stop_file, "w").write("trainer exited")
    for i, (pr, lf, n) in prods.items():
        try: pr.wait(timeout=600)
        except subprocess.TimeoutExpired: pr.kill()
    print("[launch] done", flush=True)
    sys.exit(tr.returncode)


if __name__ == "__main__":
    main()
