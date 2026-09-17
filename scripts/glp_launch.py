"""In-container launcher: N producer GPUs stream activations into a local shard dir, M consumer GPUs train the flow (DDP).
Handles: base snapshot download, waiting for stats/held-out, producer restarts (resume via progress files), clean stop."""
import argparse, json, os, subprocess, sys, time
for _p in (os.path.dirname(os.path.abspath(__file__)), os.path.join(os.environ.get("PYTHONPATH", "/root/easyNLA").split(":")[0], "scripts")):
    if os.path.isdir(_p) and _p not in sys.path: sys.path.insert(0, _p)
from playground_app import resolve_base  # noqa: E402  (HF snapshot -> local disk with shard verification)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True, help="volume dir: stats, heldout, progress, ckpts, logs, config.yaml")
    p.add_argument("--config", default=None, help="override YAML (inherits configs/glp/default_27b_l42.yaml)")
    p.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.lr=1e-4 model.n_layers=12")
    p.add_argument("--shard-dir", default="/shards"); p.add_argument("--wandb-name", default=None)
    a = p.parse_args()
    from nla.flow.config import load_config, save_config
    C = load_config(a.config, a.set)
    os.makedirs(a.out_dir, exist_ok=True); logs = os.path.join(a.out_dir, "logs"); os.makedirs(logs, exist_ok=True)
    save_config(C, os.path.join(a.out_dir, "config.yaml")); print("[launch] config:", json.dumps(C), flush=True)
    stop_file = os.path.join(a.shard_dir, "STOP"); os.makedirs(a.shard_dir, exist_ok=True)
    if os.path.exists(stop_file): os.unlink(stop_file)
    base = resolve_base(C["base_id"], local_snapshot=True); print(f"[launch] base -> {base}", flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    n_prod, n_cons = C["gpus"]["producers"], C["gpus"]["consumers"]; D, S, ST = C["data"], C["shards"], C["stats"]
    prods = {}
    def start_prod(i):
        cmd = [sys.executable, "-m", "nla.flow.produce", "--base", base, "--layer", str(C["layer"]), "--n-producers", str(n_prod), "--index", str(i),
               "--shard-dir", a.shard_dir, "--out-dir", a.out_dir, "--max-tokens", str(D["max_tokens_per_producer"]), "--stop-file", stop_file,
               "--sources-json", json.dumps(D["sources"]), "--attn-impl", D.get("attn_impl", "sdpa"), "--max-len", str(D["max_len"]), "--min-len", str(D["min_len"]),
               "--tokens-per-batch", str(D["tokens_per_batch"]), "--buffer-docs", str(D["buffer_docs"]), "--seed", str(D["seed"]),
               "--shard-size", str(S["size"]), "--max-ready", str(S["max_ready"]), "--stats-n", str(ST["n"]), "--heldout-n", str(ST["heldout_n"]),
               "--heldout-docs-full", str(ST["heldout_docs_full"])] + ([] if D.get("drop_pos0", True) else ["--keep-pos0"])
        lf = open(os.path.join(logs, f"prod_{i}.log"), "a")
        prods[i] = (subprocess.Popen(cmd, env=dict(env, CUDA_VISIBLE_DEVICES=str(i)), stdout=lf, stderr=subprocess.STDOUT), lf, 0)
        print(f"[launch] producer {i} started (gpu {i})", flush=True)
    for i in range(n_prod): start_prod(i)
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
    gpus = ",".join(str(i) for i in range(n_prod, n_prod + n_cons)); M, T, E, K = C["model"], C["train"], C["eval"], C["ckpt"]
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={n_cons}", "-m", "nla.flow.train",
           "--shard-dir", a.shard_dir, "--stats", stats, "--heldout", held, "--ckpt-dir", os.path.join(a.out_dir, "ckpts"),
           "--d-model", str(M["d_model"]), "--d-mlp", str(M["d_mlp"]), "--n-layers", str(M["n_layers"]), "--batch", str(T["batch"]), "--grad-accum", str(T.get("grad_accum", 1)), "--lr", str(T["lr"]),
           "--total-samples", str(T["total_samples"]), "--warmup", str(T["warmup"]), "--min-lr-frac", str(T["min_lr_frac"]), "--clip", str(T["clip"]),
           "--wd", str(T["wd"]), "--ema", str(T["ema"]), "--eval-every", str(E["every"]), "--eval-n", str(E["n"]), "--sample-steps", str(E["sample_steps"]),
           "--ckpt-every", str(K["every"]), "--snapshot-every-samples", str(K["snapshot_every_samples"]), "--max-hours", str(C["max_hours"]),
           "--stream-timeout", str(C.get("stream_timeout", 1800)), "--stop-file", stop_file] + (["--compile"] if T.get("compile") else []) + (["--fsdp"] if T.get("fsdp") else []) + (["--wandb-name", a.wandb_name] if a.wandb_name else [])
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
            open(stop_file, "w").write("producers done")
            break
    tr.wait(); print(f"[launch] trainer exited with {tr.returncode}", flush=True)
    open(stop_file, "w").write("trainer exited")
    for i, (pr, lf, n) in prods.items():
        try: pr.wait(timeout=90)
        except subprocess.TimeoutExpired:
            pr.terminate()
            try: pr.wait(timeout=30)
            except subprocess.TimeoutExpired: pr.kill()
    print("[launch] done", flush=True)
    sys.exit(tr.returncode)


if __name__ == "__main__":
    main()
