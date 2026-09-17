"""DDP flow-matching trainer consuming activation shards from a local shard dir (see nla/flow/shards.py).
torchrun --nproc_per_node=M -m nla.flow.train --shard-dir ... --stats ... --heldout ... --ckpt-dir ...
GLP recipe: AdamW lr 5e-5 (batch 4096), cosine with 1% warmup, grad clip 1.0, bf16 autocast, single pass over the stream."""
import argparse, json, math, os, queue, threading, time
import torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from nla.flow.model import Denoiser, Normalizer, fm_loss, euler_sample, frechet_distance
from nla.flow.shards import claim_shard, n_ready


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shard-dir", required=True); p.add_argument("--stats", required=True); p.add_argument("--heldout", default=None)
    p.add_argument("--ckpt-dir", required=True); p.add_argument("--d-input", type=int, default=5120)
    p.add_argument("--d-model", type=int, default=10240); p.add_argument("--d-mlp", type=int, default=20480); p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--batch", type=int, default=4096, help="per GPU"); p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--total-samples", type=float, default=2e9); p.add_argument("--warmup", type=float, default=0.01); p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0); p.add_argument("--wd", type=float, default=0.0); p.add_argument("--ema", type=float, default=0.9999)
    p.add_argument("--ckpt-every", type=int, default=2000); p.add_argument("--snapshot-every-samples", type=float, default=128e6)
    p.add_argument("--eval-every", type=int, default=1000); p.add_argument("--eval-n", type=int, default=16384); p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--max-hours", type=float, default=22.3); p.add_argument("--stop-file", default=None)
    p.add_argument("--wandb-project", default="nla-glp"); p.add_argument("--wandb-name", default=None); p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--stream-timeout", type=float, default=1800, help="seconds without a shard before we assume producers are done")
    p.add_argument("--compile", action="store_true")
    return p.parse_args()


class ShardFeeder:
    """Background thread: claims shards, normalises, shuffles, yields per-GPU batches."""
    def __init__(self, shard_dir, norm, batch, device, stop_file, timeout, prefetch=4):
        self.q = queue.Queue(maxsize=prefetch); self.norm = norm; self.batch = batch; self.device = device
        self.shard_dir, self.stop_file, self.timeout = shard_dir, stop_file, timeout
        self.n_tokens = 0; self.done = False
        threading.Thread(target=self._run, daemon=True).start()
    def _run(self):
        while True:
            got = claim_shard(self.shard_dir, timeout=self.timeout, stop_file=self.stop_file)
            if got is None:
                self.q.put(None); self.done = True; return
            _, payload = got
            acts = payload["acts"]; self.n_tokens += int(payload.get("n_tokens", 0))
            perm = torch.randperm(acts.shape[0])
            acts = acts[perm]
            for i in range(0, acts.shape[0] - self.batch + 1, self.batch):
                self.q.put(acts[i:i + self.batch])
    def next(self):
        x = self.q.get()
        if x is None: return None
        return self.norm.normalize(x.to(self.device, non_blocking=True))


def lr_at(step, total_steps, base_lr, warmup_frac, min_frac):
    w = max(1, int(total_steps * warmup_frac))
    if step < w: return base_lr * (step + 1) / w
    pr = min(1.0, (step - w) / max(1, total_steps - w))
    return base_lr * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * pr)))


@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)


@torch.no_grad()
def evaluate(ema, norm, held, device, n, sample_steps, d):
    """Held-out FM loss on a fixed t-grid with fixed noise + Frechet distance of EMA samples vs real (normalised space)."""
    ema.eval()
    x0 = norm.normalize(held[:n].to(device)); g = torch.Generator(device=device).manual_seed(0)
    out = {}
    tot = 0.0
    for t_val in (0.1, 0.3, 0.5, 0.7, 0.9):
        eps = torch.randn(x0.shape, device=device, generator=g); t = torch.full((x0.shape[0],), t_val, device=device)
        ls = []
        for i in range(0, x0.shape[0], 4096):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l, _ = fm_loss(ema, x0[i:i+4096], t[i:i+4096], eps[i:i+4096])
            ls.append(l.item() * x0[i:i+4096].shape[0])
        out[f"eval/fm_loss_t{t_val}"] = sum(ls) / x0.shape[0]; tot += out[f"eval/fm_loss_t{t_val}"]
    out["eval/fm_loss"] = tot / 5
    n_s = min(4096, n)
    noise = torch.randn((n_s, d), device=device, generator=g)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        samp = euler_sample(ema, noise, n_steps=sample_steps).float()
    real = x0[:n_s]
    out["eval/fd_normalised"] = frechet_distance(samp, real)
    out["eval/sample_norm_mean"] = norm.denormalize(samp).norm(dim=-1).mean().item(); out["eval/real_norm_mean"] = norm.denormalize(real).norm(dim=-1).mean().item()
    out["eval/sample_std_mean"] = samp.std(0).mean().item(); out["eval/real_std_mean"] = real.std(0).mean().item()
    ema.train()
    return out


def save_ckpt(path, model, ema, opt, step, samples, args, final=False):
    os.makedirs(path + ".tmp", exist_ok=True)
    torch.save({"model": {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}, "step": step, "samples": samples, "args": vars(args)}, os.path.join(path + ".tmp", "model.pt"))
    torch.save({"ema": {k: v.to(torch.bfloat16) for k, v in ema.state_dict().items()}}, os.path.join(path + ".tmp", "ema.pt"))
    if not final:
        torch.save({"opt": opt.state_dict(), "model_fp32": model.state_dict()}, os.path.join(path + ".tmp", "opt.pt"))
    if os.path.exists(path):
        os.rename(path, path + ".old")
    os.rename(path + ".tmp", path)
    if os.path.exists(path + ".old"):
        import shutil; shutil.rmtree(path + ".old", ignore_errors=True)


def main():
    a = get_args()
    dist.init_process_group("nccl"); rank = dist.get_rank(); world = dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))); torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    norm = Normalizer.load(a.stats).to(device)
    model = Denoiser(a.d_input, a.d_model, a.d_mlp, a.n_layers).to(device)
    ema = Denoiser(a.d_input, a.d_model, a.d_mlp, a.n_layers).to(device); ema.load_state_dict(model.state_dict()); ema.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd, fused=True)
    global_batch = a.batch * world; total_steps = int(a.total_samples // global_batch)
    step, samples = 0, 0
    latest = os.path.join(a.ckpt_dir, "latest")
    if os.path.exists(os.path.join(latest, "opt.pt")):
        ck = torch.load(os.path.join(latest, "opt.pt"), map_location="cpu"); model.load_state_dict(ck["model_fp32"]); opt.load_state_dict(ck["opt"])
        m = torch.load(os.path.join(latest, "model.pt"), map_location="cpu"); step, samples = m["step"], m["samples"]
        e = torch.load(os.path.join(latest, "ema.pt"), map_location="cpu"); ema.load_state_dict({k: v.float() for k, v in e["ema"].items()})
        if rank == 0: print(f"[train] resumed from step {step} ({samples/1e6:.0f}M samples)", flush=True)
    ddp = DDP(model, device_ids=[device.index], gradient_as_bucket_view=True)
    fwd = torch.compile(ddp) if a.compile else ddp
    if rank == 0:
        print(f"[train] denoiser {model.n_params()/1e9:.2f}B params, world {world}, global batch {global_batch}, total steps {total_steps}", flush=True)
        if not a.no_wandb:
            import wandb; wandb.init(project=a.wandb_project, name=a.wandb_name, config=vars(a) | {"n_params": model.n_params(), "world": world}, resume="allow", id=(a.wandb_name or None))
    held = torch.load(a.heldout, map_location="cpu")["acts"] if (a.heldout and os.path.exists(a.heldout)) else None
    feeder = ShardFeeder(a.shard_dir, norm, a.batch, device, a.stop_file, a.stream_timeout)
    t_start = time.time(); t_log = time.time(); loss_acc, n_acc = 0.0, 0; next_snapshot = (samples // a.snapshot_every_samples + 1) * a.snapshot_every_samples
    stop_flag = torch.zeros(1, device=device)
    while step < total_steps:
        x0 = feeder.next()
        stop_flag.fill_(1.0 if (x0 is None or (time.time() - t_start) / 3600 > a.max_hours or (a.stop_file and os.path.exists(a.stop_file))) else 0.0)
        dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
        if stop_flag.item() > 0:
            if rank == 0: print(f"[train] stopping at step {step}: {'stream ended' if x0 is None else 'time budget / stop file'}", flush=True)
            break
        lr = lr_at(step, total_steps, a.lr, a.warmup, a.min_lr_frac)
        for g in opt.param_groups: g["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = fm_loss(fwd, x0)
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step(); ema_update(ema, model, a.ema if step > 100 else 0.0)
        step += 1; samples += global_batch; loss_acc += loss.item(); n_acc += 1
        if rank == 0 and (step % 50 == 0):
            dt = time.time() - t_log; t_log = time.time()
            rec = {"train/loss": loss_acc / n_acc, "train/lr": lr, "train/grad_norm": float(gn), "train/samples": samples, "train/samples_per_s": 50 * global_batch / dt,
                   "train/ready_shards": n_ready(a.shard_dir), "train/hours": (time.time() - t_start) / 3600}
            print(f"[train] step {step} loss {rec['train/loss']:.4f} lr {lr:.2e} gn {float(gn):.2f} {rec['train/samples_per_s']/1e3:.1f}k/s ready {rec['train/ready_shards']} {samples/1e6:.0f}M", flush=True)
            if not a.no_wandb: import wandb; wandb.log(rec, step=step)
            loss_acc, n_acc = 0.0, 0
        if rank == 0 and held is not None and step % a.eval_every == 0:
            ev = evaluate(ema, norm, held, device, a.eval_n, a.sample_steps, a.d_input)
            print("[eval] " + " ".join(f"{k.split('/')[1]}={v:.4f}" for k, v in ev.items()), flush=True)
            if not a.no_wandb: import wandb; wandb.log(ev, step=step)
        if step % a.ckpt_every == 0:
            dist.barrier()
            if rank == 0:
                save_ckpt(latest, model, ema, opt, step, samples, a)
                if samples >= next_snapshot:
                    save_ckpt(os.path.join(a.ckpt_dir, f"snap_{int(samples/1e6):06d}M"), model, ema, opt, step, samples, a, final=True); next_snapshot += a.snapshot_every_samples
                print(f"[train] checkpoint @ step {step}", flush=True)
            dist.barrier()
    dist.barrier()
    if rank == 0:
        save_ckpt(latest, model, ema, opt, step, samples, a)
        save_ckpt(os.path.join(a.ckpt_dir, "final"), model, ema, opt, step, samples, a, final=True)
        json.dump({"step": step, "samples": samples, "hours": (time.time() - t_start) / 3600, "tokens_seen_by_feeder_rank0": feeder.n_tokens}, open(os.path.join(a.ckpt_dir, "final", "summary.json"), "w"))
        print(f"[train] final checkpoint @ step {step}, {samples/1e6:.0f}M samples", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
