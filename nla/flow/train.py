"""DDP flow-matching trainer consuming activation shards from a local shard dir (see nla/flow/shards.py).
torchrun --nproc_per_node=M -m nla.flow.train --shard-dir ... --stats ... --heldout ... --ckpt-dir ...
GLP recipe: AdamW lr 5e-5 (batch 4096), cosine with 1% warmup, grad clip 1.0, bf16 autocast, single pass over the stream."""
import argparse, json, math, os, queue, threading, time
import torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict, get_optimizer_state_dict, set_model_state_dict, set_optimizer_state_dict, StateDictOptions

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
    p.add_argument("--compile", action="store_true"); p.add_argument("--fsdp", action="store_true", help="shard params/grads/optimizer across consumer GPUs (needed above ~5B params)")
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
            try:
                got = claim_shard(self.shard_dir, timeout=self.timeout, stop_file=self.stop_file)
            except Exception as e:          # never let the feeder thread die silently (a dead feeder hangs the whole DDP job)
                print(f"[feeder] claim_shard error, retrying: {type(e).__name__}: {e}", flush=True); time.sleep(1); continue
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


def _local(p):
    return p.to_local() if hasattr(p, "to_local") else p


@torch.no_grad()
def ema_update(ema, model, decay):
    pe = [_local(x.data) for x in ema.parameters()]; pm = [_local(x.data) for x in model.parameters()]
    torch._foreach_mul_(pe, decay); torch._foreach_add_(pe, pm, alpha=1 - decay)


_FD_FLOOR = {}


@torch.no_grad()
def _heldout_fm(model, x0, device, prefix):
    g = torch.Generator(device=device).manual_seed(0); out = {}; tot = 0.0
    for t_val in (0.1, 0.3, 0.5, 0.7, 0.9):
        eps = torch.randn(x0.shape, device=device, generator=g); t = torch.full((x0.shape[0],), t_val, device=device)
        ls = []
        for i in range(0, x0.shape[0], 4096):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l, _ = fm_loss(model, x0[i:i+4096], t[i:i+4096], eps[i:i+4096])
            ls.append(l.item() * x0[i:i+4096].shape[0])
        out[f"{prefix}_t{t_val}"] = sum(ls) / x0.shape[0]; tot += out[f"{prefix}_t{t_val}"]
    out[prefix] = tot / 5
    return out


@torch.no_grad()
def evaluate(ema, norm, held, device, n, sample_steps, d, raw_model=None):
    """Held-out FM loss (EMA and raw weights) on a fixed t-grid with fixed noise + Frechet distance of EMA samples vs real (normalised space)."""
    ema.eval()
    x0 = norm.normalize(held[:n].to(device)); g = torch.Generator(device=device).manual_seed(0)
    out = _heldout_fm(ema, x0, device, "eval/fm_loss")
    if raw_model is not None:
        raw_model.eval(); out.update(_heldout_fm(raw_model, x0, device, "eval/fm_loss_raw")); raw_model.train()
    n_s = min(16384, n)
    noise = torch.randn((n_s, d), device=device, generator=g)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        samp = torch.cat([euler_sample(ema, noise[i:i+4096], n_steps=sample_steps).float() for i in range(0, n_s, 4096)])
    real = x0[:n_s]
    out["eval/fd_normalised"] = frechet_distance(samp, real)
    if _FD_FLOOR.get("v") is None:   # finite-sample floor: two disjoint halves of the real held-out set
        _FD_FLOOR["v"] = frechet_distance(x0[: n // 2], x0[n // 2: 2 * (n // 2)])
    out["eval/fd_floor_real_vs_real"] = _FD_FLOOR["v"]
    out["eval/sample_norm_mean"] = norm.denormalize(samp).norm(dim=-1).mean().item(); out["eval/real_norm_mean"] = norm.denormalize(real).norm(dim=-1).mean().item()
    out["eval/sample_std_mean"] = samp.std(0).mean().item(); out["eval/real_std_mean"] = real.std(0).mean().item()
    ema.train()
    return out


def _full_sd(mod, bf16=True):
    sd = get_model_state_dict(mod, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    return {k: (v.to(torch.bfloat16) if bf16 else v) for k, v in sd.items()}


def save_latest(path, model, ema, opt, step, samples, args, fsdp):
    """Resumable checkpoint. FSDP: DCP sharded (every rank writes its shards). DDP: rank-0 torch.save."""
    rank = dist.get_rank()
    tmp = path + ".tmp"
    if fsdp:
        if rank == 0 and os.path.exists(tmp): __import__("shutil").rmtree(tmp, ignore_errors=True)
        dist.barrier()
        dcp.save({"model": get_model_state_dict(model), "opt": get_optimizer_state_dict(model, opt), "ema": get_model_state_dict(ema)}, checkpoint_id=tmp)
        if rank == 0: json.dump({"step": step, "samples": samples, "args": vars(args)}, open(os.path.join(tmp, "meta.json"), "w"))
    else:
        if rank == 0:
            os.makedirs(tmp, exist_ok=True)
            torch.save({"model_fp32": model.state_dict(), "opt": opt.state_dict(), "ema": ema.state_dict(), "step": step, "samples": samples, "args": vars(args)}, os.path.join(tmp, "state.pt"))
    dist.barrier()
    if rank == 0:
        if os.path.exists(path): os.rename(path, path + ".old")
        os.rename(tmp, path)
        if os.path.exists(path + ".old"): __import__("shutil").rmtree(path + ".old", ignore_errors=True)
    dist.barrier()


def load_latest(path, model, ema, opt, fsdp):
    if fsdp:
        meta = json.load(open(os.path.join(path, "meta.json")))
        sd = {"model": get_model_state_dict(model), "opt": get_optimizer_state_dict(model, opt), "ema": get_model_state_dict(ema)}
        dcp.load(sd, checkpoint_id=path)
        set_model_state_dict(model, sd["model"]); set_optimizer_state_dict(model, opt, sd["opt"]); set_model_state_dict(ema, sd["ema"])
        return meta["step"], meta["samples"]
    ck = torch.load(os.path.join(path, "state.pt"), map_location="cpu")
    model.load_state_dict(ck["model_fp32"]); opt.load_state_dict(ck["opt"]); ema.load_state_dict(ck["ema"])
    return ck["step"], ck["samples"]


def save_snapshot(path, ema, model, step, samples, args, with_raw=False):
    """Full (unsharded) bf16 EMA weights for downstream use: <path>/ema.pt {"ema": sd}, <path>/model.pt {"args", "step", "samples"[, "model"]}."""
    ema_sd = _full_sd(ema); raw_sd = _full_sd(model) if with_raw else None
    if dist.get_rank() == 0:
        os.makedirs(path + ".tmp", exist_ok=True)
        torch.save({"ema": ema_sd}, os.path.join(path + ".tmp", "ema.pt"))
        torch.save({"args": vars(args), "step": step, "samples": samples, **({"model": raw_sd} if raw_sd else {})}, os.path.join(path + ".tmp", "model.pt"))
        if os.path.exists(path): __import__("shutil").rmtree(path, ignore_errors=True)
        os.rename(path + ".tmp", path)
    dist.barrier()


def main():
    a = get_args()
    dist.init_process_group("nccl"); rank = dist.get_rank(); world = dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))); torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    norm = Normalizer.load(a.stats).to(device)
    torch.manual_seed(0)
    model = Denoiser(a.d_input, a.d_model, a.d_mlp, a.n_layers).to(device)
    ema = Denoiser(a.d_input, a.d_model, a.d_mlp, a.n_layers).to(device); ema.load_state_dict(model.state_dict()); ema.requires_grad_(False)
    n_params = model.n_params()
    if a.compile:   # compile each MLP block (before sharding): fuses LN + gated MLP; safe with FSDP2
        for blk in model.layers: blk.compile()
    if a.fsdp:
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for mod in (model, ema):
            for blk in mod.layers: fully_shard(blk, mp_policy=mp)
            fully_shard(mod, mp_policy=mp)
        fwd = model
    else:
        fwd = DDP(model, device_ids=[device.index], gradient_as_bucket_view=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd, fused=not a.fsdp)
    global_batch = a.batch * world; total_steps = int(a.total_samples // global_batch)
    step, samples = 0, 0
    latest = os.path.join(a.ckpt_dir, "latest")
    if os.path.exists(os.path.join(latest, "meta.json" if a.fsdp else "state.pt")):
        step, samples = load_latest(latest, model, ema, opt, a.fsdp)
        if rank == 0: print(f"[train] resumed from step {step} ({samples/1e6:.0f}M samples)", flush=True)
    if rank == 0:
        print(f"[train] denoiser {n_params/1e9:.2f}B params, world {world}, fsdp {a.fsdp}, global batch {global_batch}, total steps {total_steps}", flush=True)
        if not a.no_wandb:
            import wandb
            try: wandb.init(project=a.wandb_project, name=a.wandb_name, config=vars(a) | {"n_params": n_params, "world": world}, resume="allow", id=(a.wandb_name or None))
            except Exception as ex: print(f"[train] wandb disabled: {ex}", flush=True); a.no_wandb = True
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
        gn = gn.full_tensor() if hasattr(gn, "full_tensor") else gn
        opt.step(); ema_update(ema, model, a.ema if step > 100 else 0.0)
        step += 1; samples += global_batch; loss_acc += loss.item(); n_acc += 1
        if rank == 0 and (step % 50 == 0):
            dt = time.time() - t_log; t_log = time.time()
            rec = {"train/loss": loss_acc / n_acc, "train/lr": lr, "train/grad_norm": float(gn), "train/samples": samples, "train/samples_per_s": 50 * global_batch / dt,
                   "train/ready_shards": n_ready(a.shard_dir), "train/hours": (time.time() - t_start) / 3600}
            print(f"[train] step {step} loss {rec['train/loss']:.4f} lr {lr:.2e} gn {float(gn):.2f} {rec['train/samples_per_s']/1e3:.1f}k/s ready {rec['train/ready_shards']} {samples/1e6:.0f}M", flush=True)
            if not a.no_wandb: import wandb; wandb.log(rec, step=step)
            loss_acc, n_acc = 0.0, 0
        if held is not None and step % a.eval_every == 0:
            # ALL ranks must run this: under FSDP every forward is a collective (weight all-gather); rank-0-only eval deadlocks the job
            ev = evaluate(ema, norm, held, device, a.eval_n, a.sample_steps, a.d_input, raw_model=model)
            if rank == 0:
                print("[eval] " + " ".join(f"{k.split('/')[1]}={v:.4f}" for k, v in ev.items() if not k.endswith(("_t0.1","_t0.3","_t0.5","_t0.7","_t0.9"))), flush=True)
                if not a.no_wandb: import wandb; wandb.log(ev, step=step)
        if step % a.ckpt_every == 0:
            t_ck = time.time(); save_latest(latest, model, ema, opt, step, samples, a, a.fsdp)
            if samples >= next_snapshot:
                save_snapshot(os.path.join(a.ckpt_dir, f"snap_{int(samples/1e6):06d}M"), ema, model, step, samples, a); next_snapshot += a.snapshot_every_samples
            if rank == 0: print(f"[train] checkpoint @ step {step} ({time.time()-t_ck:.0f}s)", flush=True)
    dist.barrier()
    save_latest(latest, model, ema, opt, step, samples, a, a.fsdp)
    save_snapshot(os.path.join(a.ckpt_dir, "final"), ema, model, step, samples, a, with_raw=True)
    if rank == 0:
        json.dump({"step": step, "samples": samples, "hours": (time.time() - t_start) / 3600, "tokens_seen_by_feeder_rank0": feeder.n_tokens}, open(os.path.join(a.ckpt_dir, "final", "summary.json"), "w"))
        print(f"[train] final checkpoint @ step {step}, {samples/1e6:.0f}M samples", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
