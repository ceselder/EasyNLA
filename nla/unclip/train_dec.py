"""unCLIP DECODER p(h | e): the 13.7B activation flow prior (warm start) conditioned on the frozen contrastive embedding e = f(h) through the
vector-condition path of nla.flow.cond_model.CondDenoiser (cvec -> LN -> shared features -> zero-init additive injection into the input
projection and into every block's residual + low-rank gate modulation). Trained on UNLABELLED activations only (e is computed from h on the
fly), 10 % condition dropout so the same network is the unconditional prior (classifier-free guidance at sampling).

  torchrun --nproc_per_node=M -m nla.unclip.train_dec --prior /vol_glp/glp27b_main/ckpts/snap_001966M --stats ... --encoder-json ... --out ...
       (--shard-dir <local shard queue fed by nla.flow.produce>  |  --parquet-glob <extraction shards, static, for smokes>)

FSDP2 over the ranks (fp32 master + Adam sharded, bf16 compute), two lr groups (adapter / prior; --freeze-prior = adapter only).
Evals (all ranks; FSDP forwards are collectives): held-out FineWeb activations + the 736 doubly-held-out clean1 rows: FM loss per t for the
unconditional / conditional / SHUFFLED-e branches (paired noise), x0-hat FVE at t = 0.9, a 20-step conditional Euler sample (cos, FVE, norm).
Checkpoints: <out>/latest (DCP, resumable with the same world size) and FlowBundle-style snapshots <out>/snap_<samples>M/{adapter_latest.pt,
prior_cotrained_latest.pt, eval.json} (nla.unclip.decoder.load_decoder loads them)."""
import argparse, json, math, os, time
import numpy as np, torch, torch.distributed as dist, torch.nn.functional as F
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict, get_optimizer_state_dict, set_model_state_dict, set_optimizer_state_dict, StateDictOptions

from nla.flow.model import Denoiser, Normalizer
from nla.flow.cond_model import CondDenoiser, cond_fm_loss
from nla.flow.train import ShardFeeder, lr_at
from nla.flow.shards import n_ready

COND_MODE = "clip_vec"   # adapter args tag: pooled VECTOR condition e (d_cvec = d_e), no token cross-reads, no text anywhere


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prior", required=True, help="stage-1 snapshot dir (model.pt raw weights [+ ema.pt])"); p.add_argument("--prior-weights", default="raw", choices=["raw", "ema"])
    p.add_argument("--stats", required=True); p.add_argument("--encoder-json", default="/vol_glp/unclip/encoder.json"); p.add_argument("--out", required=True); p.add_argument("--tag", default="unclip_dec")
    p.add_argument("--heldout", default="/vol_glp/glp27b_main/heldout_acts.pt"); p.add_argument("--clean1", default="/vol_q36/data/sft/av_sft_val_clean1.parquet")
    p.add_argument("--shard-dir", default=None, help="streaming: local shard queue written by nla.flow.produce"); p.add_argument("--stream-timeout", type=float, default=1800)
    p.add_argument("--parquet-glob", default=None, help="static: extraction shards (activation_vector [, is_val]); rank-disjoint rows, epochs allowed"); p.add_argument("--max-rows", type=int, default=0, help="static: rows per rank (0 = all)")
    p.add_argument("--batch", type=int, default=8192, help="per GPU per micro-step"); p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4, help="adapter lr"); p.add_argument("--prior-lr", type=float, default=2e-5); p.add_argument("--freeze-prior", action="store_true")
    p.add_argument("--total-samples", type=float, default=1e9); p.add_argument("--warmup-steps", type=int, default=500); p.add_argument("--min-lr-frac", type=float, default=0.1); p.add_argument("--clip", type=float, default=1.0); p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--p-uncond", type=float, default=0.1); p.add_argument("--gate-rank", type=int, default=128); p.add_argument("--d-c", type=int, default=4096)
    p.add_argument("--n-slots", type=int, default=8); p.add_argument("--n-heads", type=int, default=4); p.add_argument("--d-head", type=int, default=64)   # unused (no token reads); recorded for the adapter args
    p.add_argument("--eval-every", type=int, default=500); p.add_argument("--eval-n", type=int, default=8192); p.add_argument("--sample-n", type=int, default=256); p.add_argument("--sample-steps", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=2000); p.add_argument("--snapshot-every-samples", type=float, default=1e8); p.add_argument("--snap-at-end", action="store_true")
    p.add_argument("--max-hours", type=float, default=22.3); p.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (smokes)"); p.add_argument("--stop-file", default=None)
    p.add_argument("--wandb-project", default="nla-glp"); p.add_argument("--wandb-name", default=None); p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--shard-coarse", action="store_true", help="FSDP with one unit per CondMLPBlock (FAILS: prior.layers[i] is also blocks[i].base, the root meets DTensors -> 'value was None'); default = the train_cond pattern (every called sub-module of the prior block sharded separately)")
    p.add_argument("--no-fsdp", action="store_true", help="single process / CPU test: plain tensors"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu-test", action="store_true", help="tiny random prior + random encoder weights on CPU (tests/CI)")
    return p.parse_args()


# ------------------------------------------------------------------------------------------------------------------ data
class StaticFeeder:
    """extraction shards in RAM (fp16, rank-disjoint contiguous slice of the non-val rows), shuffled every pass; .next() -> standardised x0"""
    def __init__(self, glob_pat, norm, batch, device, rank, world, max_rows, seed):
        import glob as _g, pyarrow.parquet as pq
        files = sorted(f for g in glob_pat.split(",") for f in _g.glob(g.strip()) if g.strip()); assert files, glob_pat
        counts = []
        for f in files:
            pf = pq.ParquetFile(f); n = pf.metadata.num_rows
            if "is_val" in pf.schema_arrow.names: n = sum(1 for v in pq.read_table(f, columns=["is_val"]).column(0).to_pylist() if not v)
            counts.append(n)
        tot = sum(counts); per = tot // world; lo, hi = rank * per, (rank + 1) * per
        if max_rows: hi = min(hi, lo + max_rows)
        acts = []; g0 = 0
        for f, c in zip(files, counts):
            if g0 + c <= lo or g0 >= hi: g0 += c; continue
            t = pq.read_table(f, columns=["activation_vector"] + (["is_val"] if "is_val" in pq.ParquetFile(f).schema_arrow.names else []))
            keep = [i for i, v in enumerate(t.column("is_val").to_pylist()) if not v] if "is_val" in t.column_names else list(range(t.num_rows))
            keep = keep[max(lo - g0, 0): min(hi - g0, c)]; g0 += c
            if not keep: continue
            av = t.column("activation_vector").combine_chunks().take(keep)
            acts.append(torch.from_numpy(np.asarray(av.values.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(len(keep), -1)).to(torch.float16)); del t
        self.acts = torch.cat(acts); self.norm, self.batch, self.device = norm, batch, device; self.rng = torch.Generator().manual_seed(seed * 1000 + rank)
        self.perm = torch.randperm(self.acts.shape[0], generator=self.rng); self.ptr = 0; self.passes = 0; self.n_tokens = 0
        print(f"[dec] rank {rank}: static data {self.acts.shape[0]} rows of {tot} non-val ({len(files)} files)", flush=True)
    def next(self):
        if self.ptr + self.batch > self.acts.shape[0]:
            self.perm = torch.randperm(self.acts.shape[0], generator=self.rng); self.ptr = 0; self.passes += 1
        idx = self.perm[self.ptr: self.ptr + self.batch]; self.ptr += self.batch
        return self.norm.normalize(self.acts[idx].to(self.device, non_blocking=True))


def load_clean1(path, n=None):
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["activation_vector"]); t = t.slice(0, n) if n else t
    return torch.tensor(np.asarray(t.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(t.num_rows, -1))


# ------------------------------------------------------------------------------------------------------------------ model
def build(a, cfg, sd, d_e, dev):
    prior = Denoiser(cfg["d_input"], cfg["d_model"], cfg["d_mlp"], cfg["n_layers"])
    if sd is not None: prior.load_state_dict({k: v.float() for k, v in sd.items()})
    model = CondDenoiser(prior, cfg["d_input"], a.n_slots, a.n_heads, a.d_head, a.gate_rank, d_cvec=d_e, use_tokens=False, d_c=a.d_c)
    return model.to(dev)


def shard(model, a):
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32); p = model.prior
    if not a.shard_coarse:   # train_cond pattern: every module whose forward is called, separately (the ONLY layout that works with the shared prior blocks)
        for i, pblk in enumerate(p.layers):
            for sub in (pblk.ln, pblk.up_proj, pblk.gate_proj, pblk.time_proj, pblk.down_proj): fully_shard(sub, mp_policy=mp)
            for sub in (model.blocks[i].cvec_out, model.blocks[i].gate_mod): fully_shard(sub, mp_policy=mp)
    else:              # one FSDP unit per CondMLPBlock: verified to FAIL on 2026-09-24 (kept for the record)
        for blk in model.blocks: fully_shard(blk, mp_policy=mp)
    for sub in (p.in_proj, p.time_embed, p.ln, p.out_proj, model.cvec_ln, model.cvec_in, model.cvec_x): fully_shard(sub, mp_policy=mp)
    fully_shard(model, mp_policy=mp)
    return model


def is_adapter_key(k):
    return ".gate_mod." in k or ".cvec_out." in k or k.startswith("cvec_") or ".read." in k or k.startswith("token_encoder.")


def adapter_args(a, cfg, d_e, step, samples):
    return {"cond_mode": COND_MODE, "n_slots": a.n_slots, "n_heads": a.n_heads, "d_head": a.d_head, "gate_rank": a.gate_rank, "d_c": a.d_c, "d_cvec": d_e, "d_e": d_e,
            "prior": a.prior, "prior_weights": a.prior_weights, "stats": a.stats, "encoder_json": a.encoder_json, "p_uncond": a.p_uncond, "tag": a.tag,
            "lr": a.lr, "prior_lr": a.prior_lr, "freeze_prior": a.freeze_prior, "batch": a.batch, "grad_accum": a.grad_accum, "step": step, "samples": samples,
            "unfreeze_prior": not a.freeze_prior, "enc_layer": 42, "whiten": None}


# ------------------------------------------------------------------------------------------------------------------ eval
@torch.no_grad()
def fm_branches(model, x0, e, dev, prefix, ts=(0.1, 0.3, 0.5, 0.7, 0.9), bs=4096):
    """held-out FM loss per t for uncond / cond / shuffled-e, paired noise; -> dict + the FM-proxy gain in bits per activation"""
    g = torch.Generator(device=dev).manual_seed(0); n, d = x0.shape; out = {}
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(1)).to(dev); e_shuf = e[perm]
    for tv in ts:
        eps = torch.randn(x0.shape, device=dev, generator=g); acc = {"uncond": 0.0, "cond": 0.0, "shuf": 0.0}
        for i in range(0, n, bs):
            xs, ee = x0[i:i + bs], eps[i:i + bs]; xt = (1 - tv) * xs + tv * ee; tt = torch.full((xs.shape[0],), tv, device=dev); tgt = ee - xs
            for name, cv in (("uncond", None), ("cond", e[i:i + bs]), ("shuf", e_shuf[i:i + bs])):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    v = model(xt, tt, None, None, cv) if cv is not None else model(xt, tt)
                acc[name] += F.mse_loss(v.float(), tgt.float(), reduction="sum").item() / d
        for name in acc: out[f"{prefix}/fm_{name}_t{tv}"] = acc[name] / n
    for name in ("uncond", "cond", "shuf"): out[f"{prefix}/fm_{name}"] = sum(out[f"{prefix}/fm_{name}_t{tv}"] for tv in ts) / len(ts)
    out[f"{prefix}/gain_bits_proxy"] = (out[f"{prefix}/fm_uncond"] - out[f"{prefix}/fm_cond"]) / (2 * math.log(2)) * d
    out[f"{prefix}/shuf_bits_proxy"] = (out[f"{prefix}/fm_uncond"] - out[f"{prefix}/fm_shuf"]) / (2 * math.log(2)) * d
    return out


@torch.no_grad()
def euler(model, x, e, n_steps, cfg_w=1.0, t_start=1.0, bs=4096):
    """dx/dt = v from t_start to 0; e None = unconditional; cfg_w != 1: v = v_u + w (v_c - v_u)"""
    ts = torch.linspace(t_start, 0.0, n_steps + 1, device=x.device); outs = []
    for i in range(0, x.shape[0], bs):
        xs = x[i:i + bs].clone(); ee = e[i:i + bs] if e is not None else None
        for k in range(n_steps):
            tt = ts[k].expand(xs.shape[0])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
                if ee is None: v = model(xs, tt).float()
                elif cfg_w == 1.0: v = model(xs, tt, None, None, ee).float()
                else: vu = model(xs, tt).float(); vc = model(xs, tt, None, None, ee).float(); v = vu + cfg_w * (vc - vu)
            xs = xs + v * (ts[k + 1] - ts[k])
        outs.append(xs)
    return torch.cat(outs)


@torch.no_grad()
def recon_stats(h_raw, h_hat, norm, prefix):
    """NLA-convention FVE (unit-L2 to sqrt(d), MSE vs the predict-mean baseline) + cosine + norm ratio, raw activation units"""
    from nla.schema import normalize_activation, compute_predict_mean_baselines
    d = h_raw.shape[1]; msf = math.sqrt(d); _, base = compute_predict_mean_baselines(h_raw, msf)
    mse = ((normalize_activation(h_hat, msf) - normalize_activation(h_raw, msf)) ** 2).mean().item()
    return {f"{prefix}_fve": 100 * (1 - mse / base), f"{prefix}_cos": F.cosine_similarity(h_hat, h_raw, dim=-1).mean().item(),
            f"{prefix}_norm_ratio": (h_hat.norm(dim=-1) / h_raw.norm(dim=-1)).mean().item(), f"{prefix}_rel_err": ((h_hat - h_raw).norm(dim=-1) / h_raw.norm(dim=-1)).mean().item()}


@torch.no_grad()
def evaluate(model, enc, norm, held_raw, clean_raw, a, dev):
    model.eval(); out = {}
    for prefix, raw in (("eval", held_raw), ("clean1", clean_raw)):
        if raw is None: continue
        x0 = norm.normalize(raw.to(dev)); e = enc.from_standardised(x0)
        out.update(fm_branches(model, x0, e, dev, prefix))
        # x0-hat at t = 0.9 under the condition (the 'conditional FVE' the conditioners report) and a conditional Euler sample (CFG 1) on sample_n rows
        n_s = min(a.sample_n, x0.shape[0]); xs, es = x0[:n_s], e[:n_s]; g = torch.Generator(device=dev).manual_seed(7)
        eps = torch.randn(xs.shape, device=dev, generator=g); tv = 0.9; xt = (1 - tv) * xs + tv * eps; tt = torch.full((n_s,), tv, device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"): v = model(xt, tt, None, None, es).float()
        out.update({f"{prefix}/{k}": val for k, val in recon_stats(raw[:n_s].to(dev).float(), norm.denormalize(xt - tv * v), norm, "x0hat_t0.9").items()})
        noise = torch.randn(xs.shape, device=dev, generator=g)
        samp_c = norm.denormalize(euler(model, noise, es, a.sample_steps)); samp_u = norm.denormalize(euler(model, noise, None, a.sample_steps))
        out.update({f"{prefix}/{k}": val for k, val in recon_stats(raw[:n_s].to(dev).float(), samp_c, norm, f"sample{a.sample_steps}_cfg1").items()})
        out[f"{prefix}/uncond_sample_norm_ratio"] = (samp_u.norm(dim=-1) / raw[:n_s].to(dev).norm(dim=-1)).mean().item()
        out[f"{prefix}/sample_e_cos"] = (enc(samp_c) * es).sum(-1).mean().item() / enc.d_e     # does the sample land at the right e? (CLIP-space consistency)
        del x0, e
    model.train()
    if hasattr(model, "reshard"): model.reshard()
    torch.cuda.empty_cache() if dev.type == "cuda" else None
    return out


# ------------------------------------------------------------------------------------------------------------------ ckpt
def save_latest(path, model, opt, step, samples, a, fsdp, rank):
    tmp = path + ".tmp"
    if fsdp:
        if rank == 0 and os.path.exists(tmp): __import__("shutil").rmtree(tmp, ignore_errors=True)
        dist.barrier()
        dcp.save({"model": get_model_state_dict(model), "opt": get_optimizer_state_dict(model, opt)}, checkpoint_id=tmp)
        if rank == 0: json.dump({"step": step, "samples": samples, "args": vars(a)}, open(os.path.join(tmp, "meta.json"), "w"))
        dist.barrier()
    else:
        os.makedirs(tmp, exist_ok=True); torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "samples": samples, "args": vars(a)}, os.path.join(tmp, "state.pt"))
        json.dump({"step": step, "samples": samples, "args": vars(a)}, open(os.path.join(tmp, "meta.json"), "w"))
    if rank == 0:
        if os.path.exists(path): os.rename(path, path + ".old")
        os.rename(tmp, path)
        if os.path.exists(path + ".old"): __import__("shutil").rmtree(path + ".old", ignore_errors=True)
    if fsdp: dist.barrier()


def load_latest(path, model, opt, fsdp):
    meta = json.load(open(os.path.join(path, "meta.json")))
    if fsdp:
        sd = {"model": get_model_state_dict(model), "opt": get_optimizer_state_dict(model, opt)}; dcp.load(sd, checkpoint_id=path)
        set_model_state_dict(model, sd["model"]); set_optimizer_state_dict(model, opt, sd["opt"])
    else:
        ck = torch.load(os.path.join(path, "state.pt"), map_location="cpu"); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
    return meta["step"], meta["samples"]


def save_snapshot(path, model, a, cfg, d_e, step, samples, ev, fsdp, rank):
    """FlowBundle-style: adapter_latest.pt {adapter, args, prior_cfg, step} + prior_cotrained_latest.pt {model (bf16), args, step} + eval.json"""
    full = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)) if fsdp else {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if rank == 0:
        os.makedirs(path + ".tmp", exist_ok=True)
        torch.save({"adapter": {k: v.float() for k, v in full.items() if is_adapter_key(k)}, "args": adapter_args(a, cfg, d_e, step, samples), "prior_cfg": cfg, "step": step, "samples": samples}, os.path.join(path + ".tmp", "adapter_latest.pt"))
        torch.save({"model": {k[len("prior."):]: v.to(torch.bfloat16) for k, v in full.items() if k.startswith("prior.")}, "args": cfg, "step": step, "samples": samples, "cotrained_with": a.tag}, os.path.join(path + ".tmp", "prior_cotrained_latest.pt"))
        json.dump({**ev, "step": step, "samples": samples}, open(os.path.join(path + ".tmp", "eval.json"), "w"), indent=1)
        if os.path.exists(path): __import__("shutil").rmtree(path, ignore_errors=True)
        os.rename(path + ".tmp", path)
    if fsdp: dist.barrier()
    del full


# ------------------------------------------------------------------------------------------------------------------ main
def main():
    a = get_args(); ddp = "RANK" in os.environ and not a.no_fsdp
    if ddp: dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size(); dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))); torch.cuda.set_device(dev)
    else: rank, world = 0, 1; dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu_test else "cpu")
    is0 = rank == 0; torch.manual_seed(a.seed); os.makedirs(a.out, exist_ok=True)
    if dev.type == "cuda": torch.backends.cuda.matmul.allow_tf32 = True
    norm = Normalizer.load(a.stats).to(dev)
    from nla.unclip.encoder import load_encoder, ActEncoder
    if a.cpu_test:   # tiny random prior + random encoder, real recipe json (weights ignored)
        cfg = {"d_input": 5120, "d_model": 256, "d_mlp": 512, "n_layers": 2}; sd = None
        spec = json.load(open(a.encoder_json)); spec = dict(spec, act_hidden=[64, 64])
        from nla.contrastive.model import ActMLP
        enc = ActEncoder(spec, ActMLP(5120, (64, 64), spec["d_e"]).state_dict(), Normalizer.load(a.stats)).to(dev)
    else:
        m = torch.load(os.path.join(a.prior, "model.pt"), map_location="cpu", mmap=True); cfg = m["args"]
        sd = m.get("model") if a.prior_weights == "raw" and m.get("model") is not None else torch.load(os.path.join(a.prior, "ema.pt"), map_location="cpu", mmap=True)["ema"]
        enc = load_encoder(a.encoder_json, dev)
    d_e = enc.d_e
    if not hasattr(enc, "from_standardised"):
        raise RuntimeError("nla.unclip.encoder.ActEncoder needs from_standardised()")
    model = build(a, cfg, sd, d_e, dev); del sd
    n_prior = sum(p.numel() for p in model.prior.parameters()); n_ad = model.n_adapter_params()
    fsdp = ddp
    if fsdp: model = shard(model, a)
    adapter_ids = {id(p) for p in model.adapter_parameters()}
    groups = [{"params": [p for p in model.parameters() if id(p) in adapter_ids], "lr": a.lr, "base_lr": a.lr}]
    prior_params = [p for p in model.parameters() if id(p) not in adapter_ids]
    if a.freeze_prior:
        for p in prior_params: p.requires_grad_(False)
    else: groups.append({"params": prior_params, "lr": a.prior_lr, "base_lr": a.prior_lr})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=a.wd)
    trainable = [p for g in groups for p in g["params"]]
    global_batch = a.batch * a.grad_accum * world; total_steps = int(a.total_samples // global_batch)
    step, samples = 0, 0; latest = os.path.join(a.out, "latest")
    if os.path.exists(os.path.join(latest, "meta.json")):
        step, samples = load_latest(latest, model, opt, fsdp)
        if is0: print(f"[dec] resumed from step {step} ({samples/1e6:.0f}M samples)", flush=True)
    if is0:
        print(f"[dec] prior {n_prior/1e9:.2f}B ({'frozen' if a.freeze_prior else f'lr {a.prior_lr}'}) from {a.prior} ({a.prior_weights}); adapter {n_ad/1e6:.0f}M (lr {a.lr}), d_e {d_e}, e_scale {enc.e_scale}; world {world}, fsdp {fsdp}, global batch {global_batch}, total steps {total_steps}, p_uncond {a.p_uncond}", flush=True)
        if not a.no_wandb:
            try:
                import wandb; wandb.init(project=a.wandb_project, name=a.wandb_name or a.tag, config=vars(a) | {"n_prior": n_prior, "n_adapter": n_ad, "world": world, "global_batch": global_batch}, resume="allow", id=(a.wandb_name or a.tag))
            except Exception as ex: print(f"[dec] wandb disabled: {ex}", flush=True); a.no_wandb = True
    held = torch.load(a.heldout, map_location="cpu")["acts"][: a.eval_n].float() if (a.heldout and os.path.exists(a.heldout)) else None
    clean = load_clean1(a.clean1) if (a.clean1 and os.path.exists(a.clean1)) else None
    if a.cpu_test and clean is not None: held = clean[:64]; clean = clean[64:128]
    if a.shard_dir: feeder = ShardFeeder(a.shard_dir, norm, a.batch, dev, a.stop_file, a.stream_timeout)
    elif a.parquet_glob: feeder = StaticFeeder(a.parquet_glob, norm, a.batch, dev, rank, world, a.max_rows, a.seed)
    else: raise SystemExit("--shard-dir or --parquet-glob")
    t_start = time.time(); t_log = time.time(); loss_acc, n_acc = 0.0, 0; next_snapshot = (samples // a.snapshot_every_samples + 1) * a.snapshot_every_samples
    stop_flag = torch.zeros(1, device=dev); ev = {}
    if step == 0:
        ev = evaluate(model, enc, norm, held, clean, a, dev)
        if is0:
            print("[eval@0] " + " ".join(f"{k}={v:.4f}" for k, v in ev.items() if "_t0." not in k and not k.startswith("_")), flush=True)
            if not a.no_wandb: import wandb; wandb.log(ev | {"train/samples": samples}, step=0)
    while step < total_steps:
        xs = [feeder.next() for _ in range(a.grad_accum)]
        ended = any(x is None for x in xs)
        stop_flag.fill_(1.0 if (ended or (time.time() - t_start) / 3600 > a.max_hours or (a.stop_file and os.path.exists(a.stop_file)) or (a.max_steps and step >= a.max_steps)) else 0.0)
        if ddp: dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
        if stop_flag.item() > 0:
            if is0: print(f"[dec] stopping at step {step}: {'stream ended' if ended else 'time budget / stop file / max steps'}", flush=True)
            break
        lr = lr_at(step, total_steps, 1.0, 0.0, a.min_lr_frac) if step >= a.warmup_steps else (step + 1) / a.warmup_steps   # warm-up then cosine (on the step index past warm-up)
        if step >= a.warmup_steps:
            pr = min(1.0, (step - a.warmup_steps) / max(1, total_steps - a.warmup_steps)); lr = a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * pr))
        for g in opt.param_groups: g["lr"] = g["base_lr"] * lr
        opt.zero_grad(set_to_none=True); loss_sum = 0.0
        for x0 in xs:
            with torch.no_grad(): e = enc.from_standardised(x0)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                loss, _, _ = cond_fm_loss(model, x0, None, None, p_uncond=a.p_uncond, cvec=e)
            (loss / len(xs)).backward(); loss_sum += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(trainable, a.clip); gn = gn.full_tensor() if hasattr(gn, "full_tensor") else gn
        opt.step(); step += 1; samples += global_batch; loss_acc += loss_sum / len(xs); n_acc += 1
        if is0 and (step % 50 == 0 or step <= 3):
            dt = time.time() - t_log; t_log = time.time()
            rec = {"train/loss": loss_acc / n_acc, "train/lr_adapter": a.lr * lr, "train/lr_prior": (0.0 if a.freeze_prior else a.prior_lr * lr), "train/grad_norm": float(gn), "train/samples": samples,
                   "train/samples_per_s": n_acc * global_batch / dt, "train/hours": (time.time() - t_start) / 3600, "train/ready_shards": n_ready(a.shard_dir) if a.shard_dir else -1}
            print(f"[dec] step {step} loss {rec['train/loss']:.4f} lr {lr:.3f}x gn {float(gn):.2f} {rec['train/samples_per_s']/1e3:.1f}k/s ready {rec['train/ready_shards']} {samples/1e6:.1f}M | mem {torch.cuda.max_memory_allocated()/2**30 if dev.type == 'cuda' else 0:.0f} GiB", flush=True)
            if not a.no_wandb: import wandb; wandb.log(rec, step=step)
            loss_acc, n_acc = 0.0, 0
        if step % a.eval_every == 0:
            ev = evaluate(model, enc, norm, held, clean, a, dev)
            if is0:
                print(f"[eval@{step}] " + " ".join(f"{k}={v:.4f}" for k, v in ev.items() if "_t0." not in k and not k.startswith("_")), flush=True)
                if not a.no_wandb: import wandb; wandb.log(ev | {"train/samples": samples}, step=step)
                json.dump({**ev, "step": step, "samples": samples}, open(os.path.join(a.out, f"eval_{step:07d}.json"), "w"), indent=1)
        if a.ckpt_every and step % a.ckpt_every == 0:
            t_ck = time.time(); save_latest(latest, model, opt, step, samples, a, fsdp, rank)
            if samples >= next_snapshot:
                if not ev or ev.get("_step") != step: ev = evaluate(model, enc, norm, held, clean, a, dev); ev["_step"] = step
                save_snapshot(os.path.join(a.out, f"snap_{int(samples/1e6):06d}M"), model, a, cfg, d_e, step, samples, ev, fsdp, rank); next_snapshot += a.snapshot_every_samples
            if is0: print(f"[dec] checkpoint @ step {step} ({time.time()-t_ck:.0f}s)", flush=True)
    if ddp: dist.barrier()
    if a.ckpt_every: save_latest(latest, model, opt, step, samples, a, fsdp, rank)
    if a.snap_at_end or a.max_steps:
        ev = evaluate(model, enc, norm, held, clean, a, dev)
        if is0: print(f"[eval@{step}] " + " ".join(f"{k}={v:.4f}" for k, v in ev.items() if "_t0." not in k and not k.startswith("_")), flush=True)
        save_snapshot(os.path.join(a.out, f"snap_{int(samples/1e6):06d}M"), model, a, cfg, d_e, step, samples, ev, fsdp, rank)
    if is0:
        json.dump({"step": step, "samples": samples, "hours": (time.time() - t_start) / 3600}, open(os.path.join(a.out, "summary.json"), "w"))
        print(f"[dec] done: step {step}, {samples/1e6:.1f}M samples", flush=True)
    if ddp: dist.destroy_process_group()


if __name__ == "__main__":
    main()
