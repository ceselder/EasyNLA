"""Stage 2: train the text-conditional adapter of the activation flow on (activation, explanation) pairs.
Frozen: the prior denoiser (bf16, from a stage-1 snapshot) and the encoder = the target LM truncated at --enc-layer (token states of the
explanation, same residual space as h). Trainable: the per-block cross-attention adapters (fp32). Single GPU.
Evals (held-out pairs): conditional vs unconditional vs SHUFFLED-condition FM loss per noise level; FVE of the x0-prediction at high noise
(the "conditional FVE", comparable to the MSE critic); source-match accuracy (true h vs 7 distractor explanations by denoising loss)."""
import argparse, json, math, os, time
import numpy as np, pyarrow.parquet as pq, torch, torch.nn.functional as F
from nla.flow.model import Denoiser, Normalizer
from nla.flow.cond_model import CondDenoiser, cond_fm_loss


class _Stop(Exception): pass


def load_encoder(base, layer, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base); tok.padding_side = "right"
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval(); model.requires_grad_(False)
    inner = model.model; layers = inner.layers if hasattr(inner, "layers") else inner.language_model.layers
    del layers[layer + 1:]
    cap = {}
    def hook(_m, _i, out): cap["h"] = out[0] if isinstance(out, tuple) else out; raise _Stop()
    layers[layer].register_forward_hook(hook)
    @torch.no_grad()
    def encode(texts, max_len=192):
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len, add_special_tokens=False)
        ids, am = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        try: model(input_ids=ids, attention_mask=am, use_cache=False)
        except _Stop: pass
        h = cap.pop("h"); mask = am.bool(); mask[:, 0] = False          # drop position 0 (attention-sink token) from the keys
        return h, mask
    return encode, tok


def load_pairs(parquet, n, skip=0):
    """Row-batched read (a single 500k x 5120 list array overflows pyarrow's int32 offsets)."""
    from nla.schema import extract_explanation
    pf = pq.ParquetFile(parquet); acts, zs, seen = [], [], 0
    for rb in pf.iter_batches(batch_size=4096, columns=["activation_vector", "response"]):
        if seen + rb.num_rows <= skip: seen += rb.num_rows; continue
        a = np.asarray(rb.column("activation_vector").flatten(), dtype=np.float32).reshape(rb.num_rows, -1)
        z = [(extract_explanation(r) or r or "").strip() for r in rb.column("response").to_pylist()]
        lo = max(0, skip - seen); a, z = a[lo:], z[lo:]; seen += rb.num_rows
        keep = [i for i, zz in enumerate(z) if zz]; acts.append(torch.tensor(a[keep])); zs += [z[i] for i in keep]
        if len(zs) >= n: break
    acts = torch.cat(acts)[:n]; return acts, zs[:n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prior", required=True, help="snapshot dir with model.pt (raw weights) or ema.pt"); p.add_argument("--prior-weights", default="raw", choices=["raw", "ema"])
    p.add_argument("--stats", required=True); p.add_argument("--base", required=True); p.add_argument("--enc-layer", type=int, default=42)
    p.add_argument("--train-parquet", required=True); p.add_argument("--val-parquet", required=True); p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=5000); p.add_argument("--batch", type=int, default=64); p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--p-uncond", type=float, default=0.1); p.add_argument("--n-slots", type=int, default=8); p.add_argument("--n-heads", type=int, default=4); p.add_argument("--d-head", type=int, default=64); p.add_argument("--gate-rank", type=int, default=128)
    p.add_argument("--max-train", type=int, default=200000); p.add_argument("--eval-every", type=int, default=500); p.add_argument("--eval-n", type=int, default=1024); p.add_argument("--match-n", type=int, default=256)
    p.add_argument("--wandb", default="nla-glp"); p.add_argument("--tag", default="cond"); p.add_argument("--seed", type=int, default=0); p.add_argument("--max-hours", type=float, default=22.0)
    a = p.parse_args(); dev = "cuda"; torch.manual_seed(a.seed); os.makedirs(a.out, exist_ok=True)
    norm = Normalizer.load(a.stats).to(dev)
    m = torch.load(os.path.join(a.prior, "model.pt"), map_location="cpu"); cfg = m["args"]
    sd = m.get("model") if a.prior_weights == "raw" and m.get("model") is not None else torch.load(os.path.join(a.prior, "ema.pt"), map_location="cpu")["ema"]
    prior = Denoiser(cfg["d_input"], cfg["d_model"], cfg["d_mlp"], cfg["n_layers"]); prior.load_state_dict({k: v.float() for k, v in sd.items()})
    prior = prior.to(torch.bfloat16).to(dev).requires_grad_(False)                       # frozen prior in bf16
    encode, tok = load_encoder(a.base, a.enc_layer, dev)
    d_enc = cfg["d_input"]
    model = CondDenoiser(prior, d_enc, a.n_slots, a.n_heads, a.d_head, a.gate_rank).to(dev)
    for blk in model.blocks: blk.read.float(); blk.gate_mod.float()                     # adapter in fp32
    n_ad = model.n_adapter_params(); print(f"[cond] prior {cfg['n_layers']} blocks ({a.prior_weights} weights from {a.prior}); adapter {n_ad/1e6:.1f}M params; encoder layer {a.enc_layer}", flush=True)
    tr_acts, tr_z = load_pairs(a.train_parquet, a.max_train); va_acts, va_z = load_pairs(a.val_parquet, a.eval_n + a.match_n)
    print(f"[cond] {len(tr_z)} train pairs, {len(va_z)} val pairs; d_enc {d_enc}", flush=True)
    from nla.schema import compute_predict_mean_baselines, resolve_target_scale, normalize_activation
    msf = math.sqrt(cfg["d_input"]); _, base_mse = compute_predict_mean_baselines(va_acts[: a.eval_n], msf)
    opt = torch.optim.AdamW(model.adapter_parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    use_wandb = bool(a.wandb)
    if use_wandb:
        try: import wandb; wandb.init(project=a.wandb, name=a.tag, config=vars(a) | {"adapter_params": n_ad})
        except Exception as e: print("[cond] wandb off:", e, flush=True); use_wandb = False

    def enc_batch(zs):
        with torch.autocast("cuda", dtype=torch.bfloat16): return encode(zs)

    @torch.no_grad()
    def evaluate(step):
        model.eval(); out = {}
        x0 = norm.normalize(va_acts[: a.eval_n].to(dev)); g = torch.Generator(device=dev).manual_seed(0)
        perm = torch.randperm(a.eval_n, generator=torch.Generator().manual_seed(1))
        zs = va_z[: a.eval_n]; zs_shuf = [zs[i] for i in perm.tolist()]
        for t_val in (0.1, 0.3, 0.5, 0.7, 0.9):
            eps = torch.randn(x0.shape, device=dev, generator=g); t = torch.full((a.eval_n,), t_val, device=dev)
            for name, cond in (("uncond", None), ("cond", zs), ("shuf", zs_shuf)):
                ls = []
                for i in range(0, a.eval_n, 128):
                    e, mk = enc_batch(cond[i:i+128]) if cond is not None else (None, None)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        x_t = (1 - t_val) * x0[i:i+128] + t_val * eps[i:i+128]; v = model(x_t, t[i:i+128], e, mk)
                    ls.append(F.mse_loss(v.float(), (eps[i:i+128] - x0[i:i+128]).float(), reduction="sum").item() / x0.shape[1])
                out[f"eval/fm_{name}_t{t_val}"] = sum(ls) / a.eval_n
        for name in ("uncond", "cond", "shuf"): out[f"eval/fm_{name}"] = sum(out[f"eval/fm_{name}_t{t}"] for t in (0.1, 0.3, 0.5, 0.7, 0.9)) / 5
        out["eval/gain_bits_per_dim"] = (out["eval/fm_uncond"] - out["eval/fm_cond"]) / (2 * math.log(2))   # ELBO-flavoured: 0.5*Δmse per dim in nats -> bits (uniform-t weighting)
        # conditional FVE: x0-prediction at high noise, x0_hat = x_t - t*v ; NLA convention: unit-L2 to sqrt(d), MSE, predict-mean baseline
        t_val = 0.9; eps = torch.randn(x0.shape, device=dev, generator=torch.Generator(device=dev).manual_seed(7)); t = torch.full((a.eval_n,), t_val, device=dev)
        preds = []
        for i in range(0, a.eval_n, 128):
            e, mk = enc_batch(zs[i:i+128]); x_t = (1 - t_val) * x0[i:i+128] + t_val * eps[i:i+128]
            with torch.autocast("cuda", dtype=torch.bfloat16): v = model(x_t, t[i:i+128], e, mk).float()
            preds.append(norm.denormalize(x_t - t_val * v))
        pred = normalize_activation(torch.cat(preds), msf); gold = normalize_activation(va_acts[: a.eval_n].to(dev), msf)
        mse = ((pred - gold) ** 2).mean().item(); out["eval/cond_fve_x0_t0.9"] = 100 * (1 - mse / base_mse); out["eval/cond_mse_x0_t0.9"] = mse
        # source match: for each of match_n held-out pairs, rank the true explanation against 7 distractors by mean denoising loss over 8 (t, eps)
        mh = va_acts[a.eval_n: a.eval_n + a.match_n]; mz = va_z[a.eval_n: a.eval_n + a.match_n]; K = 8; correct = 0
        rng = np.random.default_rng(3); gm = torch.Generator(device=dev).manual_seed(11)
        for i in range(a.match_n):
            cands = [mz[i]] + [mz[j] for j in rng.choice([j for j in range(a.match_n) if j != i], K - 1, replace=False)]
            order = rng.permutation(K); cands = [cands[o] for o in order]; true_idx = int(np.where(order == 0)[0][0])   # shuffle: ties must not favour the true one
            e, mk = enc_batch(cands); xi = norm.normalize(mh[i:i+1].to(dev)).expand(K, -1); score = torch.zeros(K, device=dev)
            for r in range(8):
                tt = torch.rand(1, device=dev, generator=gm).expand(K); ee = torch.randn(xi.shape[1:], device=dev, generator=gm)[None].expand(K, -1)
                with torch.autocast("cuda", dtype=torch.bfloat16): v = model((1 - tt)[:, None] * xi + tt[:, None] * ee, tt, e, mk).float()
                score += ((v - (ee - xi)) ** 2).mean(-1)
            correct += int(score.argmin().item() == true_idx and (score < score[true_idx]).sum().item() == 0 and (score == score[true_idx]).sum().item() == 1)   # strict best
        out["eval/source_match_acc"] = correct / a.match_n; out["eval/source_match_chance"] = 1 / K
        model.train()
        print(f"[eval@{step}] fm uncond {out['eval/fm_uncond']:.4f} cond {out['eval/fm_cond']:.4f} shuf {out['eval/fm_shuf']:.4f} | gain {out['eval/gain_bits_per_dim']*x0.shape[1]:.1f} bits/activation | cond FVE(x0@0.9) {out['eval/cond_fve_x0_t0.9']:.1f}% | source-match {100*out['eval/source_match_acc']:.1f}% (chance 12.5%)", flush=True)
        json.dump(out, open(os.path.join(a.out, f"eval_{step:06d}.json"), "w"), indent=1)
        return out

    rng = torch.Generator().manual_seed(a.seed); t0 = time.time(); evaluate(0)
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, tr_acts.shape[0], (a.batch,), generator=rng)
        x0 = norm.normalize(tr_acts[idx].to(dev)); e, mk = enc_batch([tr_z[i] for i in idx.tolist()])
        lr = a.lr * min(1.0, step / a.warmup) * (0.5 * (1 + math.cos(math.pi * min(1.0, step / a.steps))) * 0.9 + 0.1)
        for gp in opt.param_groups: gp["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _, used = cond_fm_loss(model, x0, e, mk, p_uncond=a.p_uncond)
        opt.zero_grad(set_to_none=True); loss.backward(); gn = torch.nn.utils.clip_grad_norm_(model.adapter_parameters(), 1.0); opt.step()
        if step % 50 == 0:
            print(f"[cond] step {step} loss {loss.item():.4f} ({'cond' if used else 'uncond'}) lr {lr:.2e} gn {float(gn):.3f} {(time.time()-t0)/step:.2f}s/step", flush=True)
            if use_wandb: wandb.log({"train/loss": loss.item(), "train/lr": lr, "train/grad_norm": float(gn)}, step=step)
        if step % a.eval_every == 0 or step == a.steps or (time.time() - t0) / 3600 > a.max_hours:
            ev = evaluate(step)
            if use_wandb: wandb.log(ev, step=step)
            torch.save({"adapter": {k: v for k, v in model.state_dict().items() if ".read." in k or ".gate_mod." in k}, "args": vars(a), "prior_cfg": cfg, "step": step}, os.path.join(a.out, "adapter_latest.pt"))
            if (time.time() - t0) / 3600 > a.max_hours: break
    print("[cond] done", flush=True)


if __name__ == "__main__":
    main()
