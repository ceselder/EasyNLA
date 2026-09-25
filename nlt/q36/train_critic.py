"""Diffusion-prior critic p(u_j | u_i, z) for Qwen3.6-27B (port of nlt/prior/train.py to d = 5120 and the DIRECTION convention of critic_data.py).

  python train_critic.py --data-dir /vol/q36/data --out /vol/q36/critic/<tag> --tag <tag> \
      --pools "craft=0.5:/vol/q36/text/craft_v1/train/*.parquet,jlens=0.2:...,olens_j=0.15:...,olens_delta=0.15:..." \
      --val-sets "craft:/vol/q36/text/craft_v1/val/*.parquet,jlens:..." --width 1536 --depth 16 --steps 3000 --batch 1024 --micro-batch 128

Recipe = the validated 8B 480M arm: velocity head, x_t = (1-t) y + t eps, AdamW lr 1.2e-4 (betas .9/.999, wd .01), warm-up + cosine to 5%, EMA .999,
text dropout 0.1 (+ --uncond-frac of every batch = random (position, i<j) pairs with no text so the unconditional path sees every pair type).
Target y = sqrt(d) u_j s (radial dequantisation, s ~ lognormal(0, sigma_r)); source = sqrt(d) u_i; u = unit(h - mu_layer). No layer index anywhere.
Monitoring: proxy PMI / content, spot exact bits (Heun --spot-ode-steps, paired probes) for true / depth-matched / random-pair text, and the centred cos
of the conditional-mean estimate (x0-hat from pure noise) with u_j, with and without text. Decisions use exact bits.
"""
from __future__ import annotations
import argparse, copy, json, math, os, time
os.environ["HF_HUB_OFFLINE"] = "0"; os.environ["HF_HOME"] = "/vol/hf_cache"          # the nlt volume (rw): Qwen3-0.6B text encoder downloads once; the 27B is not needed here
import numpy as np, torch
from critic_data import Store, Directions, TextPools, load_val_sets, dm_partner

T_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)


def x0_hat_from_noise(model, eps, src, enc=None, mask=None):
    """conditional-mean estimate: one velocity step from pure noise (t = 1): x0 = x_1 - v(x_1, 1)"""
    tt = torch.ones(eps.shape[0], device=eps.device)
    with torch.autocast("cuda", dtype=torch.bfloat16): v = model(eps, tt, src, enc=enc, enc_mask=mask)
    return eps - v.float()


@torch.no_grad()
def proxy_losses(model, x0, src, eps_bank, enc, mask):
    out = torch.zeros(len(T_GRID), x0.shape[0])
    for ti, t in enumerate(T_GRID):
        tt = torch.full((x0.shape[0],), float(t), device=x0.device); x_t = (1 - tt)[:, None] * x0 + tt[:, None] * eps_bank[ti]
        with torch.autocast("cuda", dtype=torch.bfloat16): v = model(x_t, tt, src, enc=enc, enc_mask=mask)
        out[ti] = ((v.float() - (eps_bank[ti] - x0)) ** 2).mean(-1).cpu()
    return out


def evaluate(model, store_val, dirs, encoder, val_sets, dev, eps_bank, s_bank, a, probe_bank):
    from nlt.eval_bits.exact import exact_logp
    model.eval(); d = dirs.d; out = {}; B = 64
    for label, (rows, ii, jj, texts, _) in val_sets.items():
        n = len(rows); dmp = dm_partner(ii, jj); dm_texts = [texts[q] for q in dmp]; rp_texts = [texts[(q + n // 2) % n] for q in range(n)]
        L_c = torch.zeros(len(T_GRID), n); L_u = torch.zeros(len(T_GRID), n); L_dm = torch.zeros(len(T_GRID), n)
        ne = min(a.spot_exact_n, n); lp = {k: torch.zeros(ne) for k in ("c", "u", "dm", "rp")}; cosm = {k: torch.zeros(n) for k in ("c", "u", "dm")}
        for s0 in range(0, n, B):
            r, i, j = rows[s0:s0 + B], ii[s0:s0 + B], jj[s0:s0 + B]; nb = len(r)
            h_i = store_val.gather(r, i, dev); h_j = store_val.gather(r, j, dev); src = dirs.source(h_i, i); x0, _ = dirs.target(h_j, j, s=s_bank[s0:s0 + nb].to(dev))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                enc, mask = encoder(texts[s0:s0 + nb]); enc_d, mask_d = encoder(dm_texts[s0:s0 + nb]); enc_r, mask_r = encoder(rp_texts[s0:s0 + nb])
            eb = [e[s0:s0 + nb].to(dev) for e in eps_bank]
            L_c[:, s0:s0 + nb] = proxy_losses(model, x0, src, eb, enc, mask); L_u[:, s0:s0 + nb] = proxy_losses(model, x0, src, eb, None, None); L_dm[:, s0:s0 + nb] = proxy_losses(model, x0, src, eb, enc_d, mask_d)
            with torch.no_grad():
                for k_, (e_, m_) in {"c": (enc, mask), "u": (None, None), "dm": (enc_d, mask_d)}.items():
                    cosm[k_][s0:s0 + nb] = dirs.cos_to_target(x0_hat_from_noise(model, eb[-1], src, e_, m_), h_j, j).cpu()
            if s0 < ne:
                m = min(nb, ne - s0); sl = slice(0, m); kw = dict(n_steps=a.spot_ode_steps, probes=1, probe_bank=probe_bank)
                lp["c"][s0:s0 + m] = exact_logp(model, x0[sl], src[sl], enc=enc[sl], enc_mask=mask[sl], **kw).cpu(); lp["u"][s0:s0 + m] = exact_logp(model, x0[sl], src[sl], **kw).cpu()
                lp["dm"][s0:s0 + m] = exact_logp(model, x0[sl], src[sl], enc=enc_d[sl], enc_mask=mask_d[sl], **kw).cpu(); lp["rp"][s0:s0 + m] = exact_logp(model, x0[sl], src[sl], enc=enc_r[sl], enc_mask=mask_r[sl], **kw).cpu()
        pmi = (d / 2) * (L_u - L_c).mean(0) / math.log(2); cont = (d / 2) * (L_dm - L_c).mean(0) / math.log(2)
        out[f"{label}/fm_cond"] = float(L_c.mean()); out[f"{label}/fm_uncond"] = float(L_u.mean()); out[f"{label}/proxy_pmi_bits"] = float(pmi.mean()); out[f"{label}/proxy_content_bits"] = float(cont.mean()); out[f"{label}/proxy_p_z_gt_dm"] = float((cont > 0).float().mean())
        for k_ in cosm: out[f"{label}/cos_mean_{k_}"] = float(cosm[k_].mean())
        if ne:
            e = {k: (v - lp["u"]) / math.log(2) for k, v in lp.items() if k != "u"}
            out[f"{label}/exact_pmi_bits"] = float(e["c"].mean()); out[f"{label}/exact_pmi_sem"] = float(e["c"].std() / math.sqrt(ne)); out[f"{label}/exact_dm_bits"] = float(e["dm"].mean()); out[f"{label}/exact_rp_bits"] = float(e["rp"].mean())
            out[f"{label}/exact_content_bits"] = float((e["c"] - e["dm"]).mean()); out[f"{label}/exact_content_sem"] = float((e["c"] - e["dm"]).std() / math.sqrt(ne))
            out[f"{label}/exact_p_z_gt_dm"] = float((e["c"] > e["dm"]).float().mean()); out[f"{label}/exact_p_z_gt_null"] = float((e["c"] > 0).float().mean()); out[f"{label}/exact_nll_uncond_bits_per_dim"] = float(-lp["u"].mean() / (d * math.log(2)))
    model.train(); return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--stats", default=None); p.add_argument("--out", required=True); p.add_argument("--tag", default="critic")
    p.add_argument("--pools", required=True); p.add_argument("--val-sets", required=True); p.add_argument("--band", default=None, help="comma list of layers for the no-text random pairs (default: all stored)")
    p.add_argument("--width", type=int, default=1536); p.add_argument("--depth", type=int, default=16); p.add_argument("--heads", type=int, default=16); p.add_argument("--k-chunks", type=int, default=8); p.add_argument("--mlp-ratio", type=int, default=4)
    p.add_argument("--param", default="v", choices=["x0", "v", "x0res"]); p.add_argument("--t-min", type=float, default=0.02); p.add_argument("--x0-scale", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=3000); p.add_argument("--batch", type=int, default=1024); p.add_argument("--micro-batch", type=int, default=128); p.add_argument("--lr", type=float, default=1.2e-4); p.add_argument("--warmup", type=int, default=300); p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--uncond-steps", type=int, default=0, help="PRETRAIN p(u_j | u_i) with NO text on random band pairs from the whole store for this many steps (unlabelled pairs are free), then switch to the text pools (tip from the NLA flow-critic session: freeze-then-condition worked best there)"); p.add_argument("--beta2", type=float, default=0.999); p.add_argument("--ema", type=float, default=0.999); p.add_argument("--p-uncond", type=float, default=0.1); p.add_argument("--uncond-frac", type=float, default=0.15); p.add_argument("--grad-clip", type=float, default=1.0); p.add_argument("--lr-floor", type=float, default=0.05)
    p.add_argument("--sigma-r", type=float, default=0.1); p.add_argument("--enc-model", default="Qwen/Qwen3-0.6B"); p.add_argument("--enc-layer", type=int, default=20); p.add_argument("--enc-max-len", type=int, default=192)
    p.add_argument("--eval-every", type=int, default=500); p.add_argument("--eval-n", type=int, default=256); p.add_argument("--eval-offset", type=int, default=2048, help="val pairs before this index are the FIXED test set (eval_bits.py); monitoring uses pairs after it"); p.add_argument("--spot-exact-n", type=int, default=64); p.add_argument("--spot-ode-steps", type=int, default=16)
    p.add_argument("--save-every", type=int, default=1000); p.add_argument("--keep-every", type=int, default=0); p.add_argument("--max-hours", type=float, default=20.0); p.add_argument("--max-train-pos", type=int, default=None); p.add_argument("--data-device", default="cuda"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", default="nlt-qwen36-27b"); p.add_argument("--wandb-entity", default="octahedral-systems"); p.add_argument("--no-wandb", action="store_true"); p.add_argument("--resume", default=None)
    a = p.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(a.out, exist_ok=True); t_start = time.time()
    from nlt.prior.model import DiffusionPrior
    from nlt.critic.text_encoder import TextEncoder
    from nlt.eval_bits.exact import make_probe_bank
    stats_path = a.stats or os.path.join(a.data_dir, "layer_stats.pt"); dirs = Directions(stats_path, a.sigma_r, dev)
    store = Store(a.data_dir, "train", device=a.data_device, max_pos=a.max_train_pos); store_val = Store(a.data_dir, "val", device=a.data_device); d = store.d
    band = [int(x) for x in a.band.split(",")] if a.band else store.layers
    encoder = TextEncoder(a.enc_model, a.enc_layer, dev, a.enc_max_len)
    pools = TextPools(a.pools, os.path.join(a.data_dir, "pairs_train.parquet"), store)
    val_sets = load_val_sets(a.val_sets, os.path.join(a.data_dir, "pairs_val.parquet"), store_val, a.eval_n, offset=a.eval_offset)
    g_eval = torch.Generator().manual_seed(1234); eps_bank = [torch.randn(a.eval_n, d, generator=g_eval) for _ in T_GRID]; s_bank = torch.exp(a.sigma_r * torch.randn(a.eval_n, generator=g_eval))
    probe_bank = make_probe_bank(a.spot_ode_steps, 1, d, torch.Generator().manual_seed(4321))
    gen = torch.Generator().manual_seed(a.seed)
    if a.x0_scale <= 0:
        r0, i0, j0 = store.sample_pairs(4096, gen, band); y0, _ = dirs.target(store.gather(r0, j0, dev), j0); a.x0_scale = float(y0.pow(2).mean().sqrt()); print(f"[train] target rms over 4096 pairs = {a.x0_scale:.3f} -> x0_scale", flush=True)
    model = DiffusionPrior(d, a.width, a.depth, a.heads, a.k_chunks, encoder.d_enc, a.enc_max_len, a.param, a.t_min, 0, a.mlp_ratio, a.x0_scale).to(dev)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    print(f"[train] DiffusionPrior {model.n_params()/1e6:.0f}M params: width {a.width} depth {a.depth} K {a.k_chunks} param {a.param}; batch {a.batch} x {a.steps} = {a.batch*a.steps/1e6:.2f}M rows; band {band}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, a.beta2), eps=1e-8, weight_decay=a.wd); step0 = 0
    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, map_location="cpu"); model.load_state_dict(ck["model_raw"]); ema_model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step0 = ck["step"]; print(f"[train] resumed {a.resume} @ {step0}", flush=True)
    def lr_at(s):
        if s < a.warmup: return a.lr * (s + 1) / a.warmup
        pr = (s - a.warmup) / max(1, a.steps - a.warmup); return a.lr * (a.lr_floor + (1 - a.lr_floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, pr))))
    args_save = vars(a) | {"norm": "directions", "stats_path": stats_path, "cond": "text", "target": "u_j", "layers": store.layers, "band": band}
    def save(step, name="ckpt_latest.pt", with_opt=True):
        d_ = {"model": ema_model.state_dict(), "step": step, "args": args_save, "config": model.config() | {"norm": "directions", "sigma_r": a.sigma_r}, "d_enc": encoder.d_enc}
        if with_opt: d_["model_raw"] = model.state_dict(); d_["opt"] = opt.state_dict()
        torch.save(d_, os.path.join(a.out, name))
    wb = None
    if not a.no_wandb and os.environ.get("WANDB_API_KEY"):
        import wandb; os.environ.setdefault("WANDB_DIR", "/root/wandb"); os.makedirs("/root/wandb", exist_ok=True)
        wb = wandb.init(project=a.wandb, entity=a.wandb_entity, name=a.tag, config=vars(a) | {"n_params": model.n_params(), "n_train_pos": store.N, "n_text_rows": pools.n_rows}, resume="allow")
    ema_p = list(ema_model.parameters()); raw_p = list(model.parameters()); t0 = time.time(); ema_loss = None; best = None; rows_seen = step0 * a.batch
    n_unc = int(round(a.batch * a.uncond_frac)); n_txt = a.batch - n_unc
    for step in range(step0, a.steps):
        if step < a.uncond_steps:                                                      # unconditional pretraining phase: every row text-free, pairs from the whole store
            rows, i, j = store.sample_pairs(a.batch, gen, band); texts = [""] * a.batch; pname = "uncond"
        else:
            rows, i, j, texts, pname = pools.sample(n_txt, gen)
            if n_unc:
                ru, iu, ju = store.sample_pairs(n_unc, gen, band); rows = torch.cat([rows, ru]); i = torch.cat([i, iu]); j = torch.cat([j, ju]); texts = texts + [""] * n_unc
        if step == a.uncond_steps and a.uncond_steps > 0: print(f"[train] unconditional pretraining done ({a.uncond_steps} steps x {a.batch}); switching to the text pools", flush=True); save(step, "ckpt_uncond.pt", with_opt=False)
        for g_ in opt.param_groups: g_["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True); keep = torch.rand(a.batch, device=dev) >= a.p_uncond
        order = sorted(range(a.batch), key=lambda q: len(texts[q])); rows, i, j = rows[order], i[order], j[order]; texts = [texts[q] for q in order]; keep = keep[torch.tensor(order, device=dev)]
        l_all = torch.zeros(a.batch, device=dev); v_all = torch.zeros(a.batch, device=dev); mask_T = 0
        for s0 in range(0, a.batch, a.micro_batch):
            sl = slice(s0, min(a.batch, s0 + a.micro_batch)); nb = sl.stop - sl.start
            h_i = store.gather(rows[sl], i[sl], dev); h_j = store.gather(rows[sl], j[sl], dev); src = dirs.source(h_i, i[sl]); x0, _ = dirs.target(h_j, j[sl])
            with torch.autocast("cuda", dtype=torch.bfloat16): enc, mask = encoder(texts[sl])
            mask = mask & keep[sl][:, None]; mask_T = max(mask_T, int(mask.shape[1]))
            t = torch.rand(nb, device=dev); eps = torch.randn_like(x0)
            with torch.autocast("cuda", dtype=torch.bfloat16): l, v_mse = model.loss(x0, src, t, eps, enc, mask)
            (l.mean() * nb / a.batch).backward(); l_all[sl] = l.detach(); v_all[sl] = v_mse
        loss = l_all.mean(); gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip); opt.step()
        with torch.no_grad():
            dec = min(a.ema, (1 + step) / (10 + step)); torch._foreach_lerp_(ema_p, raw_p, 1 - dec)
        rows_seen += a.batch; ema_loss = loss.item() if ema_loss is None else 0.98 * ema_loss + 0.02 * loss.item()
        if step % 25 == 0:
            el = time.time() - t0; has_txt = keep & torch.tensor([len(z) > 0 for z in texts], device=dev)
            log = {"train/loss": loss.item(), "train/loss_ema": ema_loss, "train/v_mse": float(v_all.mean()), "train/loss_text": float(l_all[has_txt].mean()) if has_txt.any() else float("nan"), "train/loss_notext": float(l_all[~has_txt].mean()) if (~has_txt).any() else float("nan"),
                   "train/lr": lr_at(step), "train/grad_norm": float(gn), "train/rows_per_s": (step - step0 + 1) * a.batch / max(1e-6, el), "train/rows_seen": rows_seen, "train/seq_len": mask_T + 3 * a.k_chunks + 2}
            if wb: wb.log(log, step=step)
            if step % 100 == 0: print(f"[train] step {step} loss {loss.item():.4f} ema {ema_loss:.4f} text {log['train/loss_text']:.4f} notext {log['train/loss_notext']:.4f} lr {lr_at(step):.2e} gn {float(gn):.2f} {log['train/rows_per_s']:.0f} rows/s S={log['train/seq_len']} pool={pname}", flush=True)
        if ((step + 1) % a.eval_every == 0 and step + 1 >= a.uncond_steps) or step + 1 == a.steps:
            te = time.time(); out = evaluate(ema_model, store_val, dirs, encoder, val_sets, dev, eps_bank, s_bank, a, probe_bank); out["eval/seconds"] = time.time() - te; out["eval/rows_seen"] = rows_seen
            if wb: wb.log(out, step=step)
            json.dump({"step": step + 1, "rows_seen": rows_seen, "scalars": out}, open(os.path.join(a.out, "eval_latest.json"), "w"), indent=1)
            print(f"[eval@{step+1} rows {rows_seen}] " + " | ".join(f"{k}={v:.3f}" for k, v in out.items() if ("exact" in k or "content" in k or "p_z" in k or "cos_mean" in k)), flush=True)
            score = float(np.mean([v for k, v in out.items() if k.endswith("/exact_content_bits")] or [0.0]))
            if best is None or score > best[0]: best = (score, step + 1); save(step + 1, "ckpt_best.pt", with_opt=False); json.dump({"step": step + 1, "score": score}, open(os.path.join(a.out, "best.json"), "w"))
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps: save(step + 1)
        if a.keep_every and (step + 1) % a.keep_every == 0: save(step + 1, f"ckpt_step{step + 1:06d}.pt", with_opt=False)
        if (time.time() - t_start) / 3600 > a.max_hours: print("[train] max hours reached", flush=True); save(step + 1); break
    save(a.steps, "ckpt_final.pt", with_opt=False)
    if wb: wb.finish()
    print("[train] DONE", flush=True)


if __name__ == "__main__":
    main()
