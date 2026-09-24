"""Train the DALL-E-2-style diffusion prior critic p(h_j | h_i, z) on the Qwen3-8B activation store (nlt/prior/model.py).

  python -m nlt.prior.train --data-dir /vol/data/qwen3_8b --out /vol/prior/<tag> --tag <tag> \
      --pools "lens=0.4:/vol/z/lensdiff_v1/train/L*_part*.parquet,teacher=0.3:/vol/z/teacher-sonnet-v1/train/*.parquet" \
      --val-sets "lens_L1:/vol/z/lensdiff_v1/val/L1.parquet,teacher_v1:/vol/z/teacher-sonnet-v1/val/*.parquet@1"
  python -m nlt.prior.train ... --text-synth depth           # SMOKE GATE: 'from layer i to layer j' (must be read within ~200k rows)

Recipe (Ramesh et al. 2022): x0-prediction MSE, text dropped with p_uncond (one net = conditional + unconditional path), Adam(beta2 .999, eps 1e-8),
lr 1.2e-4 with warmup + cosine decay, EMA weights (the EMA copy is what gets saved as 'model' and evaluated).
Target = delta = n(h_j) - n(h_i) in the pooled j-agnostic affine space (no rms division, no squash; per-layer stats forbidden).
Monitoring on held-out val rows AFTER the fixed eval set: FM-proxy PMI / content vs a depth-matched wrong text, and SPOT EXACT bits (Heun,
paired Hutchinson) for the true text, the depth-matched text and a random pair's text. Decisions use exact bits only.
"""
from __future__ import annotations
import argparse, copy, json, math, os, time
import numpy as np, torch
from nlt.data.dataset import ActStore, GlobalNorm
from nlt.critic.model import make_x0
from nlt.prior.model import DiffusionPrior
from nlt.prior.data import TextPools, load_val_sets, depth_tag_texts, dm_partner

T_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)


@torch.no_grad()
def proxy_losses(model, x0, h_i, eps_bank, enc, mask):
    out = torch.zeros(len(T_GRID), x0.shape[0])
    for ti, t in enumerate(T_GRID):
        tt = torch.full((x0.shape[0],), float(t), device=x0.device); x_t = (1 - tt)[:, None] * x0 + tt[:, None] * eps_bank[ti]
        with torch.autocast("cuda", dtype=torch.bfloat16): v = model(x_t, tt, h_i, enc=enc, enc_mask=mask)
        out[ti] = ((v - (eps_bank[ti] - x0)) ** 2).mean(-1).cpu()
    return out


def evaluate(model, store_val, norm, encoder, val_sets, dev, eps_bank, a, probe_bank):
    """proxy + spot-exact metrics per val set (EMA model). Returns flat dict."""
    from nlt.eval_bits.exact import exact_logp
    model.eval(); d = norm.mean.numel(); out = {}; B = 128
    for label, (rows, ii, jj, texts, _) in val_sets.items():
        n = len(rows); dmp = dm_partner(ii, jj); dm_texts = [texts[q] for q in dmp]; rp_texts = [texts[(q + n // 2) % n] for q in range(n)]
        L_c = torch.zeros(len(T_GRID), n); L_u = torch.zeros(len(T_GRID), n); L_dm = torch.zeros(len(T_GRID), n)
        ne = min(a.spot_exact_n, n); lp = {k: torch.zeros(ne) for k in ("c", "u", "dm", "rp")}
        for s in range(0, n, B):
            r, i, j = rows[s:s + B], ii[s:s + B], jj[s:s + B]
            h_i, x0, _, _ = make_x0(norm, store_val.gather(r, i, dev), store_val.gather(r, j, dev), "delta", False, 0.0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                enc, mask = encoder(texts[s:s + B]); enc_d, mask_d = encoder(dm_texts[s:s + B]); enc_r, mask_r = encoder(rp_texts[s:s + B])
            eb = [e[s:s + B].to(dev) for e in eps_bank]
            L_c[:, s:s + B] = proxy_losses(model, x0, h_i, eb, enc, mask); L_u[:, s:s + B] = proxy_losses(model, x0, h_i, eb, None, None); L_dm[:, s:s + B] = proxy_losses(model, x0, h_i, eb, enc_d, mask_d)
            if s < ne:
                m = min(B, ne - s); sl = slice(0, m)
                kw = dict(n_steps=a.spot_ode_steps, probes=1, probe_bank=probe_bank)
                lp["c"][s:s + m] = exact_logp(model, x0[sl], h_i[sl], enc=enc[sl], enc_mask=mask[sl], **kw).cpu()
                lp["u"][s:s + m] = exact_logp(model, x0[sl], h_i[sl], **kw).cpu()
                lp["dm"][s:s + m] = exact_logp(model, x0[sl], h_i[sl], enc=enc_d[sl], enc_mask=mask_d[sl], **kw).cpu()
                lp["rp"][s:s + m] = exact_logp(model, x0[sl], h_i[sl], enc=enc_r[sl], enc_mask=mask_r[sl], **kw).cpu()
        pmi = (d / 2) * (L_u - L_c).mean(0) / math.log(2); cont = (d / 2) * (L_dm - L_c).mean(0) / math.log(2)
        out[f"{label}/fm_cond"] = float(L_c.mean()); out[f"{label}/fm_uncond"] = float(L_u.mean()); out[f"{label}/proxy_pmi_bits"] = float(pmi.mean())
        out[f"{label}/proxy_content_bits"] = float(cont.mean()); out[f"{label}/proxy_p_z_gt_dm"] = float((cont > 0).float().mean())
        if ne:
            e = {k: (v - lp["u"]) / math.log(2) for k, v in lp.items() if k != "u"}
            out[f"{label}/exact_pmi_bits"] = float(e["c"].mean()); out[f"{label}/exact_pmi_sem"] = float(e["c"].std() / math.sqrt(ne))
            out[f"{label}/exact_dm_bits"] = float(e["dm"].mean()); out[f"{label}/exact_rp_bits"] = float(e["rp"].mean())
            out[f"{label}/exact_content_bits"] = float((e["c"] - e["dm"]).mean()); out[f"{label}/exact_content_sem"] = float((e["c"] - e["dm"]).std() / math.sqrt(ne))
            out[f"{label}/exact_p_z_gt_dm"] = float((e["c"] > e["dm"]).float().mean()); out[f"{label}/exact_p_z_gt_null"] = float((e["c"] > 0).float().mean())
            out[f"{label}/exact_nll_uncond_bits_per_dim"] = float(-lp["u"].mean() / (d * math.log(2)))
    model.train(); return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--out", required=True); p.add_argument("--tag", default="prior")
    p.add_argument("--pools", default=None, help="name=weight:glob[;glob][@verb],..."); p.add_argument("--val-sets", default=None, help="label:path[@verb],...")
    p.add_argument("--batch-mode", default="pool", choices=["pool", "mixed"], help="pool: one text pool per step (short batches stay short); mixed: all pools in every batch")
    p.add_argument("--text-synth", default=None, choices=[None, "depth"], help="SMOKE GATE: synthetic depth-tag text instead of --pools")
    p.add_argument("--width", type=int, default=1024); p.add_argument("--depth", type=int, default=12); p.add_argument("--heads", type=int, default=16); p.add_argument("--k-chunks", type=int, default=8); p.add_argument("--mlp-ratio", type=int, default=4)
    p.add_argument("--param", default="x0", choices=["x0", "v", "x0res"], help="x0: predict the unnoised target (paper); v: velocity (plain FM); x0res: x0 loss with the velocity parametrisation")
    p.add_argument("--t-min", type=float, default=0.02); p.add_argument("--bidir-tail", type=int, default=0); p.add_argument("--x0-scale", type=float, default=0.0, help="constant target rescale inside the net (0 = set from the data rms)")
    p.add_argument("--steps", type=int, default=4000); p.add_argument("--batch", type=int, default=1024); p.add_argument("--micro-batch", type=int, default=256, help="rows per backward pass (gradient accumulation up to --batch)"); p.add_argument("--lr", type=float, default=1.2e-4); p.add_argument("--warmup", type=int, default=300); p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--null-dm", type=float, default=0.0, help="DECISIONS v1.16 null-dm regulariser weight: at the SAME (x_t, t, eps), the velocity under a depth-matched WRONG text (another row of the micro-batch with the same (i,j), else same j, else rolled) is pulled to the no-text velocity (detached), so a wrong text earns no bits")
    p.add_argument("--beta2", type=float, default=0.999); p.add_argument("--ema", type=float, default=0.999); p.add_argument("--p-uncond", type=float, default=0.1); p.add_argument("--grad-clip", type=float, default=1.0); p.add_argument("--lr-floor", type=float, default=0.05)
    p.add_argument("--enc-model", default="Qwen/Qwen3-0.6B"); p.add_argument("--enc-layer", type=int, default=20); p.add_argument("--enc-max-len", type=int, default=192)
    p.add_argument("--eval-every", type=int, default=500); p.add_argument("--eval-n", type=int, default=512); p.add_argument("--eval-offset", type=int, default=4096); p.add_argument("--spot-exact-n", type=int, default=128); p.add_argument("--spot-ode-steps", type=int, default=16)
    p.add_argument("--save-every", type=int, default=1000); p.add_argument("--keep-every", type=int, default=0); p.add_argument("--max-hours", type=float, default=20.0)
    p.add_argument("--max-train-pos", type=int, default=None); p.add_argument("--data-device", default="cuda"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", default="nlt-qwen3-8b"); p.add_argument("--wandb-entity", default="octahedral-systems"); p.add_argument("--resume", default=None)
    a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(a.out, exist_ok=True); t_start = time.time()
    norm = GlobalNorm.load(os.path.join(a.data_dir, "stats.pt"), "affine").to(dev)
    store = ActStore(a.data_dir, "train", device=a.data_device, max_pos=a.max_train_pos); store_val = ActStore(a.data_dir, "val", device=a.data_device); d = store.d
    from nlt.critic.text_encoder import TextEncoder
    encoder = TextEncoder(a.enc_model, a.enc_layer, dev, a.enc_max_len)
    # ---- text
    pools = None
    if a.text_synth == "depth":
        import pyarrow.parquet as pq
        vp = pq.read_table(os.path.join(a.data_dir, "pairs_val.parquet")).to_pandas(); vp = vp[vp["pos_idx"].isin(store_val.row_of)].iloc[a.eval_offset:a.eval_offset + a.eval_n]
        vi = torch.tensor(vp["i"].values.astype(np.int64)); vj = torch.tensor(vp["j"].values.astype(np.int64))
        val_sets = {"depthtag": (store_val.rows_for(vp["pos_idx"].values), vi, vj, depth_tag_texts(vi, vj), vp["pair_id"].tolist())}
        print(f"[train] SMOKE GATE: synthetic depth-tag text, e.g. {val_sets['depthtag'][3][0]!r}", flush=True)
    else:
        assert a.pools and a.val_sets
        pools = TextPools(a.pools, os.path.join(a.data_dir, "pairs_train.parquet"), store)
        val_sets = load_val_sets(a.val_sets, os.path.join(a.data_dir, "pairs_val.parquet"), store_val, a.eval_offset, a.eval_n)
    g_eval = torch.Generator().manual_seed(1234); eps_bank = [torch.randn(a.eval_n, d, generator=g_eval) for _ in T_GRID]
    from nlt.eval_bits.exact import make_probe_bank
    probe_bank = make_probe_bank(a.spot_ode_steps, 1, d, torch.Generator().manual_seed(4321))
    # ---- model
    gen = torch.Generator().manual_seed(a.seed)
    if a.x0_scale <= 0:
        r0, i0, j0 = store.sample_pairs(8192, gen); _, x0s, _, _ = make_x0(norm, store.gather(r0, i0, dev), store.gather(r0, j0, dev), "delta", False, 0.0)
        a.x0_scale = float(x0s.pow(2).mean().sqrt()); print(f"[train] x0 rms over 8192 sampled pairs = {a.x0_scale:.3f} -> x0_scale", flush=True)
    model = DiffusionPrior(d, a.width, a.depth, a.heads, a.k_chunks, encoder.d_enc, a.enc_max_len, a.param, a.t_min, a.bidir_tail, a.mlp_ratio, a.x0_scale).to(dev)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    print(f"[train] DiffusionPrior {model.n_params()/1e6:.0f}M params: width {a.width} depth {a.depth} heads {a.heads} K {a.k_chunks} param {a.param} bidir_tail {a.bidir_tail}; batch {a.batch} x {a.steps} steps = {a.batch*a.steps/1e6:.2f}M rows", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, a.beta2), eps=1e-8, weight_decay=a.wd)
    step0 = 0
    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, map_location="cpu"); model.load_state_dict(ck["model_raw"]); ema_model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step0 = ck["step"]; print(f"[train] resumed {a.resume} @ {step0}", flush=True)
    def lr_at(s):
        if s < a.warmup: return a.lr * (s + 1) / a.warmup
        pr = (s - a.warmup) / max(1, a.steps - a.warmup); return a.lr * (a.lr_floor + (1 - a.lr_floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, pr))))
    args_save = vars(a) | {"src_rms": 0, "squash": 0.0, "cond": "text", "target": "delta"}
    def save(step, name="ckpt_latest.pt", with_opt=True):
        d_ = {"model": ema_model.state_dict(), "step": step, "args": args_save, "config": model.config(), "d_enc": encoder.d_enc}
        if with_opt: d_["model_raw"] = model.state_dict(); d_["opt"] = opt.state_dict()
        torch.save(d_, os.path.join(a.out, name))
    import wandb
    wandb.init(project=a.wandb, entity=a.wandb_entity, name=a.tag, config=vars(a) | {"n_params": model.n_params(), "n_train_pos": store.N, "n_text_rows": pools.n_rows if pools else 0}, resume="allow")
    ema_p = [q for q in ema_model.parameters()]; raw_p = [q for q in model.parameters()]
    t0 = time.time(); ema_loss = None; best = None; rows_seen = step0 * a.batch
    for step in range(step0, a.steps):
        if pools is not None: rows, i, j, texts, names = pools.sample(a.batch, gen, a.batch_mode)
        else: rows, i, j = store.sample_pairs(a.batch, gen); texts = depth_tag_texts(i, j)
        for g_ in opt.param_groups: g_["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True); keep = torch.rand(a.batch, device=dev) >= a.p_uncond                       # text dropout -> unconditional rows
        if a.micro_batch < a.batch:                                                                                  # sort the step's rows by text length so each micro-batch pads to ITS OWN longest text (mean over the step is unchanged)
            order = sorted(range(a.batch), key=lambda q: len(texts[q])); rows, i, j = rows[order], i[order], j[order]; texts = [texts[q] for q in order]; keep = keep[torch.tensor(order, device=dev)]
        l_all = torch.zeros(a.batch, device=dev); v_all = torch.zeros(a.batch, device=dev); mask_T = 0; null_acc = 0.0
        for s0 in range(0, a.batch, a.micro_batch):                                                                  # gradient accumulation: same batch, bounded activation memory
            sl = slice(s0, min(a.batch, s0 + a.micro_batch)); nb = sl.stop - sl.start
            h_i, x0, _, _ = make_x0(norm, store.gather(rows[sl], i[sl], dev), store.gather(rows[sl], j[sl], dev), "delta", False, 0.0)
            with torch.autocast("cuda", dtype=torch.bfloat16): enc, mask = encoder(texts[sl])
            mask = mask & keep[sl][:, None]; mask_T = max(mask_T, int(mask.shape[1]))
            t = torch.rand(nb, device=dev); eps = torch.randn_like(x0)
            with torch.autocast("cuda", dtype=torch.bfloat16): l, v_mse = model.loss(x0, h_i, t, eps, enc, mask)
            step_loss = l.mean()
            if a.null_dm > 0:
                perm = torch.tensor(dm_partner(i[sl], j[sl]), device=dev); x_t = (1 - t)[:, None] * x0 + t[:, None] * eps
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    with torch.no_grad(): v_null = model(x_t, t, h_i)
                    v_dm = model(x_t, t, h_i, enc=enc[perm], enc_mask=mask[perm])
                nl = ((v_dm - v_null.detach()) ** 2).mean(-1) / (model.x0_scale ** 2); null_acc += float(nl.mean()) * nb / a.batch; step_loss = step_loss + a.null_dm * nl.mean()
            (step_loss * nb / a.batch).backward(); l_all[sl] = l.detach(); v_all[sl] = v_mse
        l, v_mse = l_all, v_all; loss = l.mean()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip); opt.step()
        with torch.no_grad():
            dec = min(a.ema, (1 + step) / (10 + step)); torch._foreach_lerp_(ema_p, raw_p, 1 - dec)
        rows_seen += a.batch; ema_loss = loss.item() if ema_loss is None else 0.98 * ema_loss + 0.02 * loss.item()
        if step % 25 == 0:
            el = time.time() - t0; log = {"train/loss": loss.item(), "train/loss_ema": ema_loss, "train/v_mse": float(v_mse.mean()), "train/loss_cond": float(l[keep].mean()) if keep.any() else float("nan"), "train/loss_uncond": float(l[~keep].mean()) if (~keep).any() else float("nan"),
                                          "train/lr": lr_at(step), "train/grad_norm": float(gn), "train/null_dm_loss": null_acc, "train/rows_per_s": (step - step0 + 1) * a.batch / max(1e-6, el), "train/rows_seen": rows_seen, "train/step_s": el / max(1, step - step0 + 1), "train/seq_len": mask_T + 3 * a.k_chunks + 2, "train/pool": (names[0] if pools is not None else "synth")}
            wandb.log(log, step=step)
            if step % 100 == 0: print(f"[train] step {step} loss {loss.item():.4f} ema {ema_loss:.4f} v_mse {float(v_mse.mean()):.4f} cond {log['train/loss_cond']:.4f} uncond {log['train/loss_uncond']:.4f} lr {lr_at(step):.2e} gn {float(gn):.2f} {log['train/rows_per_s']:.0f} rows/s S={log['train/seq_len']}", flush=True)
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            te = time.time(); out = evaluate(ema_model, store_val, norm, encoder, val_sets, dev, eps_bank, a, probe_bank); out["eval/seconds"] = time.time() - te; out["eval/rows_seen"] = rows_seen
            wandb.log(out, step=step); json.dump({"step": step + 1, "rows_seen": rows_seen, "scalars": out}, open(os.path.join(a.out, "eval_latest.json"), "w"), indent=1)
            print(f"[eval@{step+1} rows {rows_seen}] " + " | ".join(f"{k}={v:.3f}" for k, v in out.items() if "exact" in k or "content" in k or "p_z" in k), flush=True)
            score = float(np.mean([v for k, v in out.items() if k.endswith("/exact_content_bits")] or [out.get(next(iter(val_sets)) + "/proxy_content_bits", 0.0)]))
            if best is None or score > best[0]: best = (score, step + 1); save(step + 1, "ckpt_best.pt", with_opt=False); json.dump({"step": step + 1, "score": score}, open(os.path.join(a.out, "best.json"), "w"))
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps: save(step + 1)
        if a.keep_every and (step + 1) % a.keep_every == 0: save(step + 1, f"ckpt_step{step + 1:06d}.pt", with_opt=False)
        if (time.time() - t_start) / 3600 > a.max_hours: print("[train] max hours reached", flush=True); save(step + 1); break
    save(a.steps, "ckpt_final.pt", with_opt=False); wandb.finish(); print("[train] DONE", flush=True)


if __name__ == "__main__":
    main()
