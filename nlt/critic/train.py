"""Train the transcoder critic p(h_j | h_i [, z]) by conditional flow matching on the multi-layer Qwen3-8B activation store.

  python -m nlt.critic.train --data-dir /vol/data/qwen3_8b --cond none  --tag none_v0 --out /vol/critic/none_v0
  python -m nlt.critic.train --data-dir ... --cond depth --tag depth_v0 ...                   (forbidden diagnostic)
  python -m nlt.critic.train --data-dir ... --cond text --text-parquet /vol/text/lensdiff_v1.parquet --tag text_v0 ...

Held-out eval (fixed val pairs, fixed t grid and eps): FM loss overall / by gap / by j; for depth and text: the same batch with the
condition dropped -> FM-proxy PMI in bits = (d/2) * (L_uncond - L_cond) / ln 2 with shared eps (common random numbers).
"""
from __future__ import annotations
import argparse, json, math, os, time
import numpy as np, torch
from nlt.data.dataset import ActStore, GlobalNorm
from nlt.data.extract import K_LO, K_HI
from nlt.critic.model import PairDenoiser, make_x0, pair_fm_loss

GAP_BUCKETS = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 25)]
T_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)


def gap_bucket(g):
    for lo, hi in GAP_BUCKETS:
        if lo <= g <= hi: return f"{lo}-{hi}" if lo != hi else f"{lo}"
    return "other"


def load_text_pairs(paths, pairs_parquet, verbosity=None):
    """text rows joined to the fixed pair list -> DataFrame [pair_id, pos_idx, i, j, text, verbosity, source]"""
    import pandas as pd, pyarrow.parquet as pq
    pairs = pq.read_table(pairs_parquet, columns=["pair_id", "pos_idx", "i", "j"]).to_pandas()
    rows = []
    for p in paths:
        if p.endswith(".jsonl"): rows.append(pd.read_json(p, lines=True))
        else: rows.append(pq.read_table(p).to_pandas())
    tx = pd.concat(rows, ignore_index=True)
    if "verbosity" not in tx: tx["verbosity"] = 0
    if "source" not in tx: tx["source"] = "text"
    if verbosity is not None: tx = tx[tx["verbosity"].isin(verbosity)]
    tx = tx[tx["text"].astype(str).str.strip().str.len() > 0]
    df = tx.merge(pairs, on="pair_id", how="inner")
    return df


def smoke_texts(store, rows, i, j, tok):
    """PLUMBING TEST ONLY: text = the next token (a real fact about h_j). Never a research result."""
    nt = store.meta["next_token_id"].values[rows.cpu().numpy()]
    return [f"next token: {tok.decode([int(x)])!r}" for x in nt]


@torch.no_grad()
def evaluate(model, store_val, norm, a, val_rows, val_i, val_j, val_text, encoder, dev, eps_bank, prefix="eval"):
    """fixed pairs, fixed eps (per t) -> loss tables. Returns a flat dict of scalars + a nested breakdown."""
    model.eval()
    B = 256; n = len(val_rows); d = norm.mean.numel()
    L_c = torch.zeros(len(T_GRID), n); L_u = torch.zeros(len(T_GRID), n); mse_id = torch.zeros(n); mse_x0 = torch.zeros(n); var_j = torch.zeros(n)
    for s in range(0, n, B):
        rows, i, j = val_rows[s:s + B], val_i[s:s + B], val_j[s:s + B]
        h_i, x0 = make_x0(norm, store_val.gather(rows, i, dev), store_val.gather(rows, j, dev), a.target)
        depth = torch.stack([i, j], 1).to(dev) if a.cond == "depth" else None
        enc = mask = None
        if a.cond == "text":
            with torch.autocast("cuda", dtype=torch.bfloat16): enc, mask = encoder(val_text[s:s + B])
        mse_id[s:s + B] = ((x0 - (0 if a.target == "delta" else h_i)) ** 2).mean(-1).cpu()       # identity transcoder h_j := h_i
        for ti, t in enumerate(T_GRID):
            tt = torch.full((len(rows),), t, device=dev); eps = eps_bank[ti][s:s + B].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lc, _, _ = pair_fm_loss(model, x0, h_i, tt, eps, depth=depth, enc=enc, enc_mask=mask)
            L_c[ti, s:s + B] = lc.cpu()
            if a.cond != "none":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lu, _, _ = pair_fm_loss(model, x0, h_i, tt, eps, depth=None, enc=None, enc_mask=None)
                L_u[ti, s:s + B] = lu.cpu()
            if t == 0.9:            # x0-prediction at high noise ~ conditional mean -> FVE-like number comparable to an MSE transcoder
                x_t = (1 - tt)[:, None] * x0 + tt[:, None] * eps
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    v = model(x_t, tt, h_i, depth=depth, depth_has=None if depth is None else torch.ones(len(rows), dtype=torch.bool, device=dev), enc=enc, enc_mask=mask)
                mse_x0[s:s + B] = ((x_t - tt[:, None] * v - x0) ** 2).mean(-1).cpu()
        var_j[s:s + B] = (x0 ** 2).mean(-1).cpu()          # energy of the target around the GLOBAL mean (the j-agnostic reference)
    model.train()
    gaps = (val_j - val_i).numpy(); js = val_j.numpy()
    out = {f"{prefix}/fm_loss": float(L_c.mean()), f"{prefix}/fm_loss_t0.9": float(L_c[-1].mean()), f"{prefix}/fm_loss_t0.1": float(L_c[0].mean()),
           f"{prefix}/x0_mse_t0.9": float(mse_x0.mean()), f"{prefix}/fve_x0_t0.9_vs_globalmean": float(1 - mse_x0.sum() / var_j.sum()),
           f"{prefix}/identity_mse": float(mse_id.mean()), f"{prefix}/fve_identity_vs_globalmean": float(1 - mse_id.sum() / var_j.sum())}
    br = {"t_grid": list(T_GRID), "fm_loss_by_t": L_c.mean(1).tolist(), "by_gap": {}, "by_j": {}}
    if a.cond != "none":
        pmi = (d / 2) * (L_u - L_c).mean(0) / math.log(2)                       # bits per pair, shared eps
        out[f"{prefix}/pmi_proxy_bits"] = float(pmi.mean()); out[f"{prefix}/pmi_proxy_bits_median"] = float(pmi.median()); out[f"{prefix}/fm_loss_uncond"] = float(L_u.mean())
        br["pmi_by_t_bits"] = ((d / 2) * (L_u - L_c).mean(1) / math.log(2)).tolist()
    for lo, hi in GAP_BUCKETS:
        m = (gaps >= lo) & (gaps <= hi)
        if m.sum() == 0: continue
        key = f"{lo}-{hi}" if lo != hi else f"{lo}"; e = {"n": int(m.sum()), "fm_loss": float(L_c[:, m].mean()), "identity_mse": float(mse_id[m].mean()), "x0_mse_t0.9": float(mse_x0[m].mean())}
        if a.cond != "none": e["pmi_proxy_bits"] = float(((d / 2) * (L_u[:, m] - L_c[:, m]).mean(0) / math.log(2)).mean())
        br["by_gap"][key] = e; out[f"{prefix}_gap/fm_loss_gap{key}"] = e["fm_loss"]
        if a.cond != "none": out[f"{prefix}_gap/pmi_bits_gap{key}"] = e["pmi_proxy_bits"]
    for jj in range(K_LO + 1, K_HI + 1):
        m = js == jj
        if m.sum() == 0: continue
        e = {"n": int(m.sum()), "fm_loss": float(L_c[:, m].mean()), "identity_mse": float(mse_id[m].mean()), "x0_mse_t0.9": float(mse_x0[m].mean())}
        if a.cond != "none": e["pmi_proxy_bits"] = float(((d / 2) * (L_u[:, m] - L_c[:, m]).mean(0) / math.log(2)).mean())
        br["by_j"][str(jj)] = e
    return out, br


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--out", required=True); p.add_argument("--tag", default="critic")
    p.add_argument("--cond", default="none", choices=["none", "depth", "text"]); p.add_argument("--target", default="hj", choices=["hj", "delta"]); p.add_argument("--norm", default="affine", choices=["affine", "scalar"])
    p.add_argument("--d-model", type=int, default=2048); p.add_argument("--d-mlp", type=int, default=8192); p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-slots", type=int, default=8); p.add_argument("--n-heads", type=int, default=4); p.add_argument("--d-head", type=int, default=64); p.add_argument("--gate-rank", type=int, default=128)
    p.add_argument("--steps", type=int, default=5000); p.add_argument("--batch", type=int, default=512); p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--warmup", type=int, default=200); p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--lr-decay", default="cosine", choices=["none", "cosine"]); p.add_argument("--p-uncond", type=float, default=0.15); p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-train-pos", type=int, default=None); p.add_argument("--data-device", default="cuda", help="where the fp16 store lives (cuda on a B200; cpu on smaller GPUs)")
    p.add_argument("--eval-every", type=int, default=500); p.add_argument("--eval-n", type=int, default=4096); p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--text-parquet", default=None, help="comma-separated text files [pair_id, text, verbosity, source] (cond=text)"); p.add_argument("--text-verbosity", default=None, help="comma list of verbosity levels to train on (default all)")
    p.add_argument("--text-smoke", action="store_true", help="PLUMBING TEST: synthetic 'next token: X' text instead of --text-parquet")
    p.add_argument("--enc-model", default="Qwen/Qwen3-0.6B"); p.add_argument("--enc-layer", type=int, default=20); p.add_argument("--enc-max-len", type=int, default=128)
    p.add_argument("--wandb", default="nlt-qwen3-8b"); p.add_argument("--wandb-entity", default="octahedral-systems"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default=None); p.add_argument("--max-hours", type=float, default=20.0)
    a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(a.out, exist_ok=True); t_start = time.time()
    norm = GlobalNorm.load(os.path.join(a.data_dir, "stats.pt"), a.norm).to(dev)
    store = ActStore(a.data_dir, "train", device=a.data_device, max_pos=a.max_train_pos)
    store_val = ActStore(a.data_dir, "val", device=a.data_device)
    d = store.d
    # ---- fixed val pairs (from the finalize pair list; disjoint docs) + fixed eps bank
    import pyarrow.parquet as pq
    vp = pq.read_table(os.path.join(a.data_dir, "pairs_val.parquet")).to_pandas()
    vp = vp[vp["pos_idx"].isin(store_val.row_of)].iloc[: a.eval_n]
    val_rows = store_val.rows_for(vp["pos_idx"].values); val_i = torch.tensor(vp["i"].values); val_j = torch.tensor(vp["j"].values)
    g_eval = torch.Generator().manual_seed(1234); eps_bank = [torch.randn(len(val_rows), d, generator=g_eval) for _ in T_GRID]
    # ---- text
    encoder = None; text_df = None; val_text = None; tok8 = None
    if a.cond == "text":
        from nlt.critic.text_encoder import TextEncoder
        encoder = TextEncoder(a.enc_model, a.enc_layer, dev, a.enc_max_len)
        if a.text_smoke:
            from transformers import AutoTokenizer
            tok8 = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"); val_text = smoke_texts(store_val, val_rows, val_i, val_j, tok8)
            print("[train] TEXT SMOKE MODE: synthetic next-token texts (plumbing test, not a result)", flush=True)
        else:
            verb = [int(x) for x in a.text_verbosity.split(",")] if a.text_verbosity else None
            text_df = load_text_pairs(a.text_parquet.split(","), os.path.join(a.data_dir, "pairs_train.parquet"), verb)
            text_df = text_df[text_df["pos_idx"].isin(store.row_of)].reset_index(drop=True)
            vdf = load_text_pairs(a.text_parquet.split(","), os.path.join(a.data_dir, "pairs_val.parquet"), verb)
            vdf = vdf.drop_duplicates("pair_id").set_index("pair_id")
            pid = [f"val:{p_}:{i_}:{j_}" for p_, i_, j_ in zip(vp["pos_idx"].values, vp["i"].values, vp["j"].values)]
            have = [x in vdf.index for x in pid]
            keep = np.where(have)[0]; assert len(keep) > 0, "no val pairs have text"
            val_rows, val_i, val_j = val_rows[keep], val_i[keep], val_j[keep]; eps_bank = [e[keep] for e in eps_bank]
            val_text = [vdf.loc[pid[k], "text"] for k in keep]
            print(f"[train] text pairs: train {len(text_df)} (verbosity {sorted(text_df['verbosity'].unique().tolist())}), val {len(keep)}/{len(pid)} with text", flush=True)
    model = PairDenoiser(d, a.d_model, a.d_mlp, a.n_layers, a.cond, d_enc=(encoder.d_enc if encoder else 0), n_slots=a.n_slots, n_heads=a.n_heads, d_head=a.d_head, gate_rank=a.gate_rank, target=a.target).to(dev)
    if encoder: model.d_enc_ = encoder.d_enc
    print(f"[train] {a.cond} critic: {model.n_params()/1e6:.0f}M params, target {a.target}, norm {a.norm}, batch {a.batch}, {a.steps} steps", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd)
    step0 = 0
    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, map_location="cpu"); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step0 = ck["step"]; print(f"[train] resumed from {a.resume} @ {step0}", flush=True)
    def lr_at(s):
        if s < a.warmup: return a.lr * (s + 1) / a.warmup
        if a.lr_decay == "none": return a.lr
        pr = (s - a.warmup) / max(1, a.steps - a.warmup); return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, pr))))
    import wandb
    run = wandb.init(project=a.wandb, entity=a.wandb_entity, name=a.tag, config=vars(a) | {"n_params": model.n_params(), "n_train_pos": store.N, "n_val_pos": store_val.N}, resume="allow")
    gen = torch.Generator().manual_seed(a.seed + step0)
    def save(step, name="ckpt_latest.pt"):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "args": vars(a), "config": model.config(), "d_enc": (encoder.d_enc if encoder else 0)}, os.path.join(a.out, name))
    t0 = time.time(); ema = None
    for step in range(step0, a.steps):
        if a.cond == "text" and not a.text_smoke:
            idx = torch.randint(0, len(text_df), (a.batch,), generator=gen).numpy(); sub = text_df.iloc[idx]
            rows = store.rows_for(sub["pos_idx"].values); i = torch.tensor(sub["i"].values); j = torch.tensor(sub["j"].values); texts = sub["text"].tolist()
        else:
            rows, i, j = store.sample_pairs(a.batch, gen); texts = smoke_texts(store, rows, i, j, tok8) if (a.cond == "text") else None
        h_i, x0 = make_x0(norm, store.gather(rows, i, dev), store.gather(rows, j, dev), a.target)
        depth = torch.stack([i, j], 1).to(dev) if a.cond == "depth" else None
        enc = mask = None
        if a.cond == "text":
            with torch.autocast("cuda", dtype=torch.bfloat16): enc, mask = encoder(texts)
        for g_ in opt.param_groups: g_["lr"] = lr_at(step)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss_vec, t, kept = pair_fm_loss(model, x0, h_i, depth=depth, enc=enc, enc_mask=mask, p_uncond=(a.p_uncond if a.cond != "none" else 0.0))
        loss = loss_vec.mean(); opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip); opt.step()
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % 25 == 0:
            log = {"train/loss": loss.item(), "train/loss_ema": ema, "train/lr": lr_at(step), "train/grad_norm": float(gn), "train/step_s": (time.time() - t0) / max(1, step - step0 + 1)}
            if a.cond != "none":
                log["train/loss_cond"] = float(loss_vec[kept].mean()) if kept.any() else float("nan"); log["train/loss_uncond"] = float(loss_vec[~kept].mean()) if (~kept).any() else float("nan")
            wandb.log(log, step=step)
            if step % 100 == 0: print(f"[train] step {step} loss {loss.item():.4f} ema {ema:.4f} lr {lr_at(step):.2e} gn {float(gn):.2f} {log['train/step_s']:.3f}s/step", flush=True)
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            out, br = evaluate(model, store_val, norm, a, val_rows, val_i, val_j, val_text, encoder, dev, eps_bank)
            wandb.log(out, step=step); json.dump({"step": step + 1, "scalars": out, "breakdown": br}, open(os.path.join(a.out, "eval_latest.json"), "w"), indent=1)
            print(f"[eval@{step+1}] " + " ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in out.items() if "/" in k and "_gap/" not in k), flush=True)
            print("[eval] by gap: " + json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in br["by_gap"].items()}), flush=True)
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps: save(step + 1)
        if (time.time() - t_start) / 3600 > a.max_hours: print("[train] max hours reached", flush=True); save(step + 1); break
    save(a.steps, "ckpt_final.pt"); wandb.finish()
    print("[train] DONE", flush=True)


if __name__ == "__main__":
    main()
