"""Train the TRUNK critic (nlt/trunk/model.py) by conditional flow matching on infra's Qwen3-8B activation store + the text pools on /vol/z.

  python -m nlt.trunk.train --data-dir /vol/data/qwen3_8b --out /vol/trunk/smoke --tag trunk_smoke --prior /vol/critic/none_v1_pooled/ckpt_final.pt \
      --text-parquet "lens:/vol/z/lensdiff_v1/train/L[0-3]_part00.parquet,teacher:/vol/z/teacher-sonnet-v1/train/*.parquet,ao:/vol/z/ao-tgt-v1/train/*.parquet,ao:/vol/z/ao-delta-v1/train/*.parquet" \
      --pool-weights lens:0.45,teacher:0.4,ao:0.15 --val-text "teacher_v1:/vol/z/teacher-sonnet-v1/val/*.parquet@1,lensL1:/vol/z/lensdiff_v1/val/L1.parquet,v0:/vol/z/v0-ao-tsv1/val/*.parquet"

Loss = FM(z) with per-row condition dropout (the empty prefix IS the null path)  +  null_reg * ||v(x, z_rp) - v(x, "")||^2 on a quarter of the batch
       (+ contrast * softplus((L(z) - L(z_dm) + m)/tau) * tau on the kept rows, DECISIONS v1.10 T4, off by default).
Held-out eval (val pairs AFTER the fixed 4096 eval set, one text set at a time, fixed eps per t): proxy PMI bits vs the empty prefix, the
depth-matched shuffle z_dm, P(z > z_dm) -- health numbers only; exact bits come from nlt.trunk.eval_bits.
"""
from __future__ import annotations
import argparse, glob, json, math, os, re, time
import numpy as np, torch
from nlt.data.dataset import ActStore, GlobalNorm
from nlt.critic.model import make_x0, pair_fm_loss
from nlt.critic.train import T_GRID, GAP_BUCKETS
from nlt.trunk.model import TrunkCritic, TextIDs, load_prior


def load_pool(spec, pairs_parquet, row_of):
    """'pool:glob,pool:glob,...' -> DataFrame [pair_id, pos_idx, i, j, text, pool] (rows whose pos_idx is in the store)"""
    import pandas as pd, pyarrow.parquet as pq
    pairs = pq.read_table(pairs_parquet, columns=["pair_id", "pos_idx", "i", "j"]).to_pandas()
    frames = []
    for item in spec.split(","):
        pool, pat = item.split(":", 1)
        verb = None
        if "@" in pat: pat, v_ = pat.rsplit("@", 1); verb = [int(x) for x in v_.split("+")]
        files = sorted(glob.glob(pat)) or [pat]
        for f in files:
            df = pq.read_table(f, columns=[c for c in pq.read_schema(f).names if c in ("pair_id", "text", "verbosity")]).to_pandas()
            if "verbosity" not in df:
                m = re.search(r"[/_]L(\d)m?[_.]", os.path.basename(f)); df["verbosity"] = int(m.group(1)) if m else 0
            if verb is not None: df = df[df["verbosity"].isin(verb)]
            df = df[df["text"].astype(str).str.strip().str.len() > 0]
            df["pool"] = pool; frames.append(df[["pair_id", "text", "verbosity", "pool"]])
            print(f"[pool] {pool}: {os.path.basename(f)} {len(df)} rows", flush=True)
    tx = pd.concat(frames, ignore_index=True).merge(pairs, on="pair_id", how="inner")
    tx = tx[tx["pos_idx"].isin(row_of)].reset_index(drop=True)
    return tx


class PoolSampler:
    def __init__(self, df, weights: dict, seed=0):
        self.df = df; self.rng = np.random.default_rng(seed)
        self.pools = sorted(df["pool"].unique().tolist())
        self.idx = {p: np.where(df["pool"].values == p)[0] for p in self.pools}
        w = np.array([weights.get(p, 1.0) for p in self.pools], dtype=np.float64); self.w = w / w.sum()
        print("[pool] sampling weights: " + ", ".join(f"{p} {w_:.2f} ({len(self.idx[p])} rows)" for p, w_ in zip(self.pools, self.w)), flush=True)

    def sample(self, B):
        counts = self.rng.multinomial(B, self.w)
        out = np.concatenate([self.rng.choice(self.idx[p], n, replace=True) for p, n in zip(self.pools, counts) if n > 0])
        self.rng.shuffle(out); return self.df.iloc[out]


def dm_perm(i, j):
    """depth-matched partner within a batch: another row with the same (i, j), else the same j, else any other row"""
    ii, jj = i.tolist(), j.tolist(); by_ij = {}; by_j = {}
    for q, (a, b) in enumerate(zip(ii, jj)): by_ij.setdefault((a, b), []).append(q); by_j.setdefault(b, []).append(q)
    perm = []
    for q in range(len(jj)):
        c = [r for r in by_ij[(ii[q], jj[q])] if r != q] or [r for r in by_j[jj[q]] if r != q]
        perm.append(c[q % len(c)] if c else (q + len(jj) // 2) % len(jj))
    return torch.tensor(perm)


@torch.no_grad()
def evaluate(model, store_val, norm, space, sets, dev, eps_bank, B=64):
    """sets: label -> (rows, i, j, texts). Proxy PMI (bits) vs the empty prefix and vs the depth-matched shuffle, fixed eps per t."""
    model.eval(); d = norm.mean.numel(); out = {}; br = {}
    for label, (rows, I, J, texts) in sets.items():
        n = len(rows); Lc = torch.zeros(len(T_GRID), n); Lu = torch.zeros(len(T_GRID), n); Ld = torch.zeros(len(T_GRID), n)
        perm = dm_perm(I, J); dm_texts = [texts[int(q)] for q in perm]
        for s in range(0, n, B):
            r, i, j = rows[s:s + B], I[s:s + B], J[s:s + B]
            h_i, x0, log_s, _ = make_x0(norm, store_val.gather(r, i, dev), store_val.gather(r, j, dev), space["target"], space["src_rms"], space["squash"])
            kv, mask = model.encode(texts[s:s + B]); kv_d, mask_d = model.encode(dm_texts[s:s + B])
            for ti, t in enumerate(T_GRID):
                tt = torch.full((len(r),), t, device=dev); eps = eps_bank[ti][s:s + B].to(dev)
                lc, _, _ = pair_fm_loss(model, x0, h_i, tt, eps, enc=kv, enc_mask=mask, log_s=log_s); Lc[ti, s:s + B] = lc.float().cpu()
                lu, _, _ = pair_fm_loss(model, x0, h_i, tt, eps, log_s=log_s); Lu[ti, s:s + B] = lu.float().cpu()
                ld, _, _ = pair_fm_loss(model, x0, h_i, tt, eps, enc=kv_d, enc_mask=mask_d, log_s=log_s); Ld[ti, s:s + B] = ld.float().cpu()
        pmi = (d / 2) * (Lu - Lc).mean(0) / math.log(2); pmi_dm = (d / 2) * (Lu - Ld).mean(0) / math.log(2)
        js = J.numpy()
        e = {"n": n, "pmi_proxy_bits": float(pmi.mean()), "pmi_proxy_median": float(pmi.median()), "dm_proxy_bits": float(pmi_dm.mean()),
             "content_proxy_bits": float((pmi - pmi_dm).mean()), "p_z_beats_dm": float((pmi > pmi_dm).float().mean()),
             "fm_cond": float(Lc.mean()), "fm_uncond": float(Lu.mean()),
             "by_band": {lab: {"pmi": float(pmi[m].mean()), "content": float((pmi - pmi_dm)[m].mean()), "p_beats_dm": float((pmi > pmi_dm)[m].float().mean()), "n": int(m.sum())}
                         for lab, lo, hi in (("pre", 10, 13), ("workspace", 14, 32), ("motor", 33, 34)) for m in [(js >= lo) & (js <= hi)] if m.sum()}}
        br[label] = e
        for k in ("pmi_proxy_bits", "dm_proxy_bits", "content_proxy_bits", "p_z_beats_dm", "fm_cond", "fm_uncond"): out[f"eval_{label}/{k}"] = e[k]
        for lab, v in e["by_band"].items(): out[f"eval_{label}/content_{lab}"] = v["content"]; out[f"eval_{label}/pbeat_{lab}"] = v["p_beats_dm"]
    model.train(); return out, br


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--out", required=True); p.add_argument("--tag", default="trunk")
    p.add_argument("--prior", required=True, help="blind PairDenoiser ckpt (its space is used: target / src_rms / squash / stats)")
    p.add_argument("--trunk-id", default="Qwen/Qwen3-8B"); p.add_argument("--n-layers", type=int, default=24); p.add_argument("--lora-r", type=int, default=64); p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--n-act-tokens", type=int, default=4); p.add_argument("--fresh-every", type=int, default=4); p.add_argument("--fresh-heads", type=int, default=8); p.add_argument("--fresh-dhead", type=int, default=128)
    p.add_argument("--max-len", type=int, default=128); p.add_argument("--no-grad-ckpt", action="store_true")
    p.add_argument("--text-parquet", required=True, help="pool:glob[@verb+verb],... train text files"); p.add_argument("--pool-weights", default="", help="pool:w,... sampling weights (default equal)")
    p.add_argument("--val-text", default="", help="label:glob[@verb],... val text sets for the in-training proxy eval"); p.add_argument("--eval-n", type=int, default=256); p.add_argument("--eval-offset", type=int, default=4096)
    p.add_argument("--steps", type=int, default=3000); p.add_argument("--batch", type=int, default=64); p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--lr-lora", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=100); p.add_argument("--wd", type=float, default=0.01); p.add_argument("--grad-clip", type=float, default=1.0); p.add_argument("--lr-decay", default="cosine", choices=["none", "cosine"])
    p.add_argument("--p-uncond", type=float, default=0.3); p.add_argument("--null-reg", type=float, default=1.0); p.add_argument("--null-frac", type=float, default=0.25)
    p.add_argument("--contrast", type=float, default=0.0); p.add_argument("--contrast-tau", type=float, default=0.005); p.add_argument("--contrast-margin", type=float, default=0.005)
    p.add_argument("--data-device", default="cpu"); p.add_argument("--val-device", default="cuda"); p.add_argument("--max-train-pos", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=500); p.add_argument("--save-every", type=int, default=500); p.add_argument("--max-hours", type=float, default=10.0)
    p.add_argument("--wandb", default="nlt-qwen3-8b"); p.add_argument("--wandb-entity", default="octahedral-systems"); p.add_argument("--seed", type=int, default=0); p.add_argument("--resume", default=None)
    a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(a.out, exist_ok=True); t_start = time.time()
    prior, space, paa = load_prior(a.prior, dev)
    norm = GlobalNorm.load(space["stats"] if os.path.exists(space["stats"]) else os.path.join(a.data_dir, "stats.pt"), "affine").to(dev)
    print(f"[train] prior {a.prior} (step {space['prior_step']}), space {space}", flush=True)
    model = TrunkCritic(prior, a.trunk_id, a.n_layers, a.lora_r, a.lora_alpha, a.n_act_tokens, a.fresh_every, a.fresh_heads, a.fresh_dhead, grad_ckpt=not a.no_grad_ckpt, max_len=a.max_len, device=dev, space=space)
    store = ActStore(a.data_dir, "train", device=a.data_device, max_pos=a.max_train_pos); store_val = ActStore(a.data_dir, "val", device=a.val_device)
    d = store.d
    # ---- text pools
    df = load_pool(a.text_parquet, os.path.join(a.data_dir, "pairs_train.parquet"), store.row_of)
    weights = {k: float(v) for k, v in (x.split(":") for x in a.pool_weights.split(",") if x)}
    sampler = PoolSampler(df, weights, a.seed)
    print(f"[train] {len(df)} train text rows over {df['pair_id'].nunique()} pairs", flush=True)
    # ---- val sets (pairs after the fixed eval set)
    import pyarrow.parquet as pq
    vp = pq.read_table(os.path.join(a.data_dir, "pairs_val.parquet")).to_pandas(); vp = vp[vp["pos_idx"].isin(store_val.row_of)].iloc[a.eval_offset:]
    sets = {}
    if a.val_text:
        vdf_all = load_pool(a.val_text, os.path.join(a.data_dir, "pairs_val.parquet"), store_val.row_of)
        for label in sorted(vdf_all["pool"].unique()):
            vdf = vdf_all[vdf_all["pool"] == label].sample(frac=1.0, random_state=0).drop_duplicates("pair_id").set_index("pair_id")
            sub = vp[vp["pair_id"].isin(vdf.index)].iloc[: a.eval_n]
            if len(sub) == 0: print(f"[train] val set {label}: no pairs with text after offset {a.eval_offset}", flush=True); continue
            sets[label] = (store_val.rows_for(sub["pos_idx"].values), torch.tensor(sub["i"].values.astype(np.int64)), torch.tensor(sub["j"].values.astype(np.int64)), vdf.loc[sub["pair_id"], "text"].tolist())
            print(f"[train] val set {label}: {len(sub)} pairs", flush=True)
    g_eval = torch.Generator().manual_seed(1234); eps_bank = [torch.randn(a.eval_n, d, generator=g_eval) for _ in T_GRID]
    # ---- optimiser
    ad_params = list(model.adapter_parameters()); lora_params = model.lora_parameters()
    opt = torch.optim.AdamW([{"params": ad_params, "lr": a.lr}, {"params": lora_params, "lr": a.lr_lora}], betas=(0.9, 0.95), weight_decay=a.wd)
    base_lrs = [a.lr, a.lr_lora]; step0 = 0
    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, map_location="cpu"); model.load_state(ck["state"]); step0 = ck["step"]
        if "opt" in ck: opt.load_state_dict(ck["opt"])
        print(f"[train] resumed {a.resume} @ {step0}", flush=True)
    def lr_mult(s):
        if s < a.warmup: return (s + 1) / a.warmup
        if a.lr_decay == "none": return 1.0
        pr = (s - a.warmup) / max(1, a.steps - a.warmup); return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, pr)))
    import wandb
    wandb.init(project=a.wandb, entity=a.wandb_entity, name=a.tag, config=vars(a) | {"space": space, "n_adapter": model.n_adapter_params(), "n_lora": sum(p_.numel() for p_ in lora_params), "n_text_rows": len(df)}, resume="allow")
    def save(step, name="ckpt_latest.pt", with_opt=True):
        ck = {"config": model.config(), "state": model.state(), "step": step, "args": vars(a), "prior_args": {k: v for k, v in paa.items() if k in ("src_rms", "squash", "target", "stats", "data_dir")}}
        if with_opt: ck["opt"] = opt.state_dict()
        torch.save(ck, os.path.join(a.out, name + ".tmp")); os.replace(os.path.join(a.out, name + ".tmp"), os.path.join(a.out, name))
    gen = torch.Generator().manual_seed(a.seed + step0); model.train(); t0 = time.time(); ema = None; best = None
    for step in range(step0, a.steps):
        sub = sampler.sample(a.batch)
        rows = store.rows_for(sub["pos_idx"].values); i = torch.tensor(sub["i"].values.astype(np.int64)); j = torch.tensor(sub["j"].values.astype(np.int64)); texts = sub["text"].tolist()
        h_i, x0, log_s, _ = make_x0(norm, store.gather(rows, i, dev), store.gather(rows, j, dev), space["target"], space["src_rms"], space["squash"])
        ids, mask = model.tokenize(texts); enc = TextIDs(ids)
        m = lr_mult(step)
        for g_, b_ in zip(opt.param_groups, base_lrs): g_["lr"] = b_ * m
        t_b = torch.rand(x0.shape[0], device=dev); eps_b = torch.randn_like(x0)
        loss_vec, t, kept = pair_fm_loss(model, x0, h_i, t_b, eps_b, enc=enc, enc_mask=mask, p_uncond=a.p_uncond, log_s=log_s)
        loss = loss_vec.mean(); null_loss = torch.zeros((), device=dev); con_loss = torch.zeros((), device=dev); con_acc = float("nan")
        if a.contrast > 0:
            perm = dm_perm(i, j).to(dev); mask_dm = mask[perm] & kept[:, None]
            loss_dm, _, _ = pair_fm_loss(model, x0, h_i, t_b, eps_b, enc=TextIDs(ids[perm]), enc_mask=mask_dm, log_s=log_s)
            gap = (loss_vec - loss_dm)[kept]
            if gap.numel():
                con_loss = torch.nn.functional.softplus((gap + a.contrast_margin) / a.contrast_tau).mean() * a.contrast_tau
                con_acc = float((gap < 0).float().mean()); loss = loss + a.contrast * con_loss
        if a.null_reg > 0:
            Bn = max(2, int(a.batch * a.null_frac)); sl = slice(0, Bn)
            eps_n = torch.randn_like(x0[sl]); t_n = torch.rand(Bn, device=dev); x_tn = (1 - t_n)[:, None] * x0[sl] + t_n[:, None] * eps_n
            rp = torch.roll(torch.arange(Bn, device=dev), Bn // 2)                                          # another pair's text (same batch)
            with torch.no_grad(): v_null = model(x_tn, t_n, h_i[sl], log_s=log_s[sl])
            v_rp = model(x_tn, t_n, h_i[sl], enc=TextIDs(ids[sl][rp]), enc_mask=mask[sl][rp], log_s=log_s[sl])
            null_loss = ((v_rp - v_null.detach()) ** 2).mean(); loss = loss + a.null_reg * null_loss
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(ad_params + lora_params, a.grad_clip); opt.step()
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % 25 == 0:
            log = {"train/loss": loss.item(), "train/loss_ema": ema, "train/null_loss": float(null_loss), "train/contrast_loss": float(con_loss), "train/contrast_acc": con_acc, "train/lr_mult": m, "train/grad_norm": float(gn),
                   "train/step_s": (time.time() - t0) / max(1, step - step0 + 1), "train/loss_cond": float(loss_vec[kept].mean()) if kept.any() else float("nan"), "train/loss_uncond": float(loss_vec[~kept].mean()) if (~kept).any() else float("nan"),
                   "train/n_text_tokens": float(mask.sum(1).float().mean())}
            wandb.log(log, step=step)
            if step % 50 == 0: print(f"[train] step {step} loss {loss.item():.4f} ema {ema:.4f} cond {log['train/loss_cond']:.4f} uncond {log['train/loss_uncond']:.4f} null {float(null_loss):.5f} con {float(con_loss):.4f} P(z>dm) {con_acc:.2f} gn {float(gn):.2f} {log['train/step_s']:.3f}s/step T {log['train/n_text_tokens']:.0f}", flush=True)
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            if sets:
                te = time.time(); out, br = evaluate(model, store_val, norm, space, sets, dev, eps_bank)
                wandb.log(out, step=step); json.dump({"step": step + 1, "scalars": out, "breakdown": br}, open(os.path.join(a.out, "eval_latest.json"), "w"), indent=1)
                print(f"[eval@{step+1}] ({time.time()-te:.0f}s) " + " | ".join(f"{k}: pmi {v['pmi_proxy_bits']:+.1f} dm {v['dm_proxy_bits']:+.1f} content {v['content_proxy_bits']:+.2f} P(z>dm) {v['p_z_beats_dm']:.2f} ws {v['by_band'].get('workspace', {}).get('content', float('nan')):+.2f}/{v['by_band'].get('workspace', {}).get('p_beats_dm', float('nan')):.2f}" for k, v in br.items()), flush=True)
                score = float(np.mean([v["content_proxy_bits"] for v in br.values()]))
                if best is None or score > best[0]:
                    best = (score, step + 1); save(step + 1, "ckpt_best.pt", with_opt=False); json.dump({"step": step + 1, "score": score, "metric": "mean content_proxy_bits"}, open(os.path.join(a.out, "best.json"), "w"))
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps: save(step + 1)
        if (time.time() - t_start) / 3600 > a.max_hours: print("[train] max hours reached", flush=True); save(step + 1); break
    save(a.steps, "ckpt_final.pt", with_opt=False); wandb.finish(); print("[train] DONE", flush=True)


if __name__ == "__main__":
    main()
