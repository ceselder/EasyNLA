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
    import glob as _glob, re as _re
    rows = []
    for pat in paths:
        for p in (sorted(_glob.glob(pat)) or [pat]):
            df = pd.read_json(p, lines=True) if p.endswith(".jsonl") else pq.read_table(p).to_pandas()
            if "verbosity" not in df:                          # lens files are named L<k>.parquet
                m = _re.search(r"[/_]L(\d)\.(parquet|jsonl)$", p); df["verbosity"] = int(m.group(1)) if m else 0
            if "source" not in df: df["source"] = os.path.basename(os.path.dirname(os.path.dirname(p))) or "text"
            rows.append(df)
    tx = pd.concat(rows, ignore_index=True)
    if verbosity is not None: tx = tx[tx["verbosity"].isin(verbosity)]
    tx = tx[tx["text"].astype(str).str.strip().str.len() > 0]
    df = tx.merge(pairs, on="pair_id", how="inner")
    return df


def smoke_texts(store, rows, i, j, tok):
    """PLUMBING TEST ONLY: text = the next token (a real fact about h_j). Never a research result."""
    nt = store.meta["next_token_id"].values[rows.cpu().numpy()]
    return [f"next token: {tok.decode([int(x)])!r}" for x in nt]


def synth_texts(mode, store, rows, i, j, tok=None, lf=None):
    """SYNTHETIC critic-diagnostic texts (DECISIONS v1.8 text-channel capacity tests; FORBIDDEN as verbalizer targets):
       depth   -> 'from layer {i} to layer {j}'          (T1: can the text channel read two integers? compare with the depth EMBEDDING critic)
       nexttok -> 'next token: <tok>'                     (plumbing)"""
    if mode == "depth": return [f"from layer {int(a)} to layer {int(b)}" for a, b in zip(i.tolist(), j.tolist())]
    if mode == "nexttok": return smoke_texts(store, rows, i, j, tok)
    if mode == "jlens20":    # T2-text: raw J-lens top-20 token lists at the source and the target (needs lf = LensFeats)
        return lf.texts(store.gather(rows, i), i, store.gather(rows, j), j)
    raise ValueError(mode)


@torch.no_grad()
def evaluate(model, store_val, norm, a, val_rows, val_i, val_j, val_text, encoder, dev, eps_bank, prefix="eval", lf=None):
    """fixed pairs, fixed eps (per t) -> loss tables. Returns a flat dict of scalars + a nested breakdown."""
    model.eval()
    B = 256; n = len(val_rows); d = norm.mean.numel()
    L_c = torch.zeros(len(T_GRID), n); L_u = torch.zeros(len(T_GRID), n); mse_id = torch.zeros(n); mse_x0 = torch.zeros(n); var_j = torch.zeros(n)
    for s in range(0, n, B):
        rows, i, j = val_rows[s:s + B], val_i[s:s + B], val_j[s:s + B]
        h_i, x0, log_s, _ = make_x0(norm, store_val.gather(rows, i, dev), store_val.gather(rows, j, dev), a.target, a.src_rms, a.squash)
        depth = torch.stack([i, j], 1).to(dev) if a.cond == "depth" else None
        enc = mask = None; vec = None
        if a.cond == "text":
            with torch.autocast("cuda", dtype=torch.bfloat16): enc, mask = encoder(val_text[s:s + B])
        if a.cond == "vec": vec = lf.vec_feats(store_val.gather(rows, i), i, store_val.gather(rows, j), j)
        if a.cond == "proj":
            from nlt.critic.model import oracle_projection
            vec = oracle_projection(model, x0, eps=eps_bank[0][s:s + B, : model.proj_k].to(dev))      # fixed noise per val row
        mse_id[s:s + B] = ((x0 - (0 if a.target == "delta" else h_i)) ** 2).mean(-1).cpu()       # identity transcoder h_j := h_i
        for ti, t in enumerate(T_GRID):
            tt = torch.full((len(rows),), t, device=dev); eps = eps_bank[ti][s:s + B].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lc, _, _ = pair_fm_loss(model, x0, h_i, tt, eps, depth=depth, enc=enc, enc_mask=mask, log_s=log_s, vec=vec)
            L_c[ti, s:s + B] = lc.cpu()
            if a.cond != "none":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lu, _, _ = pair_fm_loss(model, x0, h_i, tt, eps, depth=None, enc=None, enc_mask=None, log_s=log_s)
                L_u[ti, s:s + B] = lu.cpu()
            if t == 0.9:            # x0-prediction at high noise ~ conditional mean -> FVE-like number comparable to an MSE transcoder
                x_t = (1 - tt)[:, None] * x0 + tt[:, None] * eps
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    v = model(x_t, tt, h_i, depth=depth, depth_has=None if depth is None else torch.ones(len(rows), dtype=torch.bool, device=dev), enc=enc, enc_mask=mask, log_s=log_s, vec=vec, vec_has=None if vec is None else torch.ones(len(rows), dtype=torch.bool, device=dev))
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
    p.add_argument("--cond", default="none", choices=["none", "depth", "text", "vec", "proj"]); p.add_argument("--target", default="delta", choices=["hj", "delta"]); p.add_argument("--norm", default="affine", choices=["affine", "scalar"])
    p.add_argument("--squash", type=float, default=0.0, help="DECISIONS v1.9: radial squash of the target y = x / sqrt(c^2 + rms(x)^2) with c = this value (0 = off); analytic log-det added in the exact eval")
    p.add_argument("--src-rms", type=int, default=1, help="DECISIONS D2: divide h_i and the target by rms(h_i) after the pooled affine (1) or not (0, ablation)")
    p.add_argument("--d-model", type=int, default=2048); p.add_argument("--d-mlp", type=int, default=8192); p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-slots", type=int, default=8); p.add_argument("--n-heads", type=int, default=4); p.add_argument("--d-head", type=int, default=64); p.add_argument("--gate-rank", type=int, default=128)
    p.add_argument("--steps", type=int, default=5000); p.add_argument("--batch", type=int, default=512); p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--warmup", type=int, default=200); p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--lr-decay", default="cosine", choices=["none", "cosine"]); p.add_argument("--p-uncond", type=float, default=0.3); p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-train-pos", type=int, default=None); p.add_argument("--data-device", default="cuda", help="where the fp16 store lives (cuda on a B200; cpu on smaller GPUs)")
    p.add_argument("--eval-every", type=int, default=500); p.add_argument("--eval-n", type=int, default=4096); p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--eval-offset", type=int, default=4096, help="held-out pairs for in-training eval / model selection start HERE in pairs_val (default: after the fixed 4096-row eval set, so selection never sees it)")
    p.add_argument("--text-parquet", default=None, help="comma-separated text files/globs [pair_id, text, verbosity, source] for the TRAIN pairs (cond=text)"); p.add_argument("--text-verbosity", default=None, help="comma list of verbosity levels to train on (default all)")
    p.add_argument("--val-text-parquet", default=None, help="text files/globs for the VAL pairs (default: --text-parquet with '/train/' -> '/val/')")
    p.add_argument("--text-smoke", action="store_true", help="PLUMBING TEST: synthetic 'next token: X' text instead of --text-parquet")
    p.add_argument("--lens-dir", default="/vol/lens"); p.add_argument("--text-synth", default=None, choices=["depth", "nexttok", "jlens20"], help="synthetic diagnostic texts generated from (pair) metadata for every sampled pair (v1.8 T1); overrides --text-parquet")
    p.add_argument("--enc-model", default="Qwen/Qwen3-0.6B"); p.add_argument("--enc-layer", type=int, default=20); p.add_argument("--enc-max-len", type=int, default=128)
    p.add_argument("--wandb", default="nlt-qwen3-8b"); p.add_argument("--wandb-entity", default="octahedral-systems"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default=None); p.add_argument("--max-hours", type=float, default=20.0)
    p.add_argument("--extra-data-dirs", default=None, help="comma list of additional activation dirs (same layout): the train store becomes a CyclingStore over ALL dirs (blind/depth modes only)")
    p.add_argument("--cycle-resident", type=int, default=200_000); p.add_argument("--cycle-refresh", type=int, default=400, help="batches between shard swaps")
    p.add_argument("--cond-path", default="gate", choices=["gate", "block"], help="vec/proj conditioning pathway: gate modulation only (like the depth embedding) or per-block additive + gate (like the text cross-reads)")
    p.add_argument("--proj-k", type=int, default=32); p.add_argument("--proj-sigma", type=float, default=0.1); p.add_argument("--proj-mode", default="random", choices=["random", "pca"], help="T5 directions: random orthonormal or top-PCA of the flow target (from 16k sampled pairs)")
    p.add_argument("--contrast", type=float, default=0.0, help="DECISIONS v1.10 T4: weight of the contrastive hinge softplus((L(z) - L(z_dm) + margin)/tau) with z_dm = a depth-matched WRONG text (another row of the batch with the same j, same (i,j) when available), at the SAME (x_t, t, eps)")
    p.add_argument("--contrast-tau", type=float, default=0.005, help="logistic temperature in per-dim FM-loss units (0.005 ~ 10 nats)"); p.add_argument("--contrast-margin", type=float, default=0.005)
    p.add_argument("--null-reg", type=float, default=0.0, help="text mode: weight of the NULL regulariser ||v(x_t, z_rp) - v(x_t, no text)||^2 with z_rp = another pair's text of the batch (DECISIONS v1.5: pushes bits(random text) -> 0)")
    p.add_argument("--stats", default=None, help="stats.pt to normalise with (default <data-dir>/stats.pt). MUST be the prior's stats when --init-from is used on another store")
    p.add_argument("--init-from", default=None, help="checkpoint of a trained BLIND prior (cond none): its weights are loaded into this model (text/depth extras stay zero/fresh, so at step 0 the conditional path IS the prior)")
    p.add_argument("--freeze-prior", type=int, default=0, help="1 = train only the conditioning modules (cross-reads, gate_mod, depth embeddings); the unconditional path stays exactly the loaded prior")
    a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(a.out, exist_ok=True); t_start = time.time()
    norm = GlobalNorm.load(a.stats or os.path.join(a.data_dir, "stats.pt"), a.norm).to(dev)
    if a.extra_data_dirs:
        from nlt.data.dataset import CyclingStore
        assert a.cond != "text", "CyclingStore has no pair_id index; text mode needs a single ActStore"
        store = CyclingStore([a.data_dir] + a.extra_data_dirs.split(","), "train", device=a.data_device, resident=a.cycle_resident, refresh_every=a.cycle_refresh, seed=a.seed)
    else:
        store = ActStore(a.data_dir, "train", device=a.data_device, max_pos=a.max_train_pos)
    store_val = ActStore(a.data_dir, "val", device=a.data_device)
    d = store.d
    # ---- fixed val pairs (from the finalize pair list; disjoint docs) + fixed eps bank
    import pyarrow.parquet as pq
    vp = pq.read_table(os.path.join(a.data_dir, "pairs_val.parquet")).to_pandas()
    vp = vp[vp["pos_idx"].isin(store_val.row_of)].iloc[a.eval_offset: a.eval_offset + a.eval_n]
    val_rows = store_val.rows_for(vp["pos_idx"].values); val_i = torch.tensor(vp["i"].values); val_j = torch.tensor(vp["j"].values)
    g_eval = torch.Generator().manual_seed(1234); eps_bank = [torch.randn(len(val_rows), d, generator=g_eval) for _ in T_GRID]
    # the same fixed eval on TRAIN pairs (first rows of pairs_train.parquet that are in the store): the D3 gate compares eval/fm_loss with eval_train/fm_loss
    has_train_eval = hasattr(store, "row_of")
    if has_train_eval:
        tp = pq.read_table(os.path.join(a.data_dir, "pairs_train.parquet")).to_pandas(); tp = tp[tp["pos_idx"].isin(store.row_of)].iloc[: len(val_rows)]
        tr_rows = store.rows_for(tp["pos_idx"].values); tr_i = torch.tensor(tp["i"].values); tr_j = torch.tensor(tp["j"].values)
    else:   # cycling store: a fixed sample of its currently resident rows (train-side generalisation gate stays approximate)
        g_tr = torch.Generator().manual_seed(999); tr_rows, tr_i, tr_j = store.sample_pairs(len(val_rows), g_tr)
    g_eval2 = torch.Generator().manual_seed(4321); eps_bank_tr = [torch.randn(len(tr_rows), d, generator=g_eval2) for _ in T_GRID]
    # ---- text
    encoder = None; text_df = None; val_text = None; tok8 = None; lf = None
    if a.text_synth == "jlens20" or a.cond == "vec":
        from nlt.critic.lens_feats import LensFeats
        lf = LensFeats(a.lens_dir, dev, k=20)
    if a.cond == "text":
        from nlt.critic.text_encoder import TextEncoder
        encoder = TextEncoder(a.enc_model, a.enc_layer, dev, a.enc_max_len)
        if a.text_synth:
            from transformers import AutoTokenizer
            tok8 = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"); val_text = synth_texts(a.text_synth, store_val, val_rows, val_i, val_j, tok8, lf); a.text_smoke = True
            print(f"[train] SYNTHETIC TEXT MODE '{a.text_synth}' (critic diagnostic, forbidden for the verbalizer); example: {val_text[0]!r}", flush=True)
        elif a.text_smoke:
            from transformers import AutoTokenizer
            tok8 = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"); val_text = smoke_texts(store_val, val_rows, val_i, val_j, tok8)
            print("[train] TEXT SMOKE MODE: synthetic next-token texts (plumbing test, not a result)", flush=True)
        else:
            verb = [int(x) for x in a.text_verbosity.split(",")] if a.text_verbosity else None
            text_df = load_text_pairs(a.text_parquet.split(","), os.path.join(a.data_dir, "pairs_train.parquet"), verb)
            text_df = text_df[text_df["pos_idx"].isin(store.row_of)].reset_index(drop=True)
            val_files = a.val_text_parquet.split(",") if a.val_text_parquet else [x.replace("/train/", "/val/") for x in a.text_parquet.split(",")]
            vdf = load_text_pairs(val_files, os.path.join(a.data_dir, "pairs_val.parquet"), verb)
            vdf = vdf.sample(frac=1.0, random_state=0).drop_duplicates("pair_id").set_index("pair_id")      # one random verbosity per pair
            pid = [f"val:{p_}:{i_}:{j_}" for p_, i_, j_ in zip(vp["pos_idx"].values, vp["i"].values, vp["j"].values)]
            have = [x in vdf.index for x in pid]
            keep = np.where(have)[0]; assert len(keep) > 0, "no val pairs have text"
            val_rows, val_i, val_j = val_rows[keep], val_i[keep], val_j[keep]; eps_bank = [e[keep] for e in eps_bank]
            val_text = [vdf.loc[pid[k], "text"] for k in keep]
            print(f"[train] text pairs: train {len(text_df)} (verbosity {sorted(text_df['verbosity'].unique().tolist())}), val {len(keep)}/{len(pid)} with text", flush=True)
    model = PairDenoiser(d, a.d_model, a.d_mlp, a.n_layers, a.cond, d_enc=(encoder.d_enc if encoder else 0), n_slots=a.n_slots, n_heads=a.n_heads, d_head=a.d_head, gate_rank=a.gate_rank, target=a.target, proj_k=a.proj_k, proj_sigma=a.proj_sigma, cond_path=a.cond_path).to(dev)
    if a.cond == "proj":                       # T5 directions (fixed, saved in the checkpoint as a buffer)
        g_p = torch.Generator().manual_seed(777)
        if a.proj_mode == "random":
            P, _ = torch.linalg.qr(torch.randn(d, a.proj_k, generator=g_p)); P = P.T
        else:
            rows_p, i_p, j_p = store.sample_pairs(16384, g_p); _, x0_p, _, _ = make_x0(norm, store.gather(rows_p, i_p, dev), store.gather(rows_p, j_p, dev), a.target, a.src_rms, a.squash)
            _, _, Vh = torch.linalg.svd(x0_p.float() - x0_p.float().mean(0), full_matrices=False); P = Vh[: a.proj_k].cpu()
        model.proj_P.copy_(P.to(dev)); print(f"[train] T5 oracle projection: {a.proj_mode} k={a.proj_k} sigma={a.proj_sigma}", flush=True)
    if encoder: model.d_enc_ = encoder.d_enc
    model.src_rms_ = bool(a.src_rms); model.squash_ = float(a.squash)
    print(f"[train] {a.cond} critic: {model.n_params()/1e6:.0f}M params, target {a.target}, norm {a.norm}, src_rms {a.src_rms}, squash {a.squash}, batch {a.batch}, {a.steps} steps", flush=True)
    if a.init_from:
        ck = torch.load(a.init_from, map_location="cpu"); sd = ck["model"]
        if a.cond == "text" or (a.cond in ("vec", "proj") and a.cond_path == "block"):     # prior blocks live under blocks.<k>.base.* in wrapped models
            sd = {(k.replace("blocks.", "blocks.", 1) if not k.startswith("blocks.") else "blocks." + k.split(".", 1)[1].split(".", 1)[0] + ".base." + k.split(".", 2)[2]): v for k, v in sd.items()}
        res = model.load_state_dict(sd, strict=False)
        assert not res.unexpected_keys, res.unexpected_keys[:5]
        print(f"[train] init from {a.init_from} (step {ck.get('step')}): {len(sd)} tensors loaded, {len(res.missing_keys)} fresh (conditioning) tensors", flush=True)
    trainable = list(model.parameters())
    if a.freeze_prior:
        cond_names = {n for n, _ in model.named_parameters() if (".read." in n or ".gate_mod." in n or ".cvec_out." in n or n.startswith("emb_i") or n.startswith("emb_j") or n.startswith("tok_emb") or n.startswith("vec_in") or n.startswith("proj_in"))}
        for n, p_ in model.named_parameters(): p_.requires_grad_(n in cond_names)
        trainable = [p_ for n, p_ in model.named_parameters() if n in cond_names]
        print(f"[train] prior frozen: {sum(p_.numel() for p_ in trainable)/1e6:.1f}M trainable conditioning params", flush=True)
    opt = torch.optim.AdamW(trainable, lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd)
    step0 = 0
    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, map_location="cpu"); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step0 = ck["step"]; print(f"[train] resumed from {a.resume} @ {step0}", flush=True)
    def lr_at(s):
        if s < a.warmup: return a.lr * (s + 1) / a.warmup
        if a.lr_decay == "none": return a.lr
        pr = (s - a.warmup) / max(1, a.steps - a.warmup); return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, pr))))
    import wandb
    run = wandb.init(project=a.wandb, entity=a.wandb_entity, name=a.tag, config=vars(a) | {"n_params": model.n_params(), "n_train_pos": store.N, "n_val_pos": store_val.N, "n_shards": len(getattr(store, "files", []))}, resume="allow")
    gen = torch.Generator().manual_seed(a.seed + step0)
    def save(step, name="ckpt_latest.pt"):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "args": vars(a), "config": model.config(), "d_enc": (encoder.d_enc if encoder else 0)}, os.path.join(a.out, name))
    t0 = time.time(); ema = None; best = None
    for step in range(step0, a.steps):
        if a.cond == "text" and not a.text_smoke:
            idx = torch.randint(0, len(text_df), (a.batch,), generator=gen).numpy(); sub = text_df.iloc[idx]
            rows = store.rows_for(sub["pos_idx"].values); i = torch.tensor(sub["i"].values); j = torch.tensor(sub["j"].values); texts = sub["text"].tolist()
        else:
            rows, i, j = store.sample_pairs(a.batch, gen); texts = (synth_texts(a.text_synth, store, rows, i, j, tok8, lf) if a.text_synth else smoke_texts(store, rows, i, j, tok8)) if (a.cond == "text") else None
        h_i, x0, log_s, _ = make_x0(norm, store.gather(rows, i, dev), store.gather(rows, j, dev), a.target, a.src_rms, a.squash)
        depth = torch.stack([i, j], 1).to(dev) if a.cond == "depth" else None
        enc = mask = None; vec = None
        if a.cond == "text":
            with torch.autocast("cuda", dtype=torch.bfloat16): enc, mask = encoder(texts)
        if a.cond == "vec": vec = lf.vec_feats(store.gather(rows, i), i, store.gather(rows, j), j)
        if a.cond == "proj":
            from nlt.critic.model import oracle_projection
            vec = oracle_projection(model, x0)
        for g_ in opt.param_groups: g_["lr"] = lr_at(step)
        t_b = torch.rand(x0.shape[0], device=dev); eps_b = torch.randn_like(x0)                      # shared (t, eps) for the positive and the contrastive negative
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss_vec, t, kept = pair_fm_loss(model, x0, h_i, t_b, eps_b, depth=depth, enc=enc, enc_mask=mask, p_uncond=(a.p_uncond if a.cond != "none" else 0.0), log_s=log_s, vec=vec)
        loss = loss_vec.mean(); null_loss = torch.zeros((), device=dev); con_loss = torch.zeros((), device=dev); con_acc = float("nan")
        if a.contrast > 0 and a.cond == "text":
            # T4: depth-matched wrong text = another row's text with the same j (same (i, j) when the batch has one); hinge on the per-row FM loss at the same (x_t, t, eps)
            jj = j.tolist(); ii = i.tolist(); perm = list(range(len(jj)))
            by_ij = {}; by_j = {}
            for q, (aa_, bb_) in enumerate(zip(ii, jj)): by_ij.setdefault((aa_, bb_), []).append(q); by_j.setdefault(bb_, []).append(q)
            for q in range(len(jj)):
                c = [r_ for r_ in by_ij[(ii[q], jj[q])] if r_ != q] or [r_ for r_ in by_j[jj[q]] if r_ != q]
                perm[q] = c[q % len(c)] if c else (q + len(jj) // 2) % len(jj)
            perm_t = torch.tensor(perm, device=dev); enc_dm = enc[perm_t]; mask_dm = mask[perm_t] & kept[:, None]                   # dropped rows stay dropped on both sides
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss_dm, _, _ = pair_fm_loss(model, x0, h_i, t_b, eps_b, enc=enc_dm, enc_mask=mask_dm, log_s=log_s)
            gap = (loss_vec - loss_dm)[kept]                                                                                           # < 0 = the true text wins
            if gap.numel():
                con_loss = torch.nn.functional.softplus((gap + a.contrast_margin) / a.contrast_tau).mean() * a.contrast_tau           # tau-scaled so the gradient is O(1) per row
                con_acc = float((gap < 0).float().mean()); loss = loss + a.contrast * con_loss
        if a.null_reg > 0 and a.cond == "text":
            # NULL regulariser: under ANOTHER pair's text (batch rolled by B/2) the velocity must equal the no-text velocity (the frozen prior)
            Bn = x0.shape[0]; eps_n = torch.randn_like(x0); t_n = torch.rand(Bn, device=dev); x_tn = (1 - t_n)[:, None] * x0 + t_n[:, None] * eps_n
            enc_rp = torch.roll(enc, Bn // 2, 0); mask_rp = torch.roll(mask, Bn // 2, 0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad(): v_null = model(x_tn, t_n, h_i, enc=enc_rp, enc_mask=torch.zeros_like(mask_rp), log_s=log_s)
                v_rp = model(x_tn, t_n, h_i, enc=enc_rp, enc_mask=mask_rp, log_s=log_s)
            null_loss = ((v_rp - v_null.detach()) ** 2).mean(); loss = loss + a.null_reg * null_loss
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(trainable, a.grad_clip); opt.step()
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % 25 == 0:
            log = {"train/loss": loss.item(), "train/loss_ema": ema, "train/null_loss": float(null_loss), "train/contrast_loss": float(con_loss), "train/contrast_acc": con_acc, "train/lr": lr_at(step), "train/grad_norm": float(gn), "train/step_s": (time.time() - t0) / max(1, step - step0 + 1)}
            if a.cond != "none":
                log["train/loss_cond"] = float(loss_vec[kept].mean()) if kept.any() else float("nan"); log["train/loss_uncond"] = float(loss_vec[~kept].mean()) if (~kept).any() else float("nan")
            wandb.log(log, step=step)
            if step % 100 == 0: print(f"[train] step {step} loss {loss.item():.4f} ema {ema:.4f} lr {lr_at(step):.2e} gn {float(gn):.2f} {log['train/step_s']:.3f}s/step" + (f" cond {log['train/loss_cond']:.4f} uncond {log['train/loss_uncond']:.4f} null {float(null_loss):.4f} con {float(con_loss):.4f} P(z>dm) {con_acc:.2f}" if a.cond != "none" else ""), flush=True)
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            out, br = evaluate(model, store_val, norm, a, val_rows, val_i, val_j, val_text, encoder, dev, eps_bank, lf=lf)
            if a.cond != "text" and (has_train_eval or True):       # train-pair eval (same grid, fixed eps) for the generalisation gate; text mode has no per-pair train texts here
                out_tr, br_tr = evaluate(model, store, norm, a, tr_rows, tr_i, tr_j, None, None, dev, eps_bank_tr, prefix="eval_train", lf=lf)
                out.update({k: v for k, v in out_tr.items() if "_gap/" not in k}); out["gate/heldout_over_train_fm"] = out["eval/fm_loss"] / max(1e-9, out_tr["eval_train/fm_loss"])
                br["train_by_gap"] = br_tr["by_gap"]
            wandb.log(out, step=step); json.dump({"step": step + 1, "scalars": out, "breakdown": br}, open(os.path.join(a.out, "eval_latest.json"), "w"), indent=1)
            score = out.get("eval/pmi_proxy_bits", -out["eval/fm_loss"])
            if best is None or score > best[0]:
                best = (score, step + 1); save(step + 1, "ckpt_best.pt"); json.dump({"step": step + 1, "score": score, "metric": "eval/pmi_proxy_bits" if "eval/pmi_proxy_bits" in out else "-eval/fm_loss"}, open(os.path.join(a.out, "best.json"), "w"))
                print(f"[train] new best ({best[1]}): {score:.3f} -> ckpt_best.pt", flush=True)
            print(f"[eval@{step+1}] " + " ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in out.items() if "/" in k and "_gap/" not in k), flush=True)
            print("[eval] by gap: " + json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in br["by_gap"].items()}), flush=True)
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps: save(step + 1)
        if (time.time() - t_start) / 3600 > a.max_hours: print("[train] max hours reached", flush=True); save(step + 1); break
    save(a.steps, "ckpt_final.pt"); wandb.finish()
    print("[train] DONE", flush=True)


if __name__ == "__main__":
    main()
