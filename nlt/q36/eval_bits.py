"""Held-out exact bits for the Qwen3.6-27B direction critic (port of nlt/eval_bits/run.py + score_manifest.py to critic_data.py conventions).

  python eval_bits.py --data-dir /vol/q36/data --ckpt /vol/q36/critic/<tag>/ckpt_final.pt --out /vol/q36/results/bits_<tag>.json \
      --sets "craft:/vol/q36/text/craft_v1/val/*.parquet,jlens:...,olens_delta:...,verbalizer:/vol/q36/dumps/<tag>.parquet" --n 512 --ode-steps 64 \
      [--twins "craft_twin:/vol/q36/text/craft_v1/val_twins/*.parquet"] [--n-fixed 2048]

Fixed test set = the first --n-fixed pairs of pairs_val (in file order) that are in the store; every set is scored on its own rows of it, common rows first.
Per set and row (same y = sqrt(d) u_j s with a FIXED s per row, same eps / Hutchinson probes for every variant -> paired):
  exact log p under: no text (u), the true text (c), a depth-matched wrong text (dm: another pair, same (i, j) else same j), a random pair's text (rp),
  the true text with its words shuffled (sw). PMI = c - u in bits; content = (c - u) - (dm - u) = c - dm; P(z > dm), P(z > rp), P(z > sw).
  Conditional-mean cos: centred cos between x0-hat(from pure noise) and u_j, with text / no text / dm text; plus 4 ODE samples per row (Heun) -> cos of
  each sample and of their mean with u_j.
Twins (--twins label:glob): rows [pair_id, variant, text] with variant 'true' and any other names; exact log p per row (paired) -> P(true > variant) per variant.
"""
from __future__ import annotations
import argparse, json, math, os, time
os.environ["HF_HUB_OFFLINE"] = "0"; os.environ["HF_HOME"] = "/vol/hf_cache"          # the nlt volume (rw): Qwen3-0.6B text encoder downloads once; the 27B is not needed here
import numpy as np, torch
import pyarrow.parquet as pq
from critic_data import Store, Directions, load_text_pairs, dm_partner

T_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)


def load_critic(path, dev):
    from nlt.prior.model import build_prior
    ck = torch.load(path, map_location="cpu"); m = build_prior(ck["config"]); m.load_state_dict(ck["model"]); m.to(dev).eval().requires_grad_(False)
    return m, ck["args"], ck.get("step")


@torch.no_grad()
def ode_sample(model, src, enc, mask, n_steps, eps):
    """Heun from noise (t=1) to data (t=0) with the velocity v = eps - x0 convention: x_{t-h} = x_t - h v"""
    x = eps.clone(); B = x.shape[0]; ts = torch.linspace(1, 0, n_steps + 1, device=x.device)
    for k in range(n_steps):
        t0, t1 = ts[k], ts[k + 1]; h = t0 - t1
        with torch.autocast("cuda", dtype=torch.bfloat16): v0 = model(x, torch.full((B,), float(t0), device=x.device), src, enc=enc, enc_mask=mask).float()
        xp = x - h * v0
        with torch.autocast("cuda", dtype=torch.bfloat16): v1 = model(xp, torch.full((B,), float(t1), device=x.device), src, enc=enc, enc_mask=mask).float()
        x = x - h * 0.5 * (v0 + v1)
    return x


def summarize(v, i, j):
    v = np.asarray(v, np.float64); i = np.asarray(i); j = np.asarray(j); gaps = j - i
    out = {"mean": float(v.mean()), "median": float(np.median(v)), "sem": float(v.std() / math.sqrt(len(v))), "n": int(len(v)), "by_gap": {}, "by_j": {}, "by_i": {}}
    for lo, hi in ((1, 8), (9, 16), (17, 24), (25, 48)):
        m = (gaps >= lo) & (gaps <= hi)
        if m.sum(): out["by_gap"][f"{lo}-{hi}"] = {"mean": float(v[m].mean()), "sem": float(v[m].std() / math.sqrt(m.sum())), "n": int(m.sum())}
    for jj in np.unique(j): m = j == jj; out["by_j"][str(int(jj))] = {"mean": float(v[m].mean()), "sem": float(v[m].std() / math.sqrt(m.sum())), "n": int(m.sum())}
    for ii in np.unique(i): m = i == ii; out["by_i"][str(int(ii))] = {"mean": float(v[m].mean()), "sem": float(v[m].std() / math.sqrt(m.sum())), "n": int(m.sum())}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--ckpt", required=True); p.add_argument("--out", required=True); p.add_argument("--stats", default=None)
    p.add_argument("--sets", default="", help="label:glob[;glob],... text sets (val split)"); p.add_argument("--twins", default="", help="label:glob,... twin manifests [pair_id, variant, text]")
    p.add_argument("--n", type=int, default=512); p.add_argument("--n-fixed", type=int, default=2048); p.add_argument("--batch", type=int, default=32); p.add_argument("--ode-steps", type=int, default=64); p.add_argument("--probes", type=int, default=1)
    p.add_argument("--n-samples", type=int, default=4); p.add_argument("--sample-steps", type=int, default=32); p.add_argument("--skip-samples", action="store_true"); p.add_argument("--skip-sw", action="store_true")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--data-device", default="cuda")
    p.add_argument("--neighbors", default=None, help="dir of extract_neighbors.py outputs: score each text against the SAME document's (h_i, h_j) at positions t-k (k in the file) -> content(own) - content(neighbour)")
    p.add_argument("--neighbor-n", type=int, default=256, help="rows per set for the neighbour control (exact ODE passes are 2 per offset)")
    a = p.parse_args(); dev = "cuda"; torch.manual_seed(a.seed); t_all = time.time()
    from nlt.eval_bits.exact import exact_logp, make_probe_bank
    from nlt.critic.text_encoder import TextEncoder
    model, aa, step = load_critic(a.ckpt, dev); sigma_r = float(aa.get("sigma_r", 0.1))
    dirs = Directions(a.stats or aa.get("stats_path") or os.path.join(a.data_dir, "layer_stats.pt"), sigma_r, dev, radial=aa.get("radial", "lognormal"), sigma_iso=float(aa.get("sigma_iso", 0.0)))
    store = Store(a.data_dir, "val", device=a.data_device); d = store.d
    NB = None                                                     # neighbour activations: {(shard, row) -> index}, tensors per (L, k)
    if a.neighbors:
        import glob as _g, json as _json
        spl = _json.load(open(os.path.join(a.data_dir, "splits.json")))["val"]; NB = {"idx": {}, "H": {}, "has": {}}
        for si, f in enumerate(spl):
            nf = os.path.join(a.neighbors, os.path.basename(f))
            if not os.path.exists(nf): continue
            tb = pq.read_table(nf); names = tb.schema.names; rows_ = tb.column("row").to_numpy(); offs = sorted(int(c[5:]) for c in names if c.startswith("has_m"))
            base = len(NB["idx"])
            for q, r in enumerate(rows_): NB["idx"][(si, int(r))] = base + q
            for k in offs:
                NB["has"].setdefault(k, []).append(torch.tensor(tb.column(f"has_m{k}").to_numpy()))
                for L in store.layers:
                    c = f"h_L{L}_m{k}"
                    if c in names: NB["H"].setdefault((L, k), []).append(torch.tensor(tb.column(c).combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(tb.num_rows, d).astype(np.float16)))
        NB["offs"] = sorted({k for (_, k) in NB["H"]}); NB["H"] = {key: torch.cat(v) for key, v in NB["H"].items()}; NB["has"] = {k: torch.cat(v) for k, v in NB["has"].items()}
        print(f"[bits] neighbours: {len(NB['idx'])} positions, offsets {NB['offs']}", flush=True)
    def nb_gather(kk, layer_vec, k):
        """neighbour activations at t-k for the fixed-set rows kk (list) at per-row layers -> [B, d] fp16 (cpu); rows without a neighbour get zeros + mask False"""
        out = torch.zeros((len(kk), d), dtype=torch.float16); ok = torch.zeros(len(kk), dtype=torch.bool)
        for q, k_ in enumerate(kk):
            key = (int(vp["shard"].iloc[k_]) if "shard" in vp else int(vp["pos_idx"].iloc[k_]) // 1_000_000, int(vp["pos_idx"].iloc[k_]) % 1_000_000)
            n_ = NB["idx"].get(key)
            if n_ is None or not bool(NB["has"][k][n_]): continue
            out[q] = NB["H"][(int(layer_vec[q]), k)][n_]; ok[q] = True
        return out, ok
    encoder = TextEncoder(aa.get("enc_model", "Qwen/Qwen3-0.6B"), int(aa.get("enc_layer", 20)), dev, int(aa.get("enc_max_len", 192)))
    vp = pq.read_table(os.path.join(a.data_dir, "pairs_val.parquet"), columns=["pair_id", "pos_idx", "i", "j", "shard"]).to_pandas(); vp = vp[vp["pos_idx"].isin(store.row_of)].iloc[: a.n_fixed].reset_index(drop=True); NF = len(vp)
    rows_all = store.rows_for(vp["pos_idx"].values); I_all = torch.tensor(vp["i"].values.astype(np.int64)); J_all = torch.tensor(vp["j"].values.astype(np.int64)); pid_all = vp["pair_id"].tolist()
    g = torch.Generator().manual_seed(a.seed + 1); s_all = torch.exp(sigma_r * torch.randn(NF, generator=g)); eps_all = torch.randn(NF, d, generator=g); eps_samp = torch.randn(a.n_samples, NF, d, generator=g); iso_all = torch.randn(NF, d, generator=g)
    probe_bank = make_probe_bank(a.ode_steps, a.probes, d, torch.Generator().manual_seed(a.seed + 2)); rng_sw = np.random.default_rng(a.seed + 7)
    sets = {}
    for item in [s for s in a.sets.split(",") if s.strip()]:
        label, path = item.split(":", 1); df = load_text_pairs(path.split(";"), os.path.join(a.data_dir, "pairs_val.parquet")).drop_duplicates("pair_id").set_index("pair_id"); sets[label] = df["text"].astype(str).to_dict()
        print(f"[bits] set {label}: {len(df)} pairs with text", flush=True)
    common = [k for k, pid in enumerate(pid_all) if all(pid in tm for tm in sets.values())] if sets else list(range(NF))
    print(f"[bits] critic {a.ckpt} (step {step}, {model.n_params()/1e6:.0f}M) | fixed set {NF} pairs, {len(common)} common to all {len(sets)} sets", flush=True)
    def set_indices(tm):
        own = [k for k, pid in enumerate(pid_all) if pid in tm]; cs = set(common); return (common[: a.n] + [k for k in own if k not in cs])[: a.n]
    UNC = {}          # k -> (lp_u, cos_u, cos_samples_u...)
    def inputs(kk):
        r = rows_all[kk]; i = I_all[kk]; j = J_all[kk]; h_i = store.gather(r, i, dev); h_j = store.gather(r, j, dev); src = dirs.source(h_i, i); x0, _ = dirs.target(h_j, j, s=s_all[kk].to(dev), eps_iso=iso_all[kk].to(dev))
        return x0, src, h_j, j
    def lp_only(kk, texts):
        x0, src, _, _ = inputs(kk)
        with torch.autocast("cuda", dtype=torch.bfloat16): enc, mask = encoder(texts)
        return exact_logp(model, x0, src, enc=enc, enc_mask=mask, n_steps=a.ode_steps, probes=a.probes, probe_bank=probe_bank).cpu()
    def lp_batch(kk, texts):
        x0, src, h_j, j = inputs(kk)
        enc = mask = None
        if texts is not None:
            with torch.autocast("cuda", dtype=torch.bfloat16): enc, mask = encoder(texts)
        lp = exact_logp(model, x0, src, enc=enc, enc_mask=mask, n_steps=a.ode_steps, probes=a.probes, probe_bank=probe_bank).cpu()
        e1 = eps_all[kk].to(dev); tt = torch.ones(len(kk), device=dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): v = model(e1, tt, src, enc=enc, enc_mask=mask)
        cos_mean = dirs.cos_to_target(e1 - v.float(), h_j, j).cpu()
        cos_s = cos_sm = None
        if not a.skip_samples:
            S = torch.stack([ode_sample(model, src, enc, mask, a.sample_steps, eps_samp[q][kk].to(dev)) for q in range(a.n_samples)])          # [n_s, B, d]
            cos_s = torch.stack([dirs.cos_to_target(S[q], h_j, j) for q in range(a.n_samples)]).mean(0).cpu(); cos_sm = dirs.cos_to_target(S.mean(0), h_j, j).cpu()
        return lp, cos_mean, cos_s, cos_sm
    results = {"ckpt": a.ckpt, "step": step, "n_fixed": NF, "n_common": len(common), "ode_steps": a.ode_steps, "sample_steps": a.sample_steps, "n_samples": a.n_samples, "sigma_r": sigma_r, "sets": {}, "twins": {}}
    for label, tm in sets.items():
        idx = set_indices(tm); n = len(idx); texts = [tm[pid_all[k]] for k in idx]; dmp = dm_partner(I_all[idx], J_all[idx]); dm_texts = [texts[q] for q in dmp]; rp_texts = [texts[(q + n // 2) % n] for q in range(n)]
        sw_texts = [" ".join(np.array(z.split())[rng_sw.permutation(len(z.split()))].tolist()) if len(z.split()) > 1 else z for z in texts]
        LP = {k: torch.zeros(n) for k in ("u", "c", "dm", "rp", "sw")}; COS = {k: torch.zeros(n) for k in ("u", "c", "dm")}; CS = {k: torch.zeros(n) for k in ("u", "c", "dm")}; CSM = {k: torch.zeros(n) for k in ("u", "c", "dm")}; t0 = time.time()
        for s0 in range(0, n, a.batch):
            kk = idx[s0:s0 + a.batch]; B = len(kk)
            need = [q for q, k in enumerate(kk) if k not in UNC]
            if need:
                lp, cm, cs, csm = lp_batch([kk[q] for q in need], None)
                for q_, q in enumerate(need): UNC[kk[q]] = (float(lp[q_]), float(cm[q_]), float(cs[q_]) if cs is not None else float("nan"), float(csm[q_]) if csm is not None else float("nan"))
            for q, k in enumerate(kk): LP["u"][s0 + q], COS["u"][s0 + q], CS["u"][s0 + q], CSM["u"][s0 + q] = UNC[k]
            for key, tx in (("c", texts), ("dm", dm_texts), ("rp", rp_texts), ("sw", sw_texts)):
                if key == "sw" and a.skip_sw: continue
                if key in ("rp", "sw"):
                    LP[key][s0:s0 + B] = lp_only(kk, tx[s0:s0 + B]); continue
                lp, cm, cs, csm = lp_batch(kk, tx[s0:s0 + B]); LP[key][s0:s0 + B] = lp; COS[key][s0:s0 + B] = cm
                if cs is not None: CS[key][s0:s0 + B] = cs; CSM[key][s0:s0 + B] = csm
            print(f"[bits] {label}: {min(n, s0 + a.batch)}/{n} rows, {time.time() - t0:.0f}s", flush=True)
        NBRES = {}
        if NB is not None:
            nn = min(a.neighbor_n, n); sub = idx[:nn]; sub_texts = texts[:nn]; sub_dm = dm_texts[:nn]
            for k in NB["offs"]:
                lp_u = torch.zeros(nn); lp_c = torch.zeros(nn); lp_d = torch.zeros(nn); okall = torch.zeros(nn, dtype=torch.bool)
                for s0 in range(0, nn, a.batch):
                    kk = sub[s0:s0 + a.batch]; B = len(kk); i = I_all[kk]; j = J_all[kk]
                    hi_nb, ok_i = nb_gather(kk, i, k); hj_nb, ok_j = nb_gather(kk, j, k); ok = ok_i & ok_j; okall[s0:s0 + B] = ok
                    if not ok.any(): continue
                    src_nb = dirs.source(hi_nb.to(dev), i); y_nb, _ = dirs.target(hj_nb.to(dev), j, s=s_all[kk].to(dev), eps_iso=iso_all[kk].to(dev))
                    lp_u[s0:s0 + B] = exact_logp(model, y_nb, src_nb, n_steps=a.ode_steps, probes=a.probes, probe_bank=probe_bank).cpu()
                    with torch.autocast("cuda", dtype=torch.bfloat16): e_c, m_c = encoder(sub_texts[s0:s0 + B]); e_d, m_d = encoder(sub_dm[s0:s0 + B])
                    lp_c[s0:s0 + B] = exact_logp(model, y_nb, src_nb, enc=e_c, enc_mask=m_c, n_steps=a.ode_steps, probes=a.probes, probe_bank=probe_bank).cpu()
                    lp_d[s0:s0 + B] = exact_logp(model, y_nb, src_nb, enc=e_d, enc_mask=m_d, n_steps=a.ode_steps, probes=a.probes, probe_bank=probe_bank).cpu()
                okn = okall.numpy(); cont_nb = ((lp_c - lp_d) / math.log(2)).numpy(); cont_own = ((LP["c"][:nn] - LP["dm"][:nn]) / math.log(2)).numpy(); pmi_nb = ((lp_c - lp_u) / math.log(2)).numpy()
                if okn.sum() >= 8:
                    dd = cont_own[okn] - cont_nb[okn]
                    NBRES[f"m{k}"] = {"n": int(okn.sum()), "content_neighbour": float(cont_nb[okn].mean()), "content_own_same_rows": float(cont_own[okn].mean()), "double_diff": float(dd.mean()), "double_diff_sem": float(dd.std() / math.sqrt(okn.sum())),
                                      "frac_content_kept_at_neighbour": float(cont_nb[okn].mean() / cont_own[okn].mean()) if abs(cont_own[okn].mean()) > 1e-6 else None, "p_own_gt_neighbour": float((dd > 0).mean()), "pmi_neighbour": float(pmi_nb[okn].mean())}
                    print(f"[bits] {label} neighbour -{k}: content own {cont_own[okn].mean():.1f} vs neighbour {cont_nb[okn].mean():.1f} bits (double diff {dd.mean():.1f} +- {dd.std() / math.sqrt(okn.sum()):.1f}, kept {NBRES[f'm{k}']['frac_content_kept_at_neighbour']}), n {okn.sum()}", flush=True)
        i_np = I_all[idx].numpy(); j_np = J_all[idx].numpy(); b = {k: ((LP[k] - LP["u"]) / math.log(2)).numpy() for k in ("c", "dm", "rp", "sw")}
        ntok = float(np.mean([len(encoder.tok(z, add_special_tokens=False)["input_ids"]) for z in texts]))
        res = {"n": n, "n_common": sum(1 for k in idx if k in set(common)), "pair_ids": [pid_all[k] for k in idx], "n_tokens_mean": ntok,
               "pmi_bits": summarize(b["c"], i_np, j_np), "dm_bits": summarize(b["dm"], i_np, j_np), "rp_bits": summarize(b["rp"], i_np, j_np), "sw_bits": summarize(b["sw"], i_np, j_np) if not a.skip_sw else None,
               "content_bits": summarize(b["c"] - b["dm"], i_np, j_np), "content_rp_bits": summarize(b["c"] - b["rp"], i_np, j_np),
               "p_z_gt_dm": float((b["c"] > b["dm"]).mean()), "p_z_gt_rp": float((b["c"] > b["rp"]).mean()), "p_z_gt_sw": float((b["c"] > b["sw"]).mean()) if not a.skip_sw else None, "p_z_gt_null": float((b["c"] > 0).mean()),
               "content_per_token": float((b["c"] - b["dm"]).mean() / max(1e-9, ntok)), "bits_per_token": float(b["c"].mean() / max(1e-9, ntok)), "uncond_nll_bits_per_dim": float(-LP["u"].mean() / (d * math.log(2))),
               "neighbours": NBRES, "cos_condmean": {k: float(COS[k].mean()) for k in COS}, "cos_samples": {k: float(CS[k].mean()) for k in CS}, "cos_sample_mean": {k: float(CSM[k].mean()) for k in CSM}, "p_cos_c_gt_u": float((COS["c"] > COS["u"]).mean()),
               "per_row": {"pmi": b["c"].round(3).tolist(), "dm": b["dm"].round(3).tolist(), "rp": b["rp"].round(3).tolist(), "cos_c": COS["c"].numpy().round(4).tolist(), "cos_u": COS["u"].numpy().round(4).tolist()}}
        results["sets"][label] = res
        print(f"[bits] {label}: PMI {res['pmi_bits']['mean']:.1f} | dm {res['dm_bits']['mean']:.1f} | rp {res['rp_bits']['mean']:.1f} | CONTENT {res['content_bits']['mean']:.1f} +- {res['content_bits']['sem']:.1f} | P(z>dm) {res['p_z_gt_dm']:.3f} P(z>rp) {res['p_z_gt_rp']:.3f} | cos condmean u {res['cos_condmean']['u']:.3f} c {res['cos_condmean']['c']:.3f} dm {res['cos_condmean']['dm']:.3f} | cos samples u {res['cos_samples']['u']:.3f} c {res['cos_samples']['c']:.3f} | {ntok:.0f} tok", flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(results, open(a.out, "w"), indent=1)
    # ---- twins: P(true > variant), paired ----
    for item in [s for s in a.twins.split(",") if s.strip()]:
        label, path = item.split(":", 1); import glob as _g, pandas as pd
        tw = pd.concat([pq.read_table(f).to_pandas() for f in sorted(_g.glob(path))], ignore_index=True); tw = tw[tw["pair_id"].isin(set(pid_all))]
        by_pid = {pid: k for k, pid in enumerate(pid_all)}; tw["k"] = tw["pair_id"].map(by_pid); tw = tw.sort_values(["k", "variant"]).reset_index(drop=True)
        pids = [pid for pid in tw["pair_id"].drop_duplicates().tolist()][: a.n]; tw = tw[tw["pair_id"].isin(set(pids))].reset_index(drop=True); n = len(tw); lp = np.zeros(n); t0 = time.time()
        for s0 in range(0, n, a.batch):
            sub = tw.iloc[s0:s0 + a.batch]; kk = sub["k"].tolist()
            lp[s0:s0 + len(kk)] = lp_batch(kk, sub["text"].astype(str).tolist())[0].numpy()
            if (s0 // a.batch) % 10 == 0: print(f"[twins] {label}: {min(n, s0 + a.batch)}/{n} rows, {time.time() - t0:.0f}s", flush=True)
        tw["logp"] = lp; res = {"n_pairs": len(pids), "variants": {}}
        tru = tw[tw["variant"] == "true"].set_index("pair_id")["logp"]
        for var in sorted(set(tw["variant"]) - {"true"}):
            sub = tw[tw["variant"] == var].set_index("pair_id"); com = [pid for pid in sub.index if pid in tru.index]
            dlt = (tru.loc[com].values - sub.loc[com, "logp"].values) / math.log(2)
            res["variants"][var] = {"n": len(com), "p_true_gt_twin": float((dlt > 0).mean()), "mean_bits_true_minus_twin": float(dlt.mean()), "sem": float(dlt.std() / math.sqrt(max(1, len(dlt))))}
            print(f"[twins] {label}/{var}: P(true > twin) {res['variants'][var]['p_true_gt_twin']:.3f} | true - twin {dlt.mean():.2f} +- {res['variants'][var]['sem']:.2f} bits (n {len(com)})", flush=True)
        results["twins"][label] = res; json.dump(results, open(a.out, "w"), indent=1)
    results["elapsed_min"] = (time.time() - t_all) / 60; json.dump(results, open(a.out, "w"), indent=1)
    print("[bits] DONE ->", a.out, flush=True)


if __name__ == "__main__":
    main()
