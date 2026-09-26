"""Population alignment of a text-conditioned flow critic, with no steering and no LM outputs: does the critic's condition effect for a label-shared
claim point along the POPULATION shift of the activations that claim is true of?

Data: held-out (is_val) anchors of the single-claim data (/vol_glp/claims/final/final_*.parquet); each anchor's internal next_token claim names the
model's top-1 next token X. Words with >= --min-n held-out anchors form the vocabulary W; mu_X = mean normalized layer-42 activation of the anchors
whose next token is X.
Per critic, for the canonical claim c_X = "The model expects the next word to be 'X'." (bulleted "• ..." for claim-trained critics, as in training):
  cm_align      mean over word pairs of cos(m(c_Y) - m(c_X), mu_Y - mu_X), m(c) = E_eps[eps - v(eps, t=1, c)] (the h-independent conditional mean)
  cm_ratio      median |m(c_Y) - m(c_X)| / |mu_Y - mu_X|
  cm_retrieval  zero-shot next-token retrieval: argmax_W cos(m(c_W) - mbar, h - hbar) == X on held-out anchors (chance 1/|W|)
  dlt_align_t   cos(x0hat(h,t,c_Y) - x0hat(h,t,c_X), mu_Y - mu_X) at held-out anchors h of word X, random Y != X (the delta edit's direction)
usage: python scripts/pop_align.py --critics name=adapter_path,... --tag T   -> /vol_glp/cond/popalign/<tag>.json"""
import argparse, glob, json, os, random, re, sys, time
import numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
DEV = "cuda"; OUT = "/vol_glp/cond/popalign"; TEMPLATE = "The model expects the next word to be '{w}'."


def load_heldout(glob_pat, max_files, min_n, max_per_word, seed=0):
    import pyarrow.parquet as pq
    files = sorted(glob.glob(glob_pat))[:max_files]; groups = {}
    for f in files:
        pf = pq.ParquetFile(f)
        for rb in pf.iter_batches(batch_size=4096, columns=["activation_vector", "claims", "families", "types", "is_val"]):
            isv = rb.column("is_val").to_pylist(); want = [i for i, v in enumerate(isv) if v]
            if not want: continue
            cl = rb.column("claims").to_pylist(); fa = rb.column("families").to_pylist(); ty = rb.column("types").to_pylist(); av = None
            for i in want:
                for c, g, t in zip(cl[i], fa[i], ty[i]):
                    if g != "internal" or not (t or "").startswith("next_token"): continue
                    m = re.search(r"'([^']+)'", c)
                    if not m or not re.fullmatch(r"[A-Za-z][A-Za-z\-]*", m.group(1)): continue
                    w = m.group(1)
                    if len(groups.setdefault(w, [])) >= max_per_word: break
                    if av is None:
                        a_ = rb.column("activation_vector"); av = a_.flatten().to_numpy(zero_copy_only=False).reshape(-1, a_.type.list_size)
                    groups[w].append(av[i].astype(np.float32)); break
    groups = {w: np.stack(v) for w, v in groups.items() if len(v) >= min_n}
    print(f"[data] {len(files)} files -> {len(groups)} next-token words with >= {min_n} held-out anchors: " + ", ".join(f"{w}:{len(v)}" for w, v in sorted(groups.items(), key=lambda x: -len(x[1]))[:25]), flush=True)
    return groups


def load_critic(path):
    from nla.flow.scoring import FlowBundle
    from playground_app import resolve_base
    aa = torch.load(path, map_location="cpu")["args"]; pco = os.path.join(os.path.dirname(path), "prior_cotrained_latest.pt")
    fb = FlowBundle(aa["prior"], path, aa["stats"], DEV, base=resolve_base("Qwen/Qwen3.6-27B", local_snapshot=True), enc_layer=aa.get("enc_layer", 42),
                    ar_ckpt=aa.get("ar_ckpt", "/vol/ckpts/qwen36_27b/ar_sft_merged"), prior_override=pco if os.path.exists(pco) else None)
    fb.model.eval(); return fb, (aa.get("claim_subsets") or 0) > 0


@torch.no_grad()
def vel(fb, x, t, conds):
    enc, mk, cv = conds; tt = torch.full((x.shape[0],), float(t), device=DEV)
    with torch.autocast("cuda", dtype=torch.bfloat16): return fb.model(x, tt, enc, mk, cv).float()


def rep(c, idx):
    return tuple(None if x is None else x[idx] for x in c)


@torch.no_grad()
def evaluate(fb, bullet, groups, a):
    words = sorted(groups); W = len(words); texts = [("• " if bullet else "") + TEMPLATE.format(w=w) for w in words]
    conds = fb.cond(texts); d = fb.norm.mean.shape[0] if hasattr(fb.norm, "mean") else 5120
    mu = torch.stack([fb.norm.normalize(torch.tensor(groups[w], device=DEV)).float().mean(0) for w in words])     # [W, d]
    g_ = torch.Generator(device=DEV).manual_seed(0); M = torch.zeros(W, d, device=DEV)
    for k in range(a.eps):
        E = torch.randn(W, d, device=DEV, generator=g_); M += (E - vel(fb, E, 1.0, conds)) / a.eps           # m(c_w)
    ii, jj = torch.triu_indices(W, W, 1, device=DEV)
    dm, dmu = M[jj] - M[ii], mu[jj] - mu[ii]
    cm_align = F.cosine_similarity(dm, dmu, dim=-1); cm_ratio = dm.norm(dim=-1) / dmu.norm(dim=-1).clamp_min(1e-8)
    # zero-shot retrieval on held-out anchors (centred)
    Mc = F.normalize(M - M.mean(0), dim=-1); hits, n_ = 0, 0; hbar = torch.cat([fb.norm.normalize(torch.tensor(groups[w], device=DEV)).float() for w in words]).mean(0)
    for wi, w in enumerate(words):
        X = fb.norm.normalize(torch.tensor(groups[w][: a.retr_n], device=DEV)).float(); pred = (F.normalize(X - hbar, dim=-1) @ Mc.t()).argmax(1)
        hits += int((pred == wi).sum()); n_ += len(X)
    # delta direction at held-out anchors
    rng = random.Random(0); dl = {t: [] for t in a.ts}
    for wi, w in enumerate(words):
        X = fb.norm.normalize(torch.tensor(groups[w][: a.dlt_n], device=DEV)).float(); ys = [rng.choice([j for j in range(W) if j != wi]) for _ in range(len(X))]
        cx = rep(conds, torch.full((len(X),), wi, device=DEV)); cy = rep(conds, torch.tensor(ys, device=DEV))
        for t in a.ts:
            D = (X - t * vel(fb, X, t, cy)) - (X - t * vel(fb, X, t, cx))
            dl[t] += F.cosine_similarity(D, mu[ys] - mu[wi], dim=-1).tolist()
    return dict(n_words=W, words=words, cm_align_mean=float(cm_align.mean()), cm_align_median=float(cm_align.median()), cm_ratio_median=float(cm_ratio.median()),
                cm_retrieval=hits / max(n_, 1), retrieval_chance=1.0 / W, n_retrieval=n_,
                dlt_align={f"{t:g}": float(np.mean(v)) for t, v in dl.items()}, bullet=bullet)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--critics", required=True); ap.add_argument("--tag", default="popalign")
    ap.add_argument("--glob", default="/vol_glp/claims/final/final_v2_*.parquet"); ap.add_argument("--max-files", type=int, default=60)
    ap.add_argument("--min-n", type=int, default=40); ap.add_argument("--max-per-word", type=int, default=400); ap.add_argument("--eps", type=int, default=16)
    ap.add_argument("--retr-n", type=int, default=100); ap.add_argument("--dlt-n", type=int, default=8); ap.add_argument("--ts", default="0.1,0.3")
    ap.add_argument("--no-bullet-too", action="store_true", help="also score claim-trained critics on unbulleted text (the harness format)")
    a = ap.parse_args(); a.ts = [float(x) for x in a.ts.split(",")]; os.makedirs(OUT, exist_ok=True); t0 = time.time()
    groups = load_heldout(a.glob, a.max_files, a.min_n, a.max_per_word); res = dict(tag=a.tag, glob=a.glob, n_words=len(groups), critics={})
    for c in a.critics.split(","):
        name, _, path = c.partition("="); t1 = time.time(); fb, bullet = load_critic(path)
        res["critics"][name] = dict(path=path, **evaluate(fb, bullet, groups, a))
        if bullet and a.no_bullet_too: res["critics"][name + "_nobullet"] = dict(path=path, **evaluate(fb, False, groups, a))
        for k in [name] + ([name + "_nobullet"] if bullet and a.no_bullet_too else []):
            r = res["critics"][k]; print(f"[popalign] {k:24s} cm_align {r['cm_align_mean']:+.3f} (median {r['cm_align_median']:+.3f}) ratio {r['cm_ratio_median']:.2f} | retrieval {100*r['cm_retrieval']:.1f}% "
                                          f"(chance {100*r['retrieval_chance']:.1f}%) | delta align " + " ".join(f"t{t}: {v:+.3f}" for t, v in r["dlt_align"].items()) + f" | {time.time() - t1:.0f}s", flush=True)
        del fb; torch.cuda.empty_cache()
        json.dump(res, open(f"{OUT}/{a.tag}.json", "w"), indent=1)
    print(f"[done] {time.time() - t0:.0f}s -> {OUT}/{a.tag}.json", flush=True)


if __name__ == "__main__":
    main()
