"""h_i-only baselines on the SAME train pairs and the same held-out val rows as the reconstructor: identity (delta_hat = 0, FVE 0 by
definition), an h_i-only MLP with no depth, and an h_i-only MLP TOLD (i, j) (a forbidden diagnostic: what depth alone is worth).
Same loss as R (relative MSE + cos).

  python -m nlt.bullets.baselines --data-dir /vol/data/qwen3_8b --pair-ids /vol/bullets/R_bullets/train_pairs.json --out /vol/bullets/baselines
"""
from __future__ import annotations
import argparse, json, math, os, time
import numpy as np, torch, torch.nn as nn
from nlt.data.dataset import GlobalNorm
from nlt.data.extract import K_LO, N_LAYERS
from nlt.bullets.data import load_pairs, gather_acts, band_of, gap_bucket_of
from nlt.bullets.model import recon_loss


class MLP(nn.Module):
    def __init__(self, d=4096, hidden=4096, n_hidden=2, depth=False, d_depth=256):
        super().__init__(); self.depth = depth
        din = d + (2 * d_depth if depth else 0)
        if depth: self.emb_i = nn.Embedding(N_LAYERS, d_depth); self.emb_j = nn.Embedding(N_LAYERS, d_depth)
        layers = [nn.Linear(din, hidden), nn.SiLU()]
        for _ in range(n_hidden - 1): layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, d)]; self.net = nn.Sequential(*layers)
        nn.init.normal_(self.net[-1].weight, std=1e-3); nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, i=None, j=None):
        if self.depth: x = torch.cat([x, self.emb_i(i - K_LO), self.emb_j(j - K_LO)], -1)
        return self.net(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--out", required=True); p.add_argument("--pair-ids", required=True)
    p.add_argument("--n-val", type=int, default=1536); p.add_argument("--steps", type=int, default=2000); p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4); p.add_argument("--wd", type=float, default=0.01); p.add_argument("--cos-w", type=float, default=0.5)
    p.add_argument("--extra-train", type=int, default=0, help="ALSO train both MLPs on this many random extra train pairs (data-scaling diagnostic)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = "cuda"; os.makedirs(a.out, exist_ok=True)
    norm = GlobalNorm.load(os.path.join(a.data_dir, "stats.pt"), "affine").to(dev)
    ptr = load_pairs(a.data_dir, "train"); ids = set(json.load(open(a.pair_ids))); tr = ptr[ptr["pair_id"].isin(ids)].reset_index(drop=True)
    pv = load_pairs(a.data_dir, "val").iloc[: a.n_val].reset_index(drop=True)
    Htr = gather_acts(a.data_dir, "train", tr["pos_idx"].values, tr["i"].values, tr["j"].values); Hva = gather_acts(a.data_dir, "val", pv["pos_idx"].values, pv["i"].values, pv["j"].values)
    Xtr = norm.normalize(Htr["h_i"].to(dev)); Dtr = norm.normalize(Htr["h_j"].to(dev)) - Xtr; Itr = torch.as_tensor(tr["i"].values).long().to(dev); Jtr = torch.as_tensor(tr["j"].values).long().to(dev)
    Xva = norm.normalize(Hva["h_i"].to(dev)); Dva = norm.normalize(Hva["h_j"].to(dev)) - Xva; Iva = torch.as_tensor(pv["i"].values).long().to(dev); Jva = torch.as_tensor(pv["j"].values).long().to(dev)
    EN = (Dva ** 2).sum(-1); band = band_of(pv["j"].values); gapb = gap_bucket_of((pv["j"] - pv["i"]).values)
    sets = {"matched": (Xtr, Dtr, Itr, Jtr)}
    if a.extra_train:
        ex = ptr.sample(n=min(a.extra_train, len(ptr)), random_state=a.seed).reset_index(drop=True)
        Hx = gather_acts(a.data_dir, "train", ex["pos_idx"].values, ex["i"].values, ex["j"].values)
        Xx = norm.normalize(Hx["h_i"].to(dev)); sets["extra"] = (Xx, norm.normalize(Hx["h_j"].to(dev)) - Xx, torch.as_tensor(ex["i"].values).long().to(dev), torch.as_tensor(ex["j"].values).long().to(dev))

    def fit(depth, X, D, I, J, steps):
        m = MLP(depth=depth).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95)); N = len(X); t0 = time.time()
        for step in range(steps):
            f = min(1.0, (step + 1) / 50) * (0.5 * (1 + math.cos(math.pi * step / steps)) * 0.95 + 0.05)
            for g in opt.param_groups: g["lr"] = a.lr * f
            idx = torch.as_tensor(rng.integers(0, N, a.batch), device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16): pred = m(X[idx], I[idx], J[idx]).float()
            loss, _, _ = recon_loss(pred, D[idx], a.cos_w); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            if step % 250 == 0: print(f"[mlp depth={depth}] {step}/{steps} loss {float(loss):.4f} {(time.time() - t0) / (step + 1):.3f}s/step", flush=True)
        return m.eval()

    @torch.no_grad()
    def evaluate(m):
        se = []
        for s in range(0, len(Xva), 256):
            with torch.autocast("cuda", dtype=torch.bfloat16): pred = m(Xva[s:s + 256], Iva[s:s + 256], Jva[s:s + 256]).float()
            se.append(((pred - Dva[s:s + 256]) ** 2).sum(-1))
        se = torch.cat(se); r = (se / EN).cpu().numpy()
        fve = lambda mask=None: float(1 - (se if mask is None else se[torch.as_tensor(mask, device=dev)]).sum() / (EN if mask is None else EN[torch.as_tensor(mask, device=dev)]).sum())
        return {"fve": fve(), "mean_relmse": float(r.mean()), "by_band": {b: {"n": int((band == b).sum()), "fve": fve(band == b), "mean_relmse": float(r[band == b].mean())} for b in ["pre", "workspace", "motor"] if (band == b).any()},
                "by_gap": {g: {"n": int((gapb == g).sum()), "fve": fve(gapb == g)} for g in np.unique(gapb) if g}}, se

    out = {"n_train_matched": len(tr), "n_val": len(pv), "identity": {"fve": 0.0, "mean_relmse": 1.0, "note": "delta_hat = 0 is the reference: FVE of delta is 0 by definition"}}
    for name, (X, D, I, J) in sets.items():
        steps = a.steps if name == "matched" else max(a.steps, int(len(X) / a.batch * 3))
        for depth in (False, True):
            m = fit(depth, X, D, I, J, steps); ev, se = evaluate(m); key = f"mlp_{'depth' if depth else 'nodepth'}_{name}"
            ev.update({"n_train": len(X), "steps": steps}); out[key] = ev; print(f"[baseline] {key}: {json.dumps({k: v for k, v in ev.items() if not isinstance(v, dict)})}", flush=True)
            np.save(os.path.join(a.out, f"se_{key}.npy"), se.cpu().numpy())
            del m; torch.cuda.empty_cache()
    json.dump(out, open(os.path.join(a.out, "baselines.json"), "w"), indent=1); print("[baseline] DONE", flush=True)


if __name__ == "__main__":
    main()
