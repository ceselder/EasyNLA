"""Deterministic MSE transcoder baselines h_i -> h_j (FVE reference for the flow critic): linear or MLP, with / without the
FORBIDDEN depth embedding. Same j-agnostic normalisation as the critic. FVE is reported against two references:
  vs_globalmean : 1 - sum||pred - x||^2 / sum||x - 0||^2 in the pooled-normalised space (0 = the global mean; j-agnostic)
  vs_layermean  : 1 - sum||pred - x||^2 / sum||x - mean_j||^2 with the per-layer mean of the target (metric only; the model never sees j)
plus the identity transcoder (pred = h_i) under both.
"""
from __future__ import annotations
import argparse, json, math, os, time
import numpy as np, torch, torch.nn as nn
from nlt.data.dataset import ActStore, GlobalNorm
from nlt.data.extract import K_LO, K_HI, N_LAYERS
from nlt.critic.train import GAP_BUCKETS


class Transcoder(nn.Module):
    def __init__(self, d, arch="mlp", hidden=4096, n_hidden=2, depth=False, d_depth=256):
        super().__init__()
        self.depth = depth
        din = d + (2 * d_depth if depth else 0)
        if depth: self.emb_i = nn.Embedding(N_LAYERS, d_depth); self.emb_j = nn.Embedding(N_LAYERS, d_depth)
        if arch == "linear": self.net = nn.Linear(din, d)
        else:
            layers = [nn.Linear(din, hidden), nn.SiLU()]
            for _ in range(n_hidden - 1): layers += [nn.Linear(hidden, hidden), nn.SiLU()]
            layers += [nn.Linear(hidden, d)]; self.net = nn.Sequential(*layers)
        self.skip = True          # predict the residual on top of h_i (identity init in effect)

    def forward(self, h_i, i=None, j=None):
        x = h_i
        if self.depth: x = torch.cat([h_i, self.emb_i(i - K_LO), self.emb_j(j - K_LO)], -1)
        return self.net(x) + (h_i if self.skip else 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--out", required=True); p.add_argument("--tag", default="mse")
    p.add_argument("--arch", default="mlp", choices=["linear", "mlp"]); p.add_argument("--hidden", type=int, default=4096); p.add_argument("--n-hidden", type=int, default=2); p.add_argument("--depth", action="store_true")
    p.add_argument("--steps", type=int, default=4000); p.add_argument("--batch", type=int, default=1024); p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--max-train-pos", type=int, default=None); p.add_argument("--data-device", default="cuda"); p.add_argument("--norm", default="affine"); p.add_argument("--eval-n", type=int, default=8192)
    p.add_argument("--wandb", default="nlt-qwen3-8b"); p.add_argument("--wandb-entity", default="octahedral-systems"); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(); torch.manual_seed(a.seed); dev = "cuda"; os.makedirs(a.out, exist_ok=True)
    norm = GlobalNorm.load(os.path.join(a.data_dir, "stats.pt"), a.norm).to(dev)
    store = ActStore(a.data_dir, "train", device=a.data_device, max_pos=a.max_train_pos); store_val = ActStore(a.data_dir, "val", device=a.data_device)
    import pyarrow.parquet as pq
    vp = pq.read_table(os.path.join(a.data_dir, "pairs_val.parquet")).to_pandas(); vp = vp[vp["pos_idx"].isin(store_val.row_of)].iloc[: a.eval_n]
    val_rows = store_val.rows_for(vp["pos_idx"].values); val_i = torch.tensor(vp["i"].values); val_j = torch.tensor(vp["j"].values)
    # per-layer mean of the normalised target (metric reference only)
    st = torch.load(os.path.join(a.data_dir, "stats.pt"), map_location="cpu")
    layer_mean = norm.normalize(st["per_layer"]["mean"].to(dev))                       # [L, d]
    model = Transcoder(store.d, a.arch, a.hidden, a.n_hidden, a.depth).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd); gen = torch.Generator().manual_seed(a.seed)
    import wandb; wandb.init(project=a.wandb, entity=a.wandb_entity, name=a.tag, config=vars(a) | {"n_params": sum(x.numel() for x in model.parameters())})
    print(f"[mse] {a.arch} depth={a.depth}: {sum(x.numel() for x in model.parameters())/1e6:.0f}M params", flush=True)

    @torch.no_grad()
    def evaluate():
        model.eval(); B = 512; se = []; se_id = []; en_g = []; en_l = []
        for s in range(0, len(val_rows), B):
            rows, i, j = val_rows[s:s + B], val_i[s:s + B], val_j[s:s + B]
            h_i = norm.normalize(store_val.gather(rows, i, dev)); x = norm.normalize(store_val.gather(rows, j, dev))
            with torch.autocast("cuda", dtype=torch.bfloat16): pred = model(h_i, i.to(dev), j.to(dev)).float()
            se.append(((pred - x) ** 2).sum(-1).cpu()); se_id.append(((h_i - x) ** 2).sum(-1).cpu()); en_g.append((x ** 2).sum(-1).cpu()); en_l.append(((x - layer_mean[(j - K_LO).to(dev)]) ** 2).sum(-1).cpu())
        model.train(); se, se_id, en_g, en_l = map(torch.cat, (se, se_id, en_g, en_l)); gaps = (val_j - val_i).numpy(); js = val_j.numpy()
        def fve(num, den, m=None):
            if m is None: return float(1 - num.sum() / den.sum())
            return float(1 - num[m].sum() / den[m].sum())
        out = {"fve_vs_globalmean": fve(se, en_g), "fve_vs_layermean": fve(se, en_l), "identity_fve_vs_globalmean": fve(se_id, en_g), "identity_fve_vs_layermean": fve(se_id, en_l),
               "mse_per_dim": float(se.mean() / store.d), "identity_mse_per_dim": float(se_id.mean() / store.d), "n": len(val_rows)}
        br = {"by_gap": {}, "by_j": {}}
        for lo, hi in GAP_BUCKETS:
            m = (gaps >= lo) & (gaps <= hi); key = f"{lo}-{hi}" if lo != hi else f"{lo}"
            if m.sum(): br["by_gap"][key] = {"n": int(m.sum()), "fve_vs_globalmean": fve(se, en_g, m), "fve_vs_layermean": fve(se, en_l, m), "identity_fve_vs_layermean": fve(se_id, en_l, m), "mse_per_dim": float(se[m].mean() / store.d)}
        for jj in range(K_LO + 1, K_HI + 1):
            m = js == jj
            if m.sum(): br["by_j"][str(jj)] = {"n": int(m.sum()), "fve_vs_globalmean": fve(se, en_g, m), "fve_vs_layermean": fve(se, en_l, m), "identity_fve_vs_layermean": fve(se_id, en_l, m), "mse_per_dim": float(se[m].mean() / store.d)}
        return out, br
    t0 = time.time()
    for step in range(a.steps):
        rows, i, j = store.sample_pairs(a.batch, gen)
        h_i = norm.normalize(store.gather(rows, i, dev)); x = norm.normalize(store.gather(rows, j, dev))
        lr = a.lr * min(1.0, (step + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        for g_ in opt.param_groups: g_["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16): pred = model(h_i, i.to(dev), j.to(dev)).float()
        loss = ((pred - x) ** 2).mean(); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 50 == 0: wandb.log({"train/mse": loss.item(), "train/lr": lr}, step=step)
        if step % 200 == 0: print(f"[mse] step {step} mse/dim {loss.item():.4f} {(time.time()-t0)/(step+1):.3f}s/step", flush=True)
        if (step + 1) % 1000 == 0 or step + 1 == a.steps:
            out, br = evaluate(); wandb.log({f"eval/{k}": v for k, v in out.items()}, step=step); print(f"[mse eval@{step+1}] {json.dumps({k: round(v, 4) for k, v in out.items()})}", flush=True)
    json.dump({"args": vars(a), "scalars": out, "breakdown": br}, open(os.path.join(a.out, "mse_eval.json"), "w"), indent=1)
    torch.save({"model": model.state_dict(), "args": vars(a)}, os.path.join(a.out, "ckpt_final.pt")); wandb.finish(); print("[mse] DONE", flush=True)


if __name__ == "__main__":
    main()
