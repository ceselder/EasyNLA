"""Decodability data-scaling study, probes (1 B200) — held-out accuracy vs training-set size and probe capacity, per detail type and distance.

Data: scripts/decodability_scale_build.py output. Tasks
  other_doc  2-AFC: is the true value or the true value of another row (same type, other document, not in this context) the one in the context
  near       numbers: true value vs its near-miss (10-40 % off, years +-1..30); probe TRAINED on near-miss negatives
  digits     numbers, activation only: last digit and first digit (10-way each), vs the majority class
Capacities: linear (bilinear in h and value features, rank 256), mlp2 (2 x 2048), mlp4 (4 x 4096), tf (4-layer transformer over 40 chunks of h
+ value tokens). Training sets are stratified over the six distance buckets and NESTED across sizes (a fixed seeded order per bucket).
Every 2-AFC run is repeated with the activations permuted within each split (value-only floor). Value features: mean / first / last frozen
Qwen3.6-27B input embedding of the value's tokens, a 4,096-bin bag of hashed character 2-3-grams, digit-position one-hots.
usage: python scripts/decodability_scale_train.py --data-dir /vol_glp/decodability/scale --types number --out /vol_glp/decodability/scale/results_number.json
"""
from __future__ import annotations
import argparse, glob, json, math, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
BNAME = ["0", "1", "2-4", "5-16", "17-64", "65-256"]; NHASH = 4096


def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load_embeddings(base, dev):
    from huggingface_hub import snapshot_download, hf_hub_download
    from safetensors import safe_open
    snap = snapshot_download(base, token=os.environ.get("HF_TOKEN"), local_dir="/root/base_snap", allow_patterns=["*.json"])
    idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]; key = [k for k in idx if k.endswith("embed_tokens.weight")][0]
    path = hf_hub_download(base, idx[key], token=os.environ.get("HF_TOKEN"), local_dir="/root/base_snap")
    with safe_open(path, framework="pt", device="cpu") as f: E = f.get_tensor(key)
    return E.to(dev).half()


class Values:
    """value id -> feature vector [3*5120 + 131 + 4096], computed per batch from the frozen embedding table (+ standardisation)."""
    def __init__(self, data_dir, typ, E, dev):
        self.E = E; self.dev = dev
        self.tok = torch.tensor(np.load(os.path.join(data_dir, f"values_{typ}_tokens.npy")), device=dev)              # [V, 32] int32, -1 pad
        self.hsh = torch.tensor(np.load(os.path.join(data_dir, f"values_{typ}_hash.npy")).astype(np.int64), device=dev)  # [V, 64], -1 pad
        self.dig = torch.tensor(np.load(os.path.join(data_dir, f"values_{typ}_digits.npy")), device=dev).float()        # [V, 131]
        self.mu = None; self.sd = None; self.dim = 3 * E.shape[1] + 131 + NHASH
    def raw(self, ids):
        t = self.tok[ids]; m = (t >= 0); tt = t.clamp(min=0); e = self.E[tt].float()                                   # [B, 32, 5120]
        mf = m.float().unsqueeze(-1); mean = (e * mf).sum(1) / mf.sum(1).clamp(min=1); first = e[:, 0]
        last_idx = (m.sum(1) - 1).clamp(min=0); last = e[torch.arange(len(ids), device=self.dev), last_idx]
        h = self.hsh[ids]; hm = (h >= 0); bag = torch.zeros(len(ids), NHASH, device=self.dev); bag.scatter_add_(1, h.clamp(min=0), hm.float())
        return torch.cat([mean, first, last, self.dig[ids], bag], -1)
    def fit(self, ids):
        with torch.no_grad():
            X = torch.cat([self.raw(ids[i: i + 4096]) for i in range(0, len(ids), 4096)]); self.mu = X.mean(0, keepdim=True); self.sd = X.std(0, keepdim=True) + 1e-3
    def __call__(self, ids):
        with torch.no_grad(): return (self.raw(ids) - self.mu) / self.sd


class Bilinear(nn.Module):
    def __init__(self, dh, dv, r=256):
        super().__init__(); self.A = nn.Linear(dh, r, bias=False); self.B = nn.Linear(dv, r, bias=False); self.wv = nn.Linear(dv, 1); self.r = r
    def forward(self, h, v): return (self.A(h) * self.B(v)).sum(-1) / math.sqrt(self.r) + self.wv(v).squeeze(-1)


class MLP(nn.Module):
    def __init__(self, dh, dv, w, depth, out=1):
        super().__init__(); L = []; d = dh + dv
        for _ in range(depth): L += [nn.Linear(d, w), nn.GELU(), nn.Dropout(0.1)]; d = w
        self.net = nn.Sequential(*L, nn.Linear(d, out))
    def forward(self, h, v=None): return self.net(torch.cat([h, v], -1) if v is not None else h).squeeze(-1)


class ChunkTransformer(nn.Module):
    """h -> 40 tokens of 128 dims; value -> 3 embedding tokens + digits token + hash-bag token; CLS readout."""
    def __init__(self, dh=5120, d=384, layers=4, heads=6, out=1, with_value=True):
        super().__init__(); self.nch = dh // 128; self.inp = nn.Linear(128, d); self.pos = nn.Parameter(torch.randn(1, self.nch + 6, d) * 0.02); self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.with_value = with_value
        if with_value: self.vemb = nn.Linear(5120, d); self.vdig = nn.Linear(131, d); self.vhash = nn.Linear(NHASH, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True, activation="gelu"); self.enc = nn.TransformerEncoder(enc, layers); self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, out))
    def forward(self, h, v=None):
        B = h.shape[0]; toks = [self.cls.expand(B, -1, -1), self.inp(h.view(B, self.nch, 128))]
        if self.with_value and v is not None:
            e = v[:, : 3 * 5120].view(B, 3, 5120); toks += [self.vemb(e), self.vdig(v[:, 3 * 5120: 3 * 5120 + 131]).unsqueeze(1), self.vhash(v[:, 3 * 5120 + 131:]).unsqueeze(1)]
        x = torch.cat(toks, 1); x = x + self.pos[:, : x.shape[1]]; return self.out(self.enc(x)[:, 0]).squeeze(-1)


def make_model(cap, dh, dv, out=1, with_value=True):
    if cap == "linear": return Bilinear(dh, dv) if with_value else nn.Sequential(nn.Linear(dh, out))
    if cap == "mlp2": return MLP(dh, dv if with_value else 0, 2048, 2, out)
    if cap == "mlp4": return MLP(dh, dv if with_value else 0, 4096, 4, out)
    if cap == "tf": return ChunkTransformer(dh, out=out, with_value=with_value)
    raise ValueError(cap)


def lr_of(cap): return 1e-3 if cap == "linear" else 3e-4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--out", required=True); p.add_argument("--types", default="number"); p.add_argument("--base", default="Qwen/Qwen3.6-27B")
    p.add_argument("--sizes", default="10000,30000,100000,300000,1000000,3000000"); p.add_argument("--caps", default="linear,mlp2,mlp4,tf"); p.add_argument("--tasks", default="other_doc,near,digits")
    p.add_argument("--bs", type=int, default=1024); p.add_argument("--min-steps", type=int, default=600); p.add_argument("--max-steps", type=int, default=12000); p.add_argument("--epochs", type=float, default=8.0)
    p.add_argument("--test-per-bucket", type=int, default=5000); p.add_argument("--val-per-bucket", type=int, default=2000); p.add_argument("--seed", type=int, default=0); p.add_argument("--no-floor", action="store_true")
    a = p.parse_args(); dev = "cuda:0"; t0 = time.time(); torch.backends.cuda.matmul.allow_tf32 = True
    sizes = [int(x) for x in a.sizes.split(",")]; caps = a.caps.split(","); tasks = a.tasks.split(",")
    E = load_embeddings(a.base, dev); print(f"[scale-train] embeddings {tuple(E.shape)} ({time.time() - t0:.0f}s)", flush=True)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    def save(): json.dump(res, open(a.out, "w"), indent=1)

    for typ in a.types.split(","):
        M = pq.read_table(os.path.join(a.data_dir, f"meta_{typ}.parquet")).to_pandas()
        files = sorted(glob.glob(os.path.join(a.data_dir, f"h_{typ}_*.npy"))); H = torch.cat([torch.from_numpy(np.load(f)).to(dev) for f in files]); assert H.shape[0] == len(M), (H.shape, len(M))
        print(f"[scale-train] {typ}: {len(M)} rows, h {tuple(H.shape)} fp16 on GPU ({time.time() - t0:.0f}s)", flush=True)
        vals = Values(a.data_dir, typ, E, dev); rs = np.random.default_rng(a.seed)
        bidx = np.array([BNAME.index(b) for b in M["bucket"]]); split = M["split"].to_numpy(); vid = torch.tensor(M["value_id"].to_numpy(), device=dev); nid = torch.tensor(M["neg_id"].to_numpy(), device=dev)
        ok_neg = M["neg_id"].to_numpy() >= 0
        # ---- fixed test / val sets (stratified), nested training order per bucket
        def pick(mask, per_bucket):
            out = []
            for b in range(6):
                ix = np.where(mask & (bidx == b))[0]; rs.shuffle(ix); out.append(ix[: per_bucket])
            return out
        test_b = pick((split == 0) & ok_neg, a.test_per_bucket); val_b = pick((split == 1) & ok_neg, a.val_per_bucket); train_b = pick((split == 2) & ok_neg, 10 ** 9)
        test_idx = np.concatenate(test_b); val_idx = np.concatenate(val_b); n_train_avail = sum(len(x) for x in train_b)
        print(f"[scale-train] {typ}: train avail {n_train_avail} " + " ".join(f"{BNAME[b]}:{len(train_b[b])}" for b in range(6)) + f" | val {len(val_idx)} | test {len(test_idx)}", flush=True)
        def train_subset(N):
            """stratified, nested: N/6 per bucket, shortfalls redistributed to the buckets that have more."""
            quota = [0] * 6; left = N; avail = [len(x) for x in train_b]; order = sorted(range(6), key=lambda b: avail[b])
            for i, b in enumerate(order):
                q = min(avail[b], left // (6 - i)); quota[b] = q; left -= q
            return np.concatenate([train_b[b][: quota[b]] for b in range(6)]), quota
        # standardisation of h from a 200k training sample; value features from the same rows
        samp = train_subset(min(200000, n_train_avail))[0]; hs = H[torch.tensor(samp, device=dev)].float(); mu, sd = hs.mean(0, keepdim=True), hs.std(0, keepdim=True) + 1e-3; del hs
        vals.fit(torch.cat([vid[torch.tensor(samp[:50000], device=dev)], nid[torch.tensor(samp[:50000], device=dev)]]))
        def geth(ix, hmap=None):
            ix_t = torch.tensor(ix, device=dev) if not torch.is_tensor(ix) else ix
            if hmap is not None: ix_t = hmap[ix_t]
            return (H[ix_t].float() - mu) / sd
        def floor_map():
            """permute the activation rows WITHIN each split (test rows among test rows, etc.), so no pairing survives"""
            m = torch.arange(len(M), device=dev)
            for s in (0, 1, 2):
                ix = np.where(split == s)[0]; m[torch.tensor(ix, device=dev)] = torch.tensor(rs.permutation(ix), device=dev)
            return m
        T = res.setdefault(typ, {"n_rows": int(len(M)), "train_avail_per_bucket": {BNAME[b]: int(len(train_b[b])) for b in range(6)}, "n_test": int(len(test_idx)), "n_val": int(len(val_idx)), "tasks": {}})

        def run_pair(cap, tr_idx, va_idx, te_idx, pos_id, neg_id, hmap=None, tag=""):
            torch.manual_seed(a.seed); m = make_model(cap, 5120, vals.dim).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=lr_of(cap), weight_decay=1e-2)
            N = len(tr_idx); steps = int(min(max(a.epochs * N / a.bs, a.min_steps), a.max_steps)); ev = max(100, steps // 20); best = (-1.0, None, 0); tr_t = torch.tensor(tr_idx, device=dev); t1 = time.time()
            def score(ix):
                m.eval(); out = []
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for s in range(0, len(ix), 4096):
                        j = torch.tensor(ix[s: s + 4096], device=dev); h = geth(j, hmap); out.append((m(h, vals(pos_id[j])) > m(h, vals(neg_id[j]))).float())
                m.train(); return torch.cat(out).cpu().numpy()
            for st in range(steps):
                j = tr_t[torch.randint(0, N, (a.bs,), device=dev)]; h = geth(j, hmap)
                with torch.autocast("cuda", dtype=torch.bfloat16): loss = F.softplus(m(h, vals(neg_id[j])) - m(h, vals(pos_id[j]))).mean()
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
                if (st + 1) % ev == 0 or st == steps - 1:
                    acc = float(score(va_idx).mean())
                    if acc > best[0]: best = (acc, {k: v.detach().clone() for k, v in m.state_dict().items()}, st + 1)
            m.load_state_dict(best[1]); win = score(te_idx); bk = bidx[te_idx]; out = {}
            for b in range(6):
                sel = win[bk == b]
                if len(sel): out[BNAME[b]] = {"acc": float(sel.mean()), "n": int(len(sel)), "ci": wilson(int(sel.sum()), len(sel))}
            out["all"] = {"acc": float(win.mean()), "n": int(len(win)), "ci": wilson(int(win.sum()), len(win))}
            r = {"val_acc": best[0], "best_step": best[2], "steps": steps, "n_train": int(N), "test": out, "time_s": time.time() - t1}
            print(f"[scale-train] {typ}/{tag}/{cap}/N={N}: val {best[0]:.3f} @ {best[2]}/{steps} | test " + " ".join(f"{b}:{v['acc']:.3f}" for b, v in out.items()) + f" ({time.time() - t1:.0f}s)", flush=True)
            del m, opt; torch.cuda.empty_cache(); return r

        def run_cls(cap, tr_idx, va_idx, te_idx, y, ncls, hmap=None, tag=""):
            torch.manual_seed(a.seed); m = make_model(cap, 5120, 0, out=ncls, with_value=False).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=lr_of(cap), weight_decay=1e-2)
            N = len(tr_idx); steps = int(min(max(a.epochs * N / a.bs, a.min_steps), a.max_steps)); ev = max(100, steps // 20); best = (-1.0, None, 0); tr_t = torch.tensor(tr_idx, device=dev); t1 = time.time()
            def pred(ix):
                m.eval(); out = []
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for s in range(0, len(ix), 4096): j = torch.tensor(ix[s: s + 4096], device=dev); out.append(m(geth(j, hmap)).argmax(-1))
                m.train(); return torch.cat(out).cpu().numpy()
            for st in range(steps):
                j = tr_t[torch.randint(0, N, (a.bs,), device=dev)]
                with torch.autocast("cuda", dtype=torch.bfloat16): loss = F.cross_entropy(m(geth(j, hmap)).float(), y[j])
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
                if (st + 1) % ev == 0 or st == steps - 1:
                    acc = float((pred(va_idx) == y[torch.tensor(va_idx, device=dev)].cpu().numpy()).mean())
                    if acc > best[0]: best = (acc, {k: v.detach().clone() for k, v in m.state_dict().items()}, st + 1)
            m.load_state_dict(best[1]); pr = pred(te_idx); yt = y[torch.tensor(te_idx, device=dev)].cpu().numpy(); ytr = y[tr_t].cpu().numpy(); maj = np.bincount(ytr, minlength=ncls).argmax(); bk = bidx[te_idx]; out = {}
            for b in list(range(6)) + ["all"]:
                sel = np.ones(len(te_idx), bool) if b == "all" else (bk == b)
                if sel.sum(): out[BNAME[b] if b != "all" else "all"] = {"acc": float((pr[sel] == yt[sel]).mean()), "n": int(sel.sum()), "ci": wilson(int((pr[sel] == yt[sel]).sum()), int(sel.sum())), "majority": float((yt[sel] == maj).mean())}
            r = {"val_acc": best[0], "best_step": best[2], "steps": steps, "n_train": int(N), "test": out, "time_s": time.time() - t1}
            print(f"[scale-train] {typ}/{tag}/{cap}/N={N}: val {best[0]:.3f} | test " + " ".join(f"{b}:{v['acc']:.3f}[{v['majority']:.2f}]" for b, v in out.items()) + f" ({time.time() - t1:.0f}s)", flush=True)
            del m, opt; torch.cuda.empty_cache(); return r

        for task in tasks:
            if task == "near" and typ != "number": continue
            if task == "digits" and typ != "number": continue
            TT = T["tasks"].setdefault(task, {})
            if task in ("other_doc", "near"):
                if task == "near":
                    near = M["near_id"].to_numpy(); okn = (near >= 0) & (M["near_in_ctx"].to_numpy() == 0); neg_all = torch.tensor(near, device=dev)
                    tr_bn = [x[okn[x]] for x in train_b]; va_i = val_idx[okn[val_idx]]; te_i = test_idx[okn[test_idx]]
                else: neg_all = nid; tr_bn = train_b; va_i = val_idx; te_i = test_idx
                def subset(N, tb=tr_bn):
                    quota = [0] * 6; left = N; avail = [len(x) for x in tb]; order = sorted(range(6), key=lambda b: avail[b])
                    for i, b in enumerate(order): q = min(avail[b], left // (6 - i)); quota[b] = q; left -= q
                    return np.concatenate([tb[b][: quota[b]] for b in range(6)]), quota
                avail_n = sum(len(x) for x in tr_bn); TT["train_avail"] = int(avail_n)
                for N in sizes:
                    if N > avail_n * 1.05 and str(N) in TT.get("skipped", []): continue
                    if N > avail_n * 1.05:
                        if sizes.index(N) > 0 and sizes[sizes.index(N) - 1] >= avail_n: TT.setdefault("skipped", []).append(str(N)); continue
                    tr_i, quota = subset(min(N, avail_n)); key = str(N)
                    for cap in caps:
                        C = TT.setdefault(key, {}).setdefault(cap, {})
                        if "real" not in C: C["real"] = run_pair(cap, tr_i, va_i, te_i, vid, neg_all, tag=task); C["quota"] = {BNAME[b]: int(q) for b, q in enumerate(quota)}; save()
                        if not a.no_floor and "floor" not in C: C["floor"] = run_pair(cap, tr_i, va_i, te_i, vid, neg_all, hmap=floor_map(), tag=task + "-floor"); save()
                        if task == "near" and "other_doc_eval" not in C:   # the near-trained probe scored on other-document negatives, one shot
                            pass
            elif task == "digits":
                V = json.load(open(os.path.join(a.data_dir, f"values_{typ}_strings.json")))
                import re
                def digits(v): return re.sub(r"[^\d]", "", v.split(".")[0] if "." in v else v)
                dstr = [digits(V[i]) for i in M["value_id"].to_numpy()]; okd = np.array([len(d) > 0 for d in dstr])
                for nm, fn in (("last_digit", lambda d: int(d[-1])), ("first_digit", lambda d: int(d[0]))):
                    y = torch.tensor([fn(d) if d else 0 for d in dstr], device=dev); tr_bd = [x[okd[x]] for x in train_b]; va_i = val_idx[okd[val_idx]]; te_i = test_idx[okd[test_idx]]
                    D = TT.setdefault(nm, {}); avail_n = sum(len(x) for x in tr_bd)
                    for N in sizes:
                        if N > avail_n * 1.05 and sizes.index(N) > 0 and sizes[sizes.index(N) - 1] >= avail_n: continue
                        quota = [0] * 6; left = min(N, avail_n); avail = [len(x) for x in tr_bd]; order = sorted(range(6), key=lambda b: avail[b])
                        for i, b in enumerate(order): q = min(avail[b], left // (6 - i)); quota[b] = q; left -= q
                        tr_i = np.concatenate([tr_bd[b][: quota[b]] for b in range(6)])
                        for cap in [c for c in caps if c != "tf"] + (["tf"] if "tf" in caps else []):
                            C = D.setdefault(str(N), {})
                            if cap not in C: C[cap] = run_cls(cap, tr_i, va_i, te_i, y, 10, tag=nm); save()
        del H; torch.cuda.empty_cache()
    save(); print(f"[scale-train] wrote {a.out} ({(time.time() - t0) / 60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
