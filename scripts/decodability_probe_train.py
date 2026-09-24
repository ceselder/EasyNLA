"""Decodability test 2, probes (1 GPU) — can a probe read a specific detail of the context out of the layer-42 activation, and up to what distance?

Input: scripts/decodability_probe_build.py output (fp16 activations + rows: type, value, matched other-document negative, near / far number
edits, token distance k and bucket, doc split). Tasks, per detail type (number / name / quote), trained POOLED over distances on the train
documents, evaluated per distance bucket on held-out documents:
  2-AFC "which of these two values is in the context?"  score(h, v) for the true value vs a matched value of the same type from another
  document — a bilinear ("linear in h for a fixed value") probe and a small MLP on [h ; features(v)]. features(v) = mean / first / last
  Qwen3.6-27B input-embedding of the value's tokens + hashed character n-grams + digit-position one-hots (numbers).
  Controls: the same probes trained and tested with the activations permuted (value-frequency leakage floor); a probe trained on the
  shuffled pairing gives the number to subtract.
  Numbers: the other-document probe evaluated on true-vs-near-miss and true-vs-far pairs, and a probe TRAINED on near-miss negatives.
  h-only decoding of the number's last digit (10-way) and first digit (10-way) by multinomial logistic regression, per bucket.
Output: JSON with accuracy, n and Wilson CI per (type, probe, negative kind, bucket).
usage: python scripts/decodability_probe_train.py --data /vol_glp/decodability/probe_data.pt --out /vol_glp/decodability/probe_results.json
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, re, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
NHASH = 4096


def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load_embeddings(base, dev):
    """Qwen3.6-27B input embedding matrix only (index json -> the one shard that holds it)."""
    from huggingface_hub import snapshot_download, hf_hub_download
    from safetensors import safe_open
    snap = snapshot_download(base, token=os.environ.get("HF_TOKEN"), local_dir="/root/base_snap", allow_patterns=["*.json", "tokenizer*", "*.jinja", "*.txt"])
    idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
    key = [k for k in idx if k.endswith("embed_tokens.weight")][0]; fn = idx[key]
    path = hf_hub_download(base, fn, token=os.environ.get("HF_TOKEN"), local_dir="/root/base_snap")
    with safe_open(path, framework="pt", device="cpu") as f: E = f.get_tensor(key).float()
    print(f"[probe] embeddings {key} {tuple(E.shape)} from {fn}", flush=True)
    return E.to(dev), snap


class ValueEncoder:
    def __init__(self, tok, E, dev):
        self.tok, self.E, self.dev = tok, E, dev; self.cache = {}
    def num_extras(self, v):
        x = np.zeros(131, dtype=np.float32); digits = re.sub(r"[^\d]", "", v.split(".")[0] if "." in v else v)
        if digits:
            for i, ch in enumerate(digits[:6]): x[i * 10 + int(ch)] = 1
            for i, ch in enumerate(digits[::-1][:6]): x[60 + i * 10 + int(ch)] = 1
            x[120 + min(len(digits), 8) - 1] = 1; x[128] = math.log10(max(int(digits), 1)) / 10
        x[129] = float("," in v); x[130] = float("." in v)
        return x
    def hashed(self, v):
        x = np.zeros(NHASH, dtype=np.float32); s = f"^{v.lower()}$"
        for n in (2, 3):
            for i in range(len(s) - n + 1): x[int(hashlib.md5(s[i: i + n].encode()).hexdigest()[:8], 16) % NHASH] += 1
        return x / (np.linalg.norm(x) + 1e-6)
    def __call__(self, values):
        out = []
        for v in values:
            if v not in self.cache:
                ids = self.tok.encode(" " + v, add_special_tokens=False) or [0]; e = self.E[torch.tensor(ids, device=self.dev)]
                self.cache[v] = torch.cat([e.mean(0), e[0], e[-1], torch.tensor(self.num_extras(v), device=self.dev), torch.tensor(self.hashed(v), device=self.dev)])
            out.append(self.cache[v])
        return torch.stack(out)


class Bilinear(nn.Module):
    def __init__(self, dh, dv, r=256):
        super().__init__(); self.A = nn.Linear(dh, r, bias=False); self.B = nn.Linear(dv, r, bias=False); self.wv = nn.Linear(dv, 1); self.r = r
    def forward(self, h, v): return (self.A(h) * self.B(v)).sum(-1) / math.sqrt(self.r) + self.wv(v).squeeze(-1)


class MLP(nn.Module):
    def __init__(self, dh, dv, w=1024):
        super().__init__(); self.net = nn.Sequential(nn.Linear(dh + dv, w), nn.GELU(), nn.Dropout(0.1), nn.Linear(w, 256), nn.GELU(), nn.Linear(256, 1))
    def forward(self, h, v): return self.net(torch.cat([h, v], -1)).squeeze(-1)


def train_pair_probe(kind, H, Vp, Vn, tr, va, dev, epochs=40, seed=0):
    """2-AFC pairwise logistic loss: softplus(s(h, v_neg) − s(h, v_pos)); early stopping on the val docs."""
    torch.manual_seed(seed); dh, dv = H.shape[1], Vp.shape[1]
    m = (Bilinear(dh, dv) if kind == "bilinear" else MLP(dh, dv)).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=1e-3 if kind == "bilinear" else 3e-4, weight_decay=1e-2)
    best = (-1, None); bs = 1024; tr = torch.tensor(tr, device=dev); va = torch.tensor(va, device=dev)
    for ep in range(epochs):
        m.train(); perm = tr[torch.randperm(len(tr), device=dev)]
        for s in range(0, len(perm), bs):
            ix = perm[s: s + bs]; loss = F.softplus(m(H[ix], Vn[ix]) - m(H[ix], Vp[ix])).mean(); opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): acc = float(((m(H[va], Vp[va]) > m(H[va], Vn[va])).float().mean()))
        if acc > best[0]: best = (acc, {k: v.detach().clone() for k, v in m.state_dict().items()})
    m.load_state_dict(best[1]); m.eval(); return m, best[0]


@torch.no_grad()
def eval_pairs(m, H, Vp, Vn, idx, rows_, buckets):
    """-> {bucket: (acc, n, ci)} over the rows idx (pairs with a valid negative), plus 'all'."""
    out = {}; idx = np.asarray(idx)
    if len(idx) == 0: return out
    ix = torch.tensor(idx, device=H.device); win = (m(H[ix], Vp[ix]) > m(H[ix], Vn[ix])).cpu().numpy()
    bk = np.array([rows_[i]["bucket"] for i in idx])
    for b in buckets + ["all"]:
        sel = win if b == "all" else win[bk == b]
        if len(sel): out[b] = {"acc": float(sel.mean()), "n": int(len(sel)), "ci": wilson(int(sel.sum()), len(sel))}
    return out


def logreg_multiclass(Htr, ytr, Hva, yva, Hte, dev, ncls, epochs=60, seed=0):
    """multinomial logistic regression on standardised h; the epoch is selected on the val docs (5120-d x 10 classes overfits 26k rows otherwise)."""
    torch.manual_seed(seed); m = nn.Linear(Htr.shape[1], ncls).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2); bs = 1024; best = (-1, None)
    for ep in range(epochs):
        perm = torch.randperm(len(Htr), device=dev)
        for s in range(0, len(perm), bs):
            ix = perm[s: s + bs]; loss = F.cross_entropy(m(Htr[ix]), ytr[ix]); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): acc = float((m(Hva).argmax(-1) == yva).float().mean())
        if acc > best[0]: best = (acc, {k: v.detach().clone() for k, v in m.state_dict().items()})
    m.load_state_dict(best[1])
    with torch.no_grad(): return m(Hte).argmax(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--out", required=True); p.add_argument("--base", default="Qwen/Qwen3.6-27B"); p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--types", default="number,name,quote"); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(); dev = "cuda:0"; t0 = time.time()
    D = torch.load(a.data, weights_only=False); rows = D["rows"]; buckets = D["buckets"]; H_all = D["h"].float()
    print(f"[probe] {len(rows)} rows; counts {json.dumps(D['counts'])}", flush=True)
    from transformers import AutoTokenizer
    E, snap = load_embeddings(a.base, dev); tok = AutoTokenizer.from_pretrained(snap); venc = ValueEncoder(tok, E, dev)
    res = {"counts": D["counts"], "n_scanned_rows": D.get("n_scanned_rows"), "ctx_len_pcts": D.get("ctx_len_pcts"), "buckets": buckets, "types": {}}
    rs = np.random.default_rng(a.seed)
    for typ in a.types.split(","):
        idx = [i for i, r in enumerate(rows) if r["type"] == typ]
        if not idx: continue
        R = [rows[i] for i in idx]; H = H_all[idx].to(dev)
        tr_all = [i for i, r in enumerate(R) if r["split"] == "train"]; te = [i for i, r in enumerate(R) if r["split"] == "test"]
        docs = sorted({R[i]["doc"] for i in tr_all}); vdocs = set(rs.choice(docs, max(1, len(docs) // 10), replace=False).tolist())
        va = [i for i in tr_all if R[i]["doc"] in vdocs]; tr = [i for i in tr_all if R[i]["doc"] not in vdocs]
        mu, sd = H[tr].mean(0, keepdim=True), H[tr].std(0, keepdim=True) + 1e-3; Hs = (H - mu) / sd
        Vp = venc([r["value"] for r in R]); Vn = venc([r["neg"] for r in R])
        vmu, vsd = Vp[tr].mean(0, keepdim=True), Vp[tr].std(0, keepdim=True) + 1e-3; Vp = (Vp - vmu) / vsd; Vn = (Vn - vmu) / vsd
        T = res["types"][typ] = {"n_train": len(tr), "n_val": len(va), "n_test": len(te), "probes": {}}
        print(f"[probe] {typ}: train {len(tr)} val {len(va)} test {len(te)} ({time.time() - t0:.0f}s)", flush=True)
        # shuffled-activation control: permute h across rows (train and test separately, so no pairing survives)
        perm = torch.tensor(np.concatenate([rs.permutation(tr), rs.permutation(va), rs.permutation(te)]), device=dev); Hshuf = Hs.clone(); Hshuf[torch.tensor(tr + va + te, device=dev)] = Hs[perm]
        for kind in ("bilinear", "mlp"):
            m, vacc = train_pair_probe(kind, Hs, Vp, Vn, tr, va, dev, epochs=a.epochs, seed=a.seed); P = T["probes"][kind] = {"val_acc": vacc}
            P["other_doc"] = eval_pairs(m, Hs, Vp, Vn, te, R, buckets)
            ms, _ = train_pair_probe(kind, Hshuf, Vp, Vn, tr, va, dev, epochs=a.epochs, seed=a.seed); P["other_doc_shuffled_h"] = eval_pairs(ms, Hshuf, Vp, Vn, te, R, buckets)
            if typ == "number":
                for nk in ("near", "far"):
                    ok = [i for i in te if R[i].get(nk)]; Vk = (venc([R[i][nk] for i in ok]) - vmu) / vsd
                    Vtmp = Vn.clone(); Vtmp[torch.tensor(ok, device=dev)] = Vk; P[nk] = eval_pairs(m, Hs, Vp, Vtmp, ok, R, buckets)
                # a probe TRAINED on near-miss negatives
                okn = [i for i in range(len(R)) if R[i].get("near")]; Vnear = Vn.clone(); Vnear[torch.tensor(okn, device=dev)] = (venc([R[i]["near"] for i in okn]) - vmu) / vsd
                trn = [i for i in tr if R[i].get("near")]; van = [i for i in va if R[i].get("near")]; ten = [i for i in te if R[i].get("near")]
                mn, vaccn = train_pair_probe(kind, Hs, Vp, Vnear, trn, van, dev, epochs=a.epochs, seed=a.seed)
                P["near_trained"] = {"val_acc": vaccn, "near": eval_pairs(mn, Hs, Vp, Vnear, ten, R, buckets), "other_doc": eval_pairs(mn, Hs, Vp, Vn, te, R, buckets)}
                okf = [i for i in te if R[i].get("far")]; Vf = Vn.clone(); Vf[torch.tensor(okf, device=dev)] = (venc([R[i]["far"] for i in okf]) - vmu) / vsd; P["near_trained"]["far"] = eval_pairs(mn, Hs, Vp, Vf, okf, R, buckets)
                msn, _ = train_pair_probe(kind, Hshuf, Vp, Vnear, trn, van, dev, epochs=a.epochs, seed=a.seed); P["near_trained"]["near_shuffled_h"] = eval_pairs(msn, Hshuf, Vp, Vnear, ten, R, buckets)
            print(f"[probe] {typ}/{kind}: val {vacc:.3f}; test other-doc " + " ".join(f"{b}:{v['acc']:.3f}(n{v['n']})" for b, v in P["other_doc"].items()) +
                  " | shuffled-h " + " ".join(f"{b}:{v['acc']:.3f}" for b, v in P["other_doc_shuffled_h"].items()) +
                  ("" if typ != "number" else " | near " + " ".join(f"{b}:{v['acc']:.3f}" for b, v in P["near"].items()) + " | near-trained/near " + " ".join(f"{b}:{v['acc']:.3f}" for b, v in P["near_trained"]["near"].items())), flush=True)
        if typ == "number":   # h-only: last / first digit of the number, multinomial logistic regression on standardised h
            def digits(v): return re.sub(r"[^\d]", "", v.split(".")[0] if "." in v else v)
            for nm, fn, ncls in (("last_digit", lambda v: int(digits(v)[-1]), 10), ("first_digit", lambda v: int(digits(v)[0]), 10), ("n_digits", lambda v: min(len(digits(v)), 8) - 1, 8)):
                ok_tr = [i for i in tr if digits(R[i]["value"])]; ok_va = [i for i in va if digits(R[i]["value"])]; ok_te = [i for i in te if digits(R[i]["value"])]
                ytr = torch.tensor([fn(R[i]["value"]) for i in ok_tr], device=dev); yva = torch.tensor([fn(R[i]["value"]) for i in ok_va], device=dev); yte = np.array([fn(R[i]["value"]) for i in ok_te])
                pred = logreg_multiclass(Hs[torch.tensor(ok_tr, device=dev)], ytr, Hs[torch.tensor(ok_va, device=dev)], yva, Hs[torch.tensor(ok_te, device=dev)], dev, ncls).cpu().numpy()
                maj = np.bincount(ytr.cpu().numpy(), minlength=ncls).argmax(); bk = np.array([R[i]["bucket"] for i in ok_te]); out = {}
                for b in buckets + ["all"]:
                    sel = np.ones(len(ok_te), bool) if b == "all" else (bk == b)
                    if sel.sum(): out[b] = {"acc": float((pred[sel] == yte[sel]).mean()), "n": int(sel.sum()), "ci": wilson(int((pred[sel] == yte[sel]).sum()), int(sel.sum())), "majority": float((yte[sel] == maj).mean())}
                T["probes"][f"h_only_{nm}"] = out
                print(f"[probe] number/h-only {nm}: " + " ".join(f"{b}:{v['acc']:.3f}(maj {v['majority']:.2f},n{v['n']})" for b, v in out.items()), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    json.dump(res, open(a.out, "w"), indent=1); print(f"[probe] wrote {a.out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
