"""Text pools for the diffusion-prior critic: several sources mixed by weight so the model does not learn one register.

  pools = TextPools("lens=0.4:/vol/z/lensdiff_v1/train/L*_part*.parquet,teacher=0.3:/vol/z/teacher-sonnet-v1/train/*.parquet,...",
                    pairs_parquet, store)              # spec: name=weight:glob[;glob...][@verb]
  rows, i, j, texts, names = pools.sample(B, gen)        # store rows + layer indices + the texts (one pool per row, drawn by weight)
Val sets (proxy monitoring + spot exact bits) are label:path[@verb] items joined to pairs_val, restricted to rows AFTER the fixed 4096-row
eval set so model selection never sees it.
"""
from __future__ import annotations
import glob as _glob, os
import numpy as np, torch
from nlt.critic.train import load_text_pairs


def _split_verb(path):
    if "@" in os.path.basename(path): p, v = path.rsplit("@", 1); return p, [int(v)]
    return path, None


def depth_tag_texts(i, j):
    """SMOKE-GATE synthetic text (DECISIONS v1.8 T1): trivially informative, forbidden as a verbalizer target"""
    return [f"from layer {int(a)} to layer {int(b)}" for a, b in zip(i.tolist(), j.tolist())]


class TextPools:
    def __init__(self, spec: str, pairs_parquet: str, store, verbose=True):
        self.names, self.weights, self.pools = [], [], []
        for item in [s for s in spec.split(",") if s.strip()]:
            name, rest = item.split("=", 1); w, paths = rest.split(":", 1)
            globs, verb = [], None
            for p in paths.split(";"):
                p, v = _split_verb(p); verb = v if v is not None else verb; globs.append(p)
            df = load_text_pairs(globs, pairs_parquet, verb)
            df = df[df["pos_idx"].isin(store.row_of)].reset_index(drop=True)
            if len(df) == 0: print(f"[pools] WARNING pool {name} has no rows ({globs})", flush=True); continue
            self.names.append(name); self.weights.append(float(w))
            self.pools.append({"rows": store.rows_for(df["pos_idx"].values), "i": torch.tensor(df["i"].values.astype(np.int64)), "j": torch.tensor(df["j"].values.astype(np.int64)),
                               "text": df["text"].astype(str).tolist(), "pair_id": df["pair_id"].tolist()})
            if verbose:
                nw = np.mean([len(t.split()) for t in self.pools[-1]["text"][:5000]])
                print(f"[pools] {name}: {len(df)} rows ({df['pair_id'].nunique()} pairs), weight {w}, ~{nw:.0f} words; e.g. {self.pools[-1]['text'][0][:120]!r}", flush=True)
        w = np.asarray(self.weights, dtype=np.float64); self.p = w / w.sum()
        self.n_rows = sum(len(p["text"]) for p in self.pools)

    def sample(self, B, gen=None, mode="pool"):
        """mode 'pool': ONE pool per step (drawn by weight) so a batch pads to its own register's length (bullets ~200 tokens would otherwise
        pad every mixed batch); 'mixed': rows drawn per pool by weight within the batch."""
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,), generator=gen)))
        counts = rng.multinomial(B, self.p) if mode == "mixed" else np.bincount([rng.choice(len(self.p), p=self.p)], minlength=len(self.p)) * B
        rows, ii, jj, texts, names = [], [], [], [], []
        for c, name, pool in zip(counts, self.names, self.pools):
            if c == 0: continue
            idx = torch.randint(0, len(pool["text"]), (int(c),), generator=gen)
            rows.append(pool["rows"][idx]); ii.append(pool["i"][idx]); jj.append(pool["j"][idx]); texts += [pool["text"][k] for k in idx.tolist()]; names += [name] * int(c)
        perm = torch.randperm(B, generator=gen)
        rows = torch.cat(rows)[perm]; ii = torch.cat(ii)[perm]; jj = torch.cat(jj)[perm]
        texts = [texts[k] for k in perm.tolist()]; names = [names[k] for k in perm.tolist()]
        return rows, ii, jj, texts, names


def load_val_sets(spec: str, pairs_val_parquet: str, store_val, offset: int, n: int, verbose=True):
    """label:path[@verb],... -> {label: (rows, i, j, texts, pair_ids)} on held-out pairs_val rows [offset:], one text per pair, up to n rows"""
    import pyarrow.parquet as pq
    vp = pq.read_table(pairs_val_parquet, columns=["pair_id", "pos_idx", "i", "j"]).to_pandas()
    vp = vp[vp["pos_idx"].isin(store_val.row_of)].iloc[offset:]
    out = {}
    for item in [s for s in spec.split(",") if s.strip()]:
        label, path = item.split(":", 1); path, verb = _split_verb(path)
        df = load_text_pairs(path.split(";"), pairs_val_parquet, verb).sample(frac=1.0, random_state=0).drop_duplicates("pair_id").set_index("pair_id")
        sub = vp[vp["pair_id"].isin(df.index)].iloc[:n]
        if len(sub) == 0: print(f"[val] WARNING set {label} has no held-out rows", flush=True); continue
        out[label] = (store_val.rows_for(sub["pos_idx"].values), torch.tensor(sub["i"].values.astype(np.int64)), torch.tensor(sub["j"].values.astype(np.int64)),
                      [df.loc[p, "text"] for p in sub["pair_id"]], sub["pair_id"].tolist())
        if verbose: print(f"[val] {label}: {len(sub)} held-out rows (pairs_val[{offset}:]) from {path}", flush=True)
    return out


def dm_partner(i, j, seed=0):
    """depth-matched wrong-text partner within a set: another row with the same (i, j), else the same j, else any other row"""
    i = [int(x) for x in i]; j = [int(x) for x in j]; n = len(i); by_ij, by_j = {}, {}
    for q in range(n): by_ij.setdefault((i[q], j[q]), []).append(q); by_j.setdefault(j[q], []).append(q)
    out = []
    for q in range(n):
        c = [r for r in by_ij[(i[q], j[q])] if r != q] or [r for r in by_j[j[q]] if r != q] or [r for r in range(n) if r != q]
        out.append(c[(q + 1) % len(c)] if c else q)
    return out
