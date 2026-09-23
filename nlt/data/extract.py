"""Multi-layer activation producer for the NLT: Qwen3-8B (bf16) over the corpus mix, storing the residual stream after blocks
k = K_LO..K_HI (HF hidden_states[k+1]) at a few sampled token positions per document.

Shard layout (one producer writes acts_<idx>_<n>.npy + meta_<idx>_<n>.parquet + docs_<idx>_<n>.parquet per shard):
  acts  fp16 [n_pos, n_layers, d]   axis 1 = layers K_LO..K_HI in order (index k - K_LO); all layers of one position are contiguous
  meta  cols  pos_idx (int64, unique across producers = producer * 10^9 + local counter), doc_id, pos, token_id, next_token_id, source
  docs  cols  doc_id, source, text (the truncated document that was actually run), token_ids
Positions < MIN_POS are never sampled (attention-sink / massive-activation tokens); the last position is never sampled (needs a next token).
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch

K_LO, K_HI = 9, 34
N_LAYERS = K_HI - K_LO + 1      # 26
MIN_POS = 4


class _Stop(Exception):
    pass


def build_model(base, device, attn_impl="sdpa"):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, attn_implementation=attn_impl).to(device).eval()
    inner = model.model
    layers = inner.layers if hasattr(inner, "layers") else inner.language_model.layers
    del layers[K_HI + 1:]                                   # nothing after block K_HI is needed
    try:
        model.config.num_hidden_layers = K_HI + 1
        if hasattr(model.config, "layer_types"): model.config.layer_types = model.config.layer_types[: K_HI + 1]
    except Exception:
        pass
    state = {"sel": None, "out": None}                      # sel = (batch_idx, pos_idx) LongTensors; out = [N_LAYERS, n, d] fp16

    def make_hook(k):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            b, p = state["sel"]
            state["out"][k - K_LO] = h[b, p].to(torch.float16)
            if k == K_HI: raise _Stop()
        return hook
    for k in range(K_LO, K_HI + 1):
        layers[k].register_forward_hook(make_hook(k))
    return model, state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen3-8B"); p.add_argument("--out-dir", required=True, help="volume dir: <out>/train and <out>/val")
    p.add_argument("--n-producers", type=int, default=1); p.add_argument("--index", type=int, default=0)
    p.add_argument("--target-train", type=int, default=80000, help="train positions THIS producer stops after")
    p.add_argument("--target-val", type=int, default=2500, help="val positions this producer stops after (val docs are hash-selected)")
    p.add_argument("--pos-per-doc", type=int, default=8); p.add_argument("--max-len", type=int, default=1024); p.add_argument("--min-len", type=int, default=24)
    p.add_argument("--docs-per-batch", type=int, default=32); p.add_argument("--shard-size", type=int, default=8192)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--max-minutes", type=float, default=170.0)
    a = p.parse_args()
    device = "cuda"; torch.backends.cuda.matmul.allow_tf32 = True
    from transformers import AutoTokenizer
    from nlt.data.sources import doc_stream, is_val_doc
    tok = AutoTokenizer.from_pretrained(a.base)
    hf_token = os.environ.get("HF_TOKEN")
    model, state = build_model(a.base, device, a.attn_impl)
    d = model.config.hidden_size
    print(f"[prod{a.index}] model ready: {K_HI + 1} blocks kept, d={d}, layers stored {K_LO}..{K_HI}", flush=True)
    for split in ("train", "val"): os.makedirs(os.path.join(a.out_dir, split), exist_ok=True)
    rng = np.random.default_rng(a.seed * 1009 + a.index)

    class Split:
        def __init__(self, name, target):
            self.name, self.target, self.n_total, self.n_shards = name, target, 0, 0
            self.acts, self.meta, self.docs = [], [], []
        def add(self, acts, meta, doc):
            self.acts.append(acts); self.meta += meta; self.docs.append(doc); self.n_total += len(meta)
        def n_buf(self): return sum(x.shape[0] for x in self.acts)
        def flush(self, force=False):
            if not self.acts or (not force and self.n_buf() < a.shard_size): return
            import pyarrow as pa, pyarrow.parquet as pq
            A = np.concatenate(self.acts, 0); tag = f"{a.index:02d}_{self.n_shards:04d}"
            out = os.path.join(a.out_dir, self.name)
            np.save(os.path.join(out, f"acts_{tag}.tmp.npy"), A); os.replace(os.path.join(out, f"acts_{tag}.tmp.npy"), os.path.join(out, f"acts_{tag}.npy"))
            pq.write_table(pa.Table.from_pylist(self.meta), os.path.join(out, f"meta_{tag}.parquet"))
            pq.write_table(pa.Table.from_pylist(self.docs), os.path.join(out, f"docs_{tag}.parquet"))
            print(f"[prod{a.index}] wrote {self.name} shard {tag}: {A.shape[0]} positions ({self.n_total} total, {len(self.docs)} docs)", flush=True)
            self.acts, self.meta, self.docs = [], [], []; self.n_shards += 1
        def done(self): return self.n_total >= self.target

    splits = {"train": Split("train", a.target_train), "val": Split("val", a.target_val)}
    pos_counter = 0; doc_counter = 0; t0 = time.time(); n_tok = 0; n_fwd = 0
    src_counts = {}
    stream = doc_stream(a.n_producers, a.index, a.seed, hf_token, tok)

    def run_batch(batch):
        """batch: list of (split, source, text, ids list). One padded forward; hooks gather the sampled positions."""
        nonlocal pos_counter, doc_counter, n_tok, n_fwd
        L = max(len(b[3]) for b in batch); B = len(batch)
        ids = torch.full((B, L), tok.pad_token_id or 0, dtype=torch.long); am = torch.zeros((B, L), dtype=torch.long)
        sel_b, sel_p, owners = [], [], []
        for bi, (split, src, text, t_ids) in enumerate(batch):
            n = len(t_ids); ids[bi, :n] = torch.tensor(t_ids); am[bi, :n] = 1
            cand = np.arange(MIN_POS, n - 1)                         # need a next token -> pos <= n-2
            k = min(a.pos_per_doc, len(cand)); ps = np.sort(rng.choice(cand, k, replace=False))
            sel_b += [bi] * k; sel_p += ps.tolist(); owners.append((bi, k))
        state["sel"] = (torch.tensor(sel_b, device=device), torch.tensor(sel_p, device=device))
        state["out"] = torch.empty((N_LAYERS, len(sel_b), d), dtype=torch.float16, device=device)
        with torch.no_grad():
            try: model(input_ids=ids.to(device), attention_mask=am.to(device), use_cache=False)
            except _Stop: pass
        acts = state["out"].permute(1, 0, 2).contiguous().cpu().numpy()          # [n_sel, N_LAYERS, d]
        n_tok += int(am.sum()); n_fwd += 1
        off = 0
        for (bi, k), (split, src, text, t_ids) in zip(owners, batch):
            doc_id = a.index * 1_000_000_000 + doc_counter; doc_counter += 1
            meta = []
            for q in range(k):
                pos = sel_p[off + q]
                meta.append({"pos_idx": a.index * 1_000_000_000 + pos_counter, "doc_id": doc_id, "pos": int(pos), "token_id": int(t_ids[pos]),
                             "next_token_id": int(t_ids[pos + 1]), "source": src}); pos_counter += 1
            splits[split].add(acts[off:off + k], meta, {"doc_id": doc_id, "source": src, "text": text, "token_ids": [int(x) for x in t_ids]}); off += k
            src_counts[src] = src_counts.get(src, 0) + 1
        for s in splits.values(): s.flush()

    batch = []
    for src, text in stream:
        if all(s.done() for s in splits.values()) or (time.time() - t0) / 60 > a.max_minutes: break
        split = "val" if is_val_doc(text) else "train"
        if splits[split].done(): continue
        t_ids = tok(text, add_special_tokens=False, truncation=True, max_length=a.max_len)["input_ids"]
        if len(t_ids) < a.min_len: continue
        text_run = tok.decode(t_ids) if len(t_ids) == a.max_len else text
        batch.append((split, src, text_run, t_ids))
        if len(batch) >= a.docs_per_batch:
            run_batch(batch); batch = []
            if n_fwd % 20 == 0:
                dt = time.time() - t0
                print(f"[prod{a.index}] {n_fwd} fwd, {n_tok/1e6:.2f}M tok, {n_tok/dt/1e3:.1f}k tok/s, train {splits['train'].n_total} val {splits['val'].n_total} positions, "
                      f"{dt/60:.1f} min, sources {json.dumps(src_counts)}", flush=True)
    if batch: run_batch(batch)
    for s in splits.values(): s.flush(force=True)
    dt = time.time() - t0
    prog = {"index": a.index, "train": splits["train"].n_total, "val": splits["val"].n_total, "docs": doc_counter, "tokens": n_tok, "seconds": dt,
            "tok_per_s": n_tok / max(dt, 1e-6), "sources": src_counts, "args": vars(a)}
    json.dump(prog, open(os.path.join(a.out_dir, f"progress_{a.index:02d}.json"), "w"), indent=1)
    print(f"[prod{a.index}] DONE {json.dumps({k: v for k, v in prog.items() if k != 'args'})}", flush=True)


if __name__ == "__main__":
    main()
