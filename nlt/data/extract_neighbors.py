"""NEIGHBOUR-POSITION residuals for hard negatives (rl #544 / redteam #545): for every stored position (doc_id, pos) in a pos_idx list, re-forward the
stored document and gather the residual stream h_k (k = K_LO..K_HI, HF hidden_states[k+1]) at pos + off for off in OFFSETS -> a store dir in the
SAME layout as the main store (acts_<tag>.npy [n, 26, 4096] fp16 + meta_<tag>.parquet with pos_idx, doc_id, pos, token_id, next_token_id, source
PLUS nbr_of = the original pos_idx and offset), so nlt.data.dataset.ActStore(out_dir, split) loads it and rl can gather (h_i, h_j) of the neighbour
with the same (i, j). New pos_idx = 7_000_000_000 + 10 * original + (offset + 5)  (unique, decodable).

  python -m nlt.data.extract_neighbors --data-dir /vol/data/qwen3_8b --split train --pos-idx-file /vol/feat/train20k_pos_idx.txt --out-dir /vol/data/qwen3_8b_nbr [--offsets -2,-1,1,2]
"""
import argparse, glob, json, os, time
import numpy as np, torch

K_LO, K_HI = 9, 34
N_LAYERS = K_HI - K_LO + 1


class _Stop(Exception):
    pass


def build(base, device):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, attn_implementation="sdpa", token=os.environ.get("HF_TOKEN")).to(device).eval()
    inner = model.model if hasattr(model, "model") else model
    layers = inner.layers if hasattr(inner, "layers") else inner.language_model.layers
    del layers[K_HI + 1:]
    try:
        model.config.num_hidden_layers = K_HI + 1
        if hasattr(model.config, "layer_types"): model.config.layer_types = model.config.layer_types[: K_HI + 1]
    except Exception:
        pass
    state = {"sel": None, "out": None}

    def hook(k):
        def f(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            b, p = state["sel"]; state["out"][k - K_LO] = h[b, p].to(torch.float16)
            if k == K_HI: raise _Stop()
        return f
    for k in range(K_LO, K_HI + 1): layers[k].register_forward_hook(hook(k))
    return model, state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--split", default="train"); p.add_argument("--out-dir", required=True); p.add_argument("--base", default="Qwen/Qwen3-8B")
    p.add_argument("--pos-idx-file", default=""); p.add_argument("--offsets", default="-2,-1,1,2"); p.add_argument("--docs-per-batch", type=int, default=8); p.add_argument("--shard-size", type=int, default=8192)
    a = p.parse_args(); offsets = [int(x) for x in a.offsets.split(",")]
    import pyarrow.parquet as pq, pyarrow as pa
    dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = True
    model, state = build(a.base, dev); d = model.config.hidden_size
    out_dir = os.path.join(a.out_dir, a.split); os.makedirs(out_dir, exist_ok=True)
    keep = None
    if a.pos_idx_file:
        txt = open(a.pos_idx_file).read().strip(); keep = set(int(x) for x in (json.loads(txt) if txt.startswith("[") else txt.split()))
    metas = sorted(glob.glob(os.path.join(a.data_dir, a.split, "meta_*.parquet"))); t0 = time.time(); n_tot = 0; shard = 0
    buf_acts, buf_meta = [], []

    def flush(force=False):
        nonlocal shard, buf_acts, buf_meta
        if not buf_meta or (not force and sum(x.shape[0] for x in buf_acts) < a.shard_size): return
        A = np.concatenate(buf_acts, 0); tag = f"nbr_{shard:04d}"
        np.save(os.path.join(out_dir, f"acts_{tag}.tmp.npy"), A); os.replace(os.path.join(out_dir, f"acts_{tag}.tmp.npy"), os.path.join(out_dir, f"acts_{tag}.npy"))
        pq.write_table(pa.Table.from_pylist(buf_meta), os.path.join(out_dir, f"meta_{tag}.parquet")); shard += 1; buf_acts, buf_meta = [], []
    for mpath in metas:
        meta = pq.read_table(mpath).to_pandas(); docs = pq.read_table(mpath.replace("meta_", "docs_")).to_pandas().set_index("doc_id")
        if keep is not None: meta = meta[meta["pos_idx"].isin(keep)]
        if len(meta) == 0: continue
        groups = list(meta.groupby("doc_id"))
        for g0 in range(0, len(groups), a.docs_per_batch):
            batch = groups[g0:g0 + a.docs_per_batch]
            seqs = [torch.tensor(docs.loc[doc_id, "token_ids"], dtype=torch.long) for doc_id, _ in batch]
            L = max(len(s) for s in seqs); ids = torch.zeros(len(seqs), L, dtype=torch.long); att = torch.zeros(len(seqs), L, dtype=torch.long)
            for bi, s in enumerate(seqs): ids[bi, :len(s)] = s; att[bi, :len(s)] = 1
            bsel, psel, rows_meta = [], [], []
            for bi, (doc_id, g) in enumerate(batch):
                tid = docs.loc[doc_id, "token_ids"]; n = len(tid); src = g["source"].iloc[0]
                for pi, pos in zip(g["pos_idx"].values, g["pos"].values):
                    for off in offsets:
                        q = int(pos) + off
                        if q < 4 or q + 1 >= n: continue                     # same rules as the main store: pos >= 4 and a next token must exist
                        bsel.append(bi); psel.append(q)
                        rows_meta.append({"pos_idx": 7_000_000_000 + 10 * int(pi) + (off + 5), "doc_id": int(doc_id), "pos": q, "token_id": int(tid[q]), "next_token_id": int(tid[q + 1]), "source": src, "nbr_of": int(pi), "offset": off})
            if not bsel: continue
            state["sel"] = (torch.tensor(bsel, device=dev), torch.tensor(psel, device=dev)); state["out"] = torch.zeros(N_LAYERS, len(bsel), d, dtype=torch.float16, device=dev)
            with torch.no_grad():
                try: model(input_ids=ids.to(dev), attention_mask=att.to(dev))
                except _Stop: pass
            buf_acts.append(state["out"].permute(1, 0, 2).cpu().numpy()); buf_meta += rows_meta; n_tot += len(bsel); flush()
        print(f"[nbr:{a.split}] {os.path.basename(mpath)}: {n_tot} neighbour positions so far, {time.time() - t0:.0f}s", flush=True)
    flush(force=True)
    json.dump({"offsets": offsets, "n_positions": n_tot, "pos_idx_rule": "7e9 + 10*orig + (offset+5)", "layout": "same as the main store (acts_<tag>.npy [n,26,4096] fp16 + meta_<tag>.parquet); extra meta cols nbr_of, offset"}, open(os.path.join(out_dir, "nbr_info.json"), "w"), indent=1)
    print(f"[nbr:{a.split}] DONE {n_tot} positions -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
