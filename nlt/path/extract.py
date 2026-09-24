"""Re-forward the stored documents through Qwen3-8B and store, at the stored positions, the attention-block output a_k and the MLP output m_k
for k = 10..34 (so that h_j = h_i + sum_{k=i+1..j} (a_k + m_k)). Also checks (a) per-layer h_{k-1} + a_k + m_k == h_k, (b) the re-forwarded
residuals against the STORED acts (cos), (c) the full-path sum from h_9 to h_34.

  python -m nlt.path.extract --data-dir /vol/data/qwen3_8b --split train --out-dir /vol/path/qwen3_8b \
      --rows /vol/z/v0b_mix/train/rows.parquet [--pairs-head 4096] [--pairs-slice 0:20000] --tag v0b

Output: <out-dir>/<split>/path_<tag>_<shard>.npy fp16 [n, 25, 2, d] + pathmeta_<tag>_<shard>.parquet (pos_idx, doc_id, pos) + check_<tag>.json.
"""
from __future__ import annotations
import argparse, glob, json, os, time
import numpy as np
import torch
from nlt.data.extract import K_LO, K_HI, N_LAYERS
from nlt.data.finalize import meta_of

W_LO = K_LO + 1; N_W = K_HI - W_LO + 1


class _Stop(Exception):
    pass


def wanted_positions(a, data_dir, split):
    import pyarrow.parquet as pq, pandas as pd
    pairs = pq.read_table(os.path.join(data_dir, f"pairs_{split}.parquet"), columns=["pair_id", "pos_idx"]).to_pandas()
    sp = os.path.join(data_dir, "spikes.json"); bad = set(json.load(open(sp)).get(split, [])) if os.path.exists(sp) else set()
    pos = set()
    if a.rows:
        for pat in a.rows.split(","):
            for f in (sorted(glob.glob(pat)) or [pat]):
                pid = pq.read_table(f, columns=["pair_id"]).column(0).to_pylist()
                sub = pairs[pairs["pair_id"].isin(set(pid))]; pos |= set(sub["pos_idx"].astype(int).tolist())
                print(f"[extract] {f}: {len(pid)} rows -> {len(sub)} pairs matched", flush=True)
    if a.pairs_head:
        vp = pairs[~pairs["pos_idx"].isin(bad)].iloc[: a.pairs_head]; pos |= set(vp["pos_idx"].astype(int).tolist())      # = dump_rollouts' fixed set
    if a.pairs_slice:
        lo, hi = [int(x) for x in a.pairs_slice.split(":")]; vp = pairs[~pairs["pos_idx"].isin(bad)].iloc[lo:hi]; pos |= set(vp["pos_idx"].astype(int).tolist())
    pos -= bad
    return sorted(pos)


def stored_lookup(data_dir, split, pos_list):
    """pos_idx -> stored fp16 [26, d] rows via mmap (no full ActStore load)"""
    import pyarrow.parquet as pq
    want = set(int(p) for p in pos_list); out = {}
    for f in sorted(glob.glob(os.path.join(data_dir, split, "acts_*.npy"))):
        m = pq.read_table(meta_of(f), columns=["pos_idx"]).column(0).to_numpy()
        hit = np.where(np.isin(m, list(want)))[0]
        if len(hit) == 0: continue
        A = np.load(f, mmap_mode="r")
        for q in hit: out[int(m[q])] = np.array(A[q])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--split", default="train"); p.add_argument("--out-dir", required=True); p.add_argument("--tag", default="x")
    p.add_argument("--rows", default=None, help="comma list / globs of text row parquets (pair_id) whose positions are needed")
    p.add_argument("--pairs-head", type=int, default=0, help="also the positions of the first N spike-free rows of pairs_<split> (the fixed eval set)")
    p.add_argument("--pairs-slice", default=None, help="lo:hi rows of pairs_<split>")
    p.add_argument("--base", default="Qwen/Qwen3-8B"); p.add_argument("--tokens-per-batch", type=int, default=24000); p.add_argument("--shard-size", type=int, default=4096)
    p.add_argument("--check-n", type=int, default=512); p.add_argument("--max-pos", type=int, default=None)
    a = p.parse_args(); dev = "cuda"; t0 = time.time()
    import pyarrow as pa, pyarrow.parquet as pq
    from transformers import AutoModelForCausalLM
    os.makedirs(os.path.join(a.out_dir, a.split), exist_ok=True)
    pos_list = wanted_positions(a, a.data_dir, a.split)
    if a.max_pos: pos_list = pos_list[: a.max_pos]
    print(f"[extract] {len(pos_list)} positions wanted ({a.split})", flush=True)
    # ---- meta: pos_idx -> (doc_id, pos); docs: doc_id -> token_ids (only the needed docs)
    want = set(pos_list); meta = {}
    for f in sorted(glob.glob(os.path.join(a.data_dir, a.split, "meta_*.parquet"))):
        t = pq.read_table(f, columns=["pos_idx", "doc_id", "pos"]).to_pydict()
        for pi, di, ps in zip(t["pos_idx"], t["doc_id"], t["pos"]):
            if pi in want: meta[int(pi)] = (int(di), int(ps))
    assert len(meta) == len(want), f"{len(want) - len(meta)} positions not found in meta"
    need_docs = set(d for d, _ in meta.values()); docs = {}
    for f in sorted(glob.glob(os.path.join(a.data_dir, a.split, "docs_*.parquet"))):
        t = pq.read_table(f, columns=["doc_id", "token_ids"]).to_pydict()
        for di, ids in zip(t["doc_id"], t["token_ids"]):
            if di in need_docs: docs[int(di)] = ids
    assert len(docs) == len(need_docs), f"{len(need_docs) - len(docs)} docs missing"
    by_doc = {}
    for pi, (di, ps) in meta.items(): by_doc.setdefault(di, []).append((pi, ps))
    print(f"[extract] {len(docs)} docs, {sum(len(v) for v in by_doc.values())} positions, meta/doc load {time.time() - t0:.0f}s", flush=True)
    # ---- model (blocks 0..34 only) + hooks
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    inner = model.model; layers = inner.layers
    del layers[K_HI + 1:]; model.config.num_hidden_layers = K_HI + 1
    if hasattr(model.config, "layer_types"): model.config.layer_types = model.config.layer_types[: K_HI + 1]
    d = model.config.hidden_size
    state = {"sel": None, "res": None, "wr": None}

    def res_hook(k):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            b, pp = state["sel"]; state["res"][k - K_LO] = h[b, pp].float()
            if k == K_HI: raise _Stop()
        return hook

    def write_hook(k, which):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            b, pp = state["sel"]; state["wr"][k - W_LO, which] = h[b, pp].float()
        return hook
    for k in range(K_LO, K_HI + 1): layers[k].register_forward_hook(res_hook(k))
    for k in range(W_LO, K_HI + 1):
        layers[k].self_attn.register_forward_hook(write_hook(k, 0)); layers[k].mlp.register_forward_hook(write_hook(k, 1))
    pad_id = model.config.pad_token_id if getattr(model.config, "pad_token_id", None) is not None else 0
    # ---- batches of docs by length
    order = sorted(by_doc.keys(), key=lambda di: len(docs[di]))
    batches, cur, cur_tok = [], [], 0
    for di in order:
        L = len(docs[di])
        if cur and (len(cur) + 1) * max(L, cur_tok) > a.tokens_per_batch: batches.append(cur); cur, cur_tok = [], 0
        cur.append(di); cur_tok = max(cur_tok, L)
    if cur: batches.append(cur)
    print(f"[extract] {len(batches)} forward batches (<= {a.tokens_per_batch} padded tokens each)", flush=True)
    # ---- run
    buf_w, buf_meta, n_shard, n_done = [], [], 0, 0
    chk = {"layer_cos": [], "path_cos": [], "stored_cos": [], "write_norm": np.zeros((N_W, 2)), "write_n": 0, "h_norm": np.zeros(N_LAYERS)}
    stored = stored_lookup(a.data_dir, a.split, pos_list[: a.check_n]) if a.check_n else {}
    n_tok = 0

    def flush(force=False):
        nonlocal buf_w, buf_meta, n_shard
        if not buf_w or (not force and sum(x.shape[0] for x in buf_w) < a.shard_size): return
        W = np.concatenate(buf_w, 0); tag = f"{a.tag}_{n_shard:04d}"; out = os.path.join(a.out_dir, a.split)
        np.save(os.path.join(out, f"path_{tag}.tmp.npy"), W); os.replace(os.path.join(out, f"path_{tag}.tmp.npy"), os.path.join(out, f"path_{tag}.npy"))
        pq.write_table(pa.Table.from_pylist(buf_meta), os.path.join(out, f"pathmeta_{tag}.parquet"))
        print(f"[extract] wrote shard {tag}: {W.shape[0]} positions ({n_done} total, {time.time() - t0:.0f}s)", flush=True)
        buf_w, buf_meta = [], []; n_shard += 1

    for bi, dids in enumerate(batches):
        L = max(len(docs[di]) for di in dids); B = len(dids)
        ids = torch.full((B, L), pad_id, dtype=torch.long); am = torch.zeros((B, L), dtype=torch.long)
        sel_b, sel_p, sel_pi = [], [], []
        for r, di in enumerate(dids):
            t = docs[di]; ids[r, : len(t)] = torch.tensor(t); am[r, : len(t)] = 1
            for pi, ps in by_doc[di]: sel_b.append(r); sel_p.append(ps); sel_pi.append(pi)
        n = len(sel_b)
        state["sel"] = (torch.tensor(sel_b, device=dev), torch.tensor(sel_p, device=dev))
        state["res"] = torch.empty((N_LAYERS, n, d), dtype=torch.float32, device=dev); state["wr"] = torch.empty((N_W, 2, n, d), dtype=torch.float32, device=dev)
        with torch.no_grad():
            try: model(input_ids=ids.to(dev), attention_mask=am.to(dev), use_cache=False)
            except _Stop: pass
        res, wr = state["res"], state["wr"]                                  # [26, n, d], [25, 2, n, d]
        # checks (fp32, on the GPU)
        with torch.no_grad():
            rec = res[:-1] + wr[:, 0] + wr[:, 1]                               # h_{k-1} + a_k + m_k, k = 10..34
            cos_l = torch.nn.functional.cosine_similarity(rec, res[1:], dim=-1)   # [25, n]
            path = res[0] + (wr[:, 0] + wr[:, 1]).sum(0)
            cos_p = torch.nn.functional.cosine_similarity(path, res[-1], dim=-1)  # [n]
            chk["layer_cos"].append(cos_l.min(0).values.cpu().numpy()); chk["path_cos"].append(cos_p.cpu().numpy())
            chk["write_norm"] += wr.norm(dim=-1).sum(-1).cpu().numpy(); chk["write_n"] += n; chk["h_norm"] += res.norm(dim=-1).sum(-1).cpu().numpy()
            for q, pi in enumerate(sel_pi):
                if pi in stored:
                    s = torch.from_numpy(stored[pi].astype(np.float32)).to(dev)
                    chk["stored_cos"].append(float(torch.nn.functional.cosine_similarity(s, res[:, q], dim=-1).min()))
        W = wr.permute(2, 0, 1, 3).contiguous().to(torch.float16).cpu().numpy()     # [n, 25, 2, d]
        buf_w.append(W); buf_meta += [{"pos_idx": int(pi), "doc_id": int(meta[pi][0]), "pos": int(meta[pi][1])} for pi in sel_pi]
        n_done += n; n_tok += int(am.sum())
        flush()
        if bi % 20 == 0:
            dt = time.time() - t0
            print(f"[extract] batch {bi}/{len(batches)}: {n_done} positions, {n_tok / 1e6:.2f}M tok, {n_tok / dt / 1e3:.1f}k tok/s, {dt / 60:.1f} min; "
                  f"layer-cos min so far {float(np.concatenate(chk['layer_cos']).min()):.5f}, path-cos min {float(np.concatenate(chk['path_cos']).min()):.5f}", flush=True)
    flush(force=True)
    lc = np.concatenate(chk["layer_cos"]); pc = np.concatenate(chk["path_cos"]); sc = np.array(chk["stored_cos"]) if chk["stored_cos"] else np.array([np.nan])
    wn = chk["write_norm"] / max(1, chk["write_n"]); hn = chk["h_norm"] / max(1, chk["write_n"])
    summ = {"split": a.split, "tag": a.tag, "positions": n_done, "docs": len(docs), "tokens": n_tok, "seconds": time.time() - t0,
            "check_layer_cos": {"min": float(lc.min()), "mean": float(lc.mean()), "frac_lt_0.999": float((lc < 0.999).mean())},
            "check_path_cos_h9_to_h34": {"min": float(pc.min()), "mean": float(pc.mean()), "frac_lt_0.999": float((pc < 0.999).mean())},
            "check_vs_stored_cos": {"n": int(len(chk["stored_cos"])), "min": float(np.nanmin(sc)), "mean": float(np.nanmean(sc)), "frac_lt_0.999": float(np.nanmean(sc < 0.999))},
            "mean_norm_attn_by_layer": {str(k): float(wn[k - W_LO, 0]) for k in range(W_LO, K_HI + 1)}, "mean_norm_mlp_by_layer": {str(k): float(wn[k - W_LO, 1]) for k in range(W_LO, K_HI + 1)},
            "mean_norm_h_by_layer": {str(k): float(hn[k - K_LO]) for k in range(K_LO, K_HI + 1)}, "args": vars(a)}
    json.dump(summ, open(os.path.join(a.out_dir, a.split, f"check_{a.tag}.json"), "w"), indent=1)
    print("[extract] DONE " + json.dumps({k: v for k, v in summ.items() if not k.startswith("mean_norm") and k != "args"}), flush=True)
    print("[extract] mean write norms attn/mlp vs h by layer: " + ", ".join(f"{k}: {wn[k - W_LO, 0]:.1f}/{wn[k - W_LO, 1]:.1f} (h {hn[k - K_LO]:.0f})" for k in range(W_LO, K_HI + 1)), flush=True)


if __name__ == "__main__":
    main()
