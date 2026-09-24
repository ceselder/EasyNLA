"""Per-layer ATTENTION and MLP writes (a_k, m_k; k = K_LO..K_HI) at the positions already in the activation store (DECISIONS v1.26 pathverb /
v1.27 featurizer). Re-runs Qwen3-8B on the stored documents (docs_*.parquet token_ids), hooks self_attn and mlp of blocks K_LO..K_HI, gathers the
outputs at the stored (doc_id, pos) and writes, per input shard, acts_attn_<tag>.npy + acts_mlp_<tag>.npy [n, N_LAYERS, d] fp16 plus a
meta_<tag>.parquet with the SAME pos_idx as the main store (so consumers join on pos_idx). Identity check per position: a_k + m_k == h_k - h_(k-1)
(residual stream), reported as a relative error in stats_writes.json together with per-layer RMS of each write type.

  python -m nlt.data.extract_writes --data-dir /vol/data/qwen3_8b --split val --out-dir /vol/data/qwen3_8b_writes [--pos-idx-file ids.json] [--shards 0,1]
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
    state = {"sel": None, "attn": None, "mlp": None, "res": None}

    def attn_hook(k):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            b, p = state["sel"]; state["attn"][k - K_LO] = h[b, p].to(torch.float16)
        return hook

    def mlp_hook(k):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            b, p = state["sel"]; state["mlp"][k - K_LO] = h[b, p].to(torch.float16)
        return hook

    def block_hook(k):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            b, p = state["sel"]; state["res"][k - K_LO + 1] = h[b, p].to(torch.float16)
            if k == K_HI: raise _Stop()
        return hook
    for k in range(K_LO, K_HI + 1):
        layers[k].self_attn.register_forward_hook(attn_hook(k)); layers[k].mlp.register_forward_hook(mlp_hook(k)); layers[k].register_forward_hook(block_hook(k))
    layers[K_LO - 1].register_forward_hook(lambda _m, _i, out: state["res"].__setitem__(0, (out[0] if isinstance(out, tuple) else out)[state["sel"][0], state["sel"][1]].to(torch.float16)))
    return model, state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--split", default="val"); p.add_argument("--out-dir", required=True)
    p.add_argument("--base", default="Qwen/Qwen3-8B"); p.add_argument("--pos-idx-file", default="", help="json list of pos_idx to keep (default: all positions of the split)")
    p.add_argument("--shards", default="", help="comma list of shard indices to process (default all)"); p.add_argument("--docs-per-batch", type=int, default=8)
    a = p.parse_args()
    import pyarrow.parquet as pq, pyarrow as pa
    dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = True
    model, state = build(a.base, dev); d = model.config.hidden_size
    out_dir = os.path.join(a.out_dir, a.split); os.makedirs(out_dir, exist_ok=True)
    keep = None
    if a.pos_idx_file:   # json list OR one int per line (featurizer's /vol/feat/train20k_pos_idx.txt)
        txt = open(a.pos_idx_file).read().strip()
        keep = set(int(x) for x in (json.loads(txt) if txt.startswith("[") else txt.split()))
        print(f"[writes:{a.split}] keeping {len(keep)} pos_idx from {a.pos_idx_file}", flush=True)
    metas = sorted(glob.glob(os.path.join(a.data_dir, a.split, "meta_*.parquet")))
    if a.shards: want = {int(x) for x in a.shards.split(",")}; metas = [m for i, m in enumerate(metas) if i in want]
    rms_attn = np.zeros(N_LAYERS); rms_mlp = np.zeros(N_LAYERS); rms_res = np.zeros(N_LAYERS); n_tot = 0; ident_err = []; t0 = time.time()
    for mpath in metas:
        tag = os.path.basename(mpath)[len("meta_"):-len(".parquet")]
        meta = pq.read_table(mpath).to_pandas(); docs = pq.read_table(mpath.replace("meta_", "docs_")).to_pandas().set_index("doc_id")
        if keep is not None: meta = meta[meta["pos_idx"].isin(keep)]
        if len(meta) == 0: continue
        A_out = np.zeros((len(meta), N_LAYERS, d), np.float16); M_out = np.zeros_like(A_out); R_out = np.zeros((len(meta), N_LAYERS + 1, d), np.float16)
        row_of = {int(pi): r for r, pi in enumerate(meta["pos_idx"].values)}
        groups = list(meta.groupby("doc_id"))
        for g0 in range(0, len(groups), a.docs_per_batch):
            batch = groups[g0:g0 + a.docs_per_batch]
            seqs = [torch.tensor(docs.loc[doc_id, "token_ids"], dtype=torch.long) for doc_id, _ in batch]
            L = max(len(s) for s in seqs); ids = torch.zeros(len(seqs), L, dtype=torch.long); att = torch.zeros(len(seqs), L, dtype=torch.long)
            for bi, s in enumerate(seqs): ids[bi, :len(s)] = s; att[bi, :len(s)] = 1
            bsel, psel, rows = [], [], []
            for bi, (doc_id, g) in enumerate(batch):
                for pi, pos in zip(g["pos_idx"].values, g["pos"].values): bsel.append(bi); psel.append(int(pos)); rows.append(row_of[int(pi)])
            state["sel"] = (torch.tensor(bsel, device=dev), torch.tensor(psel, device=dev))
            state["attn"] = torch.zeros(N_LAYERS, len(rows), d, dtype=torch.float16, device=dev); state["mlp"] = torch.zeros_like(state["attn"]); state["res"] = torch.zeros(N_LAYERS + 1, len(rows), d, dtype=torch.float16, device=dev)
            with torch.no_grad():
                try: model(input_ids=ids.to(dev), attention_mask=att.to(dev))
                except _Stop: pass
            A_out[rows] = state["attn"].permute(1, 0, 2).cpu().numpy(); M_out[rows] = state["mlp"].permute(1, 0, 2).cpu().numpy(); R_out[rows] = state["res"].permute(1, 0, 2).cpu().numpy()
        # identity check a_k + m_k == h_k - h_(k-1) (bf16 forward -> expect ~1e-2 relative)
        delta = R_out[:, 1:].astype(np.float32) - R_out[:, :-1].astype(np.float32); summ = A_out.astype(np.float32) + M_out.astype(np.float32)
        ident_err.append(float(np.linalg.norm(delta - summ) / (np.linalg.norm(delta) + 1e-6)))
        rms_attn += (A_out.astype(np.float32) ** 2).mean(-1).sum(0); rms_mlp += (M_out.astype(np.float32) ** 2).mean(-1).sum(0); rms_res += (delta ** 2).mean(-1).sum(0); n_tot += len(meta)
        np.save(os.path.join(out_dir, f"acts_attn_{tag}.npy"), A_out); np.save(os.path.join(out_dir, f"acts_mlp_{tag}.npy"), M_out)
        pq.write_table(pa.Table.from_pandas(meta.reset_index(drop=True)), os.path.join(out_dir, f"meta_{tag}.parquet"))
        print(f"[writes:{a.split}] shard {tag}: {len(meta)} positions, identity rel err {ident_err[-1]:.4f}, {time.time() - t0:.0f}s", flush=True)
    stats = {"layers": list(range(K_LO, K_HI + 1)), "n_positions": n_tot, "rms_attn_by_layer": np.sqrt(rms_attn / max(1, n_tot)).tolist(), "rms_mlp_by_layer": np.sqrt(rms_mlp / max(1, n_tot)).tolist(),
             "rms_delta_by_layer": np.sqrt(rms_res / max(1, n_tot)).tolist(), "identity_rel_err_by_shard": ident_err, "layout": "acts_attn_<tag>.npy / acts_mlp_<tag>.npy: [n, 26, 4096] fp16, row r <-> meta_<tag>.parquet row r (pos_idx joins the main store); layer axis = k-9 for k=9..34; a_k = self_attn output of block k, m_k = mlp output of block k; a_k + m_k = h_k - h_(k-1)"}
    json.dump(stats, open(os.path.join(out_dir, "stats_writes.json"), "w"), indent=1); print(f"[writes:{a.split}] DONE {n_tot} positions -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
