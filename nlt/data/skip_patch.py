"""Skip-patch KL and final top-16 log-probs for the val pairs (redteam EVALS rows 8-9, DECISIONS D6 "causal and usefulness evals").

For a pair (i, j) at position p of a document: run Qwen3-8B on tokens[:p+1]; at the OUTPUT of block j replace position p's residual by
h_i = hidden_states[i+1][p] (i.e. skip the computation of blocks i+1..j at that position; all other positions untouched); measure
KL(clean || patched) of the next-token distribution at p, and store the clean top-16 (ids, log-probs), the patched top-1, and whether
the argmax changed. One forward per batch handles every j: each layer's hook patches the rows whose j equals that layer.

  python -m nlt.data.skip_patch --data-dir /vol/data/qwen3_8b --out /vol/data/qwen3_8b/val_skip.parquet [--max-pairs 0]
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np, torch, torch.nn.functional as F
import pyarrow as pa, pyarrow.parquet as pq
from nlt.data.dataset import ActStore
from nlt.data.extract import K_LO, K_HI


class _Stop(Exception):
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--out", required=True); p.add_argument("--base", default="Qwen/Qwen3-8B")
    p.add_argument("--split", default="val"); p.add_argument("--max-pairs", type=int, default=0); p.add_argument("--batch", type=int, default=24); p.add_argument("--max-ctx", type=int, default=1024)
    a = p.parse_args(); dev = "cuda"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base); pad = tok.pad_token_id or 0
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    layers = model.model.layers
    store = ActStore(a.data_dir, a.split, device="cpu"); store.load_docs(a.data_dir)
    pairs = pq.read_table(os.path.join(a.data_dir, f"pairs_{a.split}.parquet")).to_pandas()
    pairs = pairs[pairs["pos_idx"].isin(store.row_of)]
    if a.max_pairs: pairs = pairs.iloc[: a.max_pairs]
    pairs = pairs.sort_values("pos").reset_index(drop=True)                     # similar lengths together -> less padding
    state = {"patch": {}}                                                       # layer k -> (row idx tensor, pos tensor, replacement [n, d])

    def make_hook(k):
        def hook(_m, _i, out):
            if k not in state["patch"]: return None
            h = out[0] if isinstance(out, tuple) else out
            r, q, rep = state["patch"][k]; h[r, q] = rep.to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook
    for k in range(K_LO, K_HI + 1): layers[k].register_forward_hook(make_hook(k))
    rows_out = []; t0 = time.time()
    for s in range(0, len(pairs), a.batch):
        sub = pairs.iloc[s:s + a.batch]; B = len(sub)
        ctx = [store.context_ids(int(pi), a.max_ctx) for pi in sub["pos_idx"]]; L = max(len(c) for c in ctx)
        ids = torch.full((B, L), pad, dtype=torch.long); am = torch.zeros((B, L), dtype=torch.long); pos = torch.zeros(B, dtype=torch.long)
        for b, c in enumerate(ctx): ids[b, :len(c)] = torch.tensor(c); am[b, :len(c)] = 1; pos[b] = len(c) - 1
        ids, am, pos_d = ids.to(dev), am.to(dev), pos.to(dev)
        with torch.no_grad():
            state["patch"] = {}
            clean = model(input_ids=ids, attention_mask=am, use_cache=False).logits[torch.arange(B, device=dev), pos_d].float()
            state["patch"] = {}
            srows = store.rows_for(sub["pos_idx"].values)
            for k in range(K_LO, K_HI + 1):
                m = (sub["j"].values == k)
                if m.any():
                    r = torch.tensor(np.where(m)[0], device=dev); rep = store.gather(srows[m], torch.tensor(sub["i"].values[m]), dev).float()
                    state["patch"][k] = (r, pos_d[r], rep)
            patched = model(input_ids=ids, attention_mask=am, use_cache=False).logits[torch.arange(B, device=dev), pos_d].float()
            state["patch"] = {}
        lc = F.log_softmax(clean, -1); lp = F.log_softmax(patched, -1)
        kl = (lc.exp() * (lc - lp)).sum(-1); top = lc.topk(16, -1)
        nxt = torch.tensor(sub["next_token_id"].values, device=dev)
        for b in range(B):
            rows_out.append({"pair_id": sub["pair_id"].iloc[b], "pos_idx": int(sub["pos_idx"].iloc[b]), "i": int(sub["i"].iloc[b]), "j": int(sub["j"].iloc[b]), "pos": int(sub["pos"].iloc[b]),
                             "kl_skip_nats": float(kl[b]), "clean_top16_ids": top.indices[b].tolist(), "clean_top16_logprobs": top.values[b].tolist(),
                             "clean_top1": int(top.indices[b, 0]), "patched_top1": int(lp[b].argmax()), "top1_changed": bool(top.indices[b, 0] != lp[b].argmax()),
                             "clean_logprob_true_next": float(lc[b, nxt[b]]), "patched_logprob_true_next": float(lp[b, nxt[b]])})
        if (s // a.batch) % 20 == 0: print(f"[skip] {s + B}/{len(pairs)} pairs, {time.time() - t0:.0f}s, mean KL so far {np.mean([r_['kl_skip_nats'] for r_ in rows_out]):.3f} nats", flush=True)
    tab = pa.Table.from_pylist(rows_out); pq.write_table(tab, a.out)
    df = tab.to_pandas(); g = df.groupby(df["j"] - df["i"])["kl_skip_nats"].mean()
    print("[skip] mean KL by gap:", json.dumps({int(k): round(float(v), 3) for k, v in g.items()}), flush=True)
    print("[skip] mean KL by j:", json.dumps({int(k): round(float(v), 3) for k, v in df.groupby("j")["kl_skip_nats"].mean().items()}), flush=True)
    print(f"[skip] DONE {len(df)} pairs -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
