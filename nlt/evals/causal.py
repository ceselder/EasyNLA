"""Measured causal / usefulness quantities for the val pairs (EVALS 8a, 8b, 9a): one table, computed ONCE from the real forward pass,
never from flow samples.
  kl_skip      KL(p_clean || p_patched) of the model's next-token distribution at the position when the residual AFTER block j is
               replaced by the clean residual after block i (i.e. blocks i+1..j are skipped at that position)           -> 8a magnitude target
  lens_i/j     logit-lens top-5 tokens at i and at j (W_U . final_norm(h)), and KL(lens_j || lens_i)                      -> 8b direction candidates
  final_top16  the model's final next-token top-16 ids + logprobs at the position; true next token                          -> 9a candidates
Also checks that the recomputed h_k matches the stored activations (cos >= 0.99) so the pairs table and the model agree.

  python -m nlt.evals.causal --data-dir /vol/data/qwen3_8b --out /vol/evals/causal_val.parquet [--n-pairs 4096 --model Qwen/Qwen3-8B --max-ctx 1024]
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np, torch, torch.nn.functional as F


def main():
    from nlt.data.dataset import ActStore
    from nlt.evals.common import load_table, save_table
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--out", required=True); p.add_argument("--n-pairs", type=int, default=4096)
    p.add_argument("--model", default="Qwen/Qwen3-8B"); p.add_argument("--max-ctx", type=int, default=1024); p.add_argument("--topk", type=int, default=16); p.add_argument("--pairs", default=None)
    p.add_argument("--shard", default="0/1", help="k/n: process pairs k::n (run n containers in parallel, then concatenate the parquets)")
    a = p.parse_args(); dev = "cuda"; sk, sn = (int(x) for x in a.shard.split("/"))
    store = ActStore(a.data_dir, "val", device="cpu", verbose=True); store.load_docs(a.data_dir)
    pairs = load_table(a.pairs or os.path.join(a.data_dir, "pairs_val.parquet")); pairs = pairs[pairs["pos_idx"].isin(store.row_of)].iloc[: a.n_pairs]
    pairs = pairs.sort_values("pos_idx").iloc[sk::sn].reset_index(drop=True)          # sorted by position so the clean-forward cache hits; shard = strided slice
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval(); model.requires_grad_(False)
    layers = model.model.layers; W_U = model.lm_head.weight; fnorm = model.model.norm
    patch = {"on": False, "layer": None, "vec": None}
    def make_hook(k):
        def hook(_m, _i, out):
            if patch["on"] and patch["layer"] == k:
                h = out[0] if isinstance(out, tuple) else out
                h[:, -1, :] = patch["vec"].to(h.dtype)
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook
    for k, blk in enumerate(layers): blk.register_forward_hook(make_hook(k))

    @torch.no_grad()
    def lens_top(h, k=5):
        logits = (W_U @ fnorm(h.to(torch.bfloat16)).float().to(W_U.dtype)).float(); lp = F.log_softmax(logits, -1)
        top = torch.topk(lp, k); return top.indices.tolist(), top.values.tolist(), lp

    out_rows = []; cache = {}; t0 = time.time(); cos_check = []
    for n, r in enumerate(pairs.itertuples()):
        pos_idx, i, j = int(r.pos_idx), int(r.i), int(r.j)
        if pos_idx not in cache:
            ids = torch.tensor(store.context_ids(pos_idx, ctx=a.max_ctx), device=dev)[None]
            with torch.no_grad():
                o = model(input_ids=ids, output_hidden_states=True, use_cache=False)
            hs = [h[0, -1].float() for h in o.hidden_states]          # hs[k+1] = residual after block k
            lp_final = F.log_softmax(o.logits[0, -1].float(), -1)
            m_ = store.meta.iloc[store.row_of[pos_idx]]
            st = store.gather(torch.tensor([store.row_of[pos_idx]]), torch.tensor([j]))[0].float()
            cos_check.append(F.cosine_similarity(hs[j + 1].cpu(), st, dim=0).item())
            cache = {pos_idx: (ids, hs, lp_final, int(m_["next_token_id"]))}       # keep only the current position (pairs are grouped by pos in practice; else recompute)
        ids, hs, lp_final, next_id = cache[pos_idx]
        patch.update(on=True, layer=j, vec=hs[i + 1])
        with torch.no_grad():
            lp_patch = F.log_softmax(model(input_ids=ids, use_cache=False).logits[0, -1].float(), -1)
        patch["on"] = False
        kl_skip = float((lp_final.exp() * (lp_final - lp_patch)).sum())
        li_ids, li_lp, li_full = lens_top(hs[i + 1]); lj_ids, lj_lp, lj_full = lens_top(hs[j + 1])
        kl_lens = float((lj_full.exp() * (lj_full - li_full)).sum())
        top = torch.topk(lp_final, a.topk)
        out_rows.append(dict(pair_id=str(r.pair_id), pos_idx=pos_idx, i=i, j=j, kl_skip=kl_skip, kl_lens_j_vs_i=kl_lens,
                             lens_i_top=[tok.decode([t]) for t in li_ids], lens_i_lp=li_lp, lens_j_top=[tok.decode([t]) for t in lj_ids], lens_j_lp=lj_lp,
                             final_top_ids=top.indices.tolist(), final_top_lp=top.values.tolist(), final_top_tokens=[tok.decode([t]) for t in top.indices.tolist()],
                             next_token_id=next_id, next_token=tok.decode([next_id]), patched_top1=tok.decode([int(lp_patch.argmax())]), final_top1=tok.decode([int(lp_final.argmax())])))
        if n % 200 == 0: print(f"[causal] {n}/{len(pairs)} pairs, {time.time() - t0:.0f}s, mean cos(stored, recomputed h_j) {np.mean(cos_check):.4f}", flush=True)
    import pandas as pd
    df = pd.DataFrame(out_rows); save_table(df, a.out)
    summ = {"n": len(df), "kl_skip_mean": float(df.kl_skip.mean()), "kl_skip_median": float(df.kl_skip.median()),
            "kl_skip_by_gap": {str(g): float(v) for g, v in df.groupby(df.j - df.i).kl_skip.mean().items()},
            "kl_skip_by_j": {str(g): float(v) for g, v in df.groupby("j").kl_skip.mean().items()},
            "lens_top1_changes_share": float((df.lens_i_top.str[0] != df.lens_j_top.str[0]).mean()), "final_top1_is_next_share": float((df.final_top1 == df.next_token).mean()),
            "cos_stored_vs_recomputed_mean": float(np.mean(cos_check)), "cos_min": float(np.min(cos_check))}
    json.dump(summ, open(a.out + ".summary.json", "w"), indent=1); print(json.dumps(summ, indent=1)); print("[causal] DONE ->", a.out, flush=True)


if __name__ == "__main__":
    main()
