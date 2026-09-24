"""HF-generate rollouts of a PATH verbalizer on the fixed eval pairs (first --n-pairs spike-free rows of pairs_val, the same set as
nlt.rl.dump_rollouts), in the board #31 text format so redteam's watcher scores it: [pair_id, text, n_tokens, verbosity, source, sample_idx].

  python -m nlt.path.dump --data-dir /vol/data/qwen3_8b --init lora:/vol/rl/sft/v0b_path_d/lora --path-mode delta --source v0b_path_d_0 \
      --out /vol/z/v0b_path_d_0/val/part_0000000_0004096.parquet

Plain HF generate (no vLLM): the N-marker injection is one HF hook, and 4096 x 96 tokens is minutes on one GPU. T=0.7, one sample per pair.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np, torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--out", required=True); p.add_argument("--split", default="val")
    p.add_argument("--base", default="Qwen/Qwen3-8B"); p.add_argument("--init", required=True); p.add_argument("--source", required=True); p.add_argument("--question", default=None)
    p.add_argument("--path-mode", default="delta", choices=["none", "count", "delta", "attn_mlp"]); p.add_argument("--path-dir", default="/vol/path/qwen3_8b")
    p.add_argument("--n-pairs", type=int, default=4096); p.add_argument("--n-samples", type=int, default=1); p.add_argument("--batch", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.7); p.add_argument("--max-new-tokens", type=int, default=96); p.add_argument("--seed", type=int, default=0); p.add_argument("--verbosity", type=int, default=1)
    a = p.parse_args(); torch.manual_seed(a.seed); dev = "cuda"
    import pyarrow as pa, pyarrow.parquet as pq
    from nlt.data.dataset import ActStore
    from nlt.verbalizer.model import load_tokenizer, load_policy
    from nlt.path.prompt import build_path_prompt
    from nlt.path.inject import MultiMarkerInjector, pack_slot
    from nlt.path.vectors import PathStore, path_inputs, n_mid_of
    tok = load_tokenizer(a.base); pad = tok.pad_token_id
    store = ActStore(a.data_dir, a.split, device="cpu")
    vp = pq.read_table(os.path.join(a.data_dir, f"pairs_{a.split}.parquet")).to_pandas(); vp = vp[vp["pos_idx"].isin(store.row_of)].iloc[: a.n_pairs].reset_index(drop=True)
    N = len(vp); pstore = None
    if a.path_mode == "attn_mlp":
        pstore = PathStore(a.path_dir, a.split, pos_idx_needed=vp["pos_idx"].values)
        ok = vp["pos_idx"].map(lambda q: int(q) in pstore.row_of).values; assert ok.all(), f"{(~ok).sum()} eval positions lack path vectors"
    policy = load_policy(a.base, a.init, device=dev); policy.eval(); inj = MultiMarkerInjector(policy, build_path_prompt(tok, 0).marker_id)
    eos_ids = sorted(set([tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]))
    I = vp["i"].values.astype(int); J = vp["j"].values.astype(int)
    n_mid = np.array([n_mid_of(i_, j_, a.path_mode) for i_, j_ in zip(I, J)]); order = np.argsort(n_mid, kind="stable")
    print(f"[path-dump] {N} pairs, markers per prompt {int(n_mid.min()) + 2}..{int(n_mid.max()) + 2}, mode {a.path_mode}, init {a.init}", flush=True)
    t0 = time.time(); out_rows = {}; n_writes = 0; n_expected = 0
    for s in range(0, N, a.batch):
        idx = order[s: s + a.batch]; sub = vp.iloc[idx]
        vecs = path_inputs(store, sub["pos_idx"].values, sub["i"].values, sub["j"].values, a.path_mode, pstore)
        specs = [build_path_prompt(tok, int(n_mid[q]), a.question) for q in idx]
        L = max(sp.n for sp in specs); B = len(idx)
        ids = torch.full((B, L), pad, dtype=torch.long); am = torch.zeros((B, L), dtype=torch.long); offs = []
        for r, sp in enumerate(specs): off = L - sp.n; ids[r, off:] = torch.tensor(sp.ids); am[r, off:] = 1; offs.append(off)      # LEFT padding for generate
        slot = pack_slot(vecs, [sp.positions for sp in specs], offsets=offs); n_expected += int((slot[1] >= 0).sum())
        for g in range(a.n_samples):
            inj.ref[0] = (slot[0].to(dev), slot[1].to(dev)); inj.reset_count()
            try:
                with torch.no_grad():
                    gen = policy.generate(input_ids=ids.to(dev), attention_mask=am.to(dev), do_sample=a.temperature > 0, temperature=a.temperature if a.temperature > 0 else None,
                                          top_p=1.0, top_k=0, max_new_tokens=a.max_new_tokens, pad_token_id=pad, eos_token_id=eos_ids, use_cache=True)
            finally:
                inj.ref[0] = None
            n_writes += inj.reset_count()
            for r, q in enumerate(idx):
                toks = gen[r, L:].tolist(); n_resp = 0; kept = []
                for t in toks:
                    if t in eos_ids or t == pad: break
                    kept.append(t); n_resp += 1
                out_rows[(int(q), g)] = (vp["pair_id"].iloc[q], tok.decode(kept, skip_special_tokens=True).strip(), n_resp, g, len(kept) >= a.max_new_tokens)
        if (s // a.batch) % 8 == 0:
            print(f"[path-dump] {min(s + a.batch, N)}/{N} pairs, {time.time() - t0:.0f}s, prompt len {L}, writes {n_writes}/{n_expected * (1 if a.n_samples == 1 else a.n_samples)}", flush=True)
    keys = sorted(out_rows); rows = [out_rows[k] for k in keys]
    tbl = pa.table({"pair_id": [r[0] for r in rows], "text": [r[1] for r in rows], "n_tokens": pa.array([int(r[2]) for r in rows], pa.int32()),
                    "verbosity": pa.array([a.verbosity] * len(rows), pa.int32()), "source": [a.source] * len(rows), "sample_idx": pa.array([int(r[3]) for r in rows], pa.int32())})
    os.makedirs(os.path.dirname(a.out), exist_ok=True); pq.write_table(tbl, a.out)
    ntok = np.array([r[2] for r in rows])
    summ = {"rows": len(rows), "pairs": N, "n_samples": a.n_samples, "temperature": a.temperature, "init": a.init, "source": a.source, "path_mode": a.path_mode,
            "tokens_mean": float(ntok.mean()), "tokens_median": float(np.median(ntok)), "empty_frac": float(np.mean([len(r[1]) == 0 for r in rows])),
            "truncated_frac": float(np.mean([r[4] for r in rows])), "marker_writes": n_writes, "marker_writes_expected": n_expected * a.n_samples, "seconds": time.time() - t0}
    json.dump(summ, open(a.out.replace(".parquet", ".summary.json"), "w"), indent=1); print(json.dumps(summ), flush=True)
    for r, q in list(zip(rows, keys))[:: max(1, len(rows) // 12)][:12]: print(f"  [{r[0]} {I[q[0]]}->{J[q[0]]}] {r[1][:200]!r}", flush=True)
    print(f"[path-dump] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
