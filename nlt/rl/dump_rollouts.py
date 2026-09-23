"""Dump verbalizer rollouts on the fixed eval pairs in the board #31 text format (redteam #102), so a policy checkpoint goes through the
same gates as every pool source (text-only rows, control manifests for the exact scorer, paraphrase / twin batches, readers).

  python -m nlt.rl.dump_rollouts --data-dir /vol/data/qwen3_8b --init lora:/vol/rl/sft/v0_ao_tsv1/lora --source v0-ao-tsv1 \
      --out /vol/z/v0-ao-tsv1/val/part_0000000_0004096.parquet --n-pairs 4096 --n-multi 512 --group-multi 4

Rows: [pair_id, text, n_tokens, verbosity=1, source, sample_idx]; the first --n-multi pairs get --group-multi samples (sample_idx 0..G-1),
the rest one sample (sample_idx 0). Sampling at --temperature (0.7 default; 0 = greedy).
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np, torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--out", required=True); p.add_argument("--split", default="val")
    p.add_argument("--base", default="Qwen/Qwen3-8B"); p.add_argument("--init", required=True); p.add_argument("--source", required=True); p.add_argument("--question", default=None)
    p.add_argument("--n-pairs", type=int, default=4096); p.add_argument("--n-multi", type=int, default=512); p.add_argument("--group-multi", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.7); p.add_argument("--max-new-tokens", type=int, default=96); p.add_argument("--vllm-gpu-mem", type=float, default=0.40)
    p.add_argument("--vllm-max-len", type=int, default=512); p.add_argument("--seed", type=int, default=0); p.add_argument("--verbosity", type=int, default=1)
    a = p.parse_args(); torch.manual_seed(a.seed)
    import pyarrow as pa, pyarrow.parquet as pq
    from nlt.data.dataset import ActStore
    from nlt.verbalizer.prompt import build_prompt, DEFAULT_QUESTION
    from nlt.verbalizer.model import load_tokenizer, load_policy
    from nlt.verbalizer.vllm_rollout import make_engine, rollout
    tok = load_tokenizer(a.base); spec = build_prompt(tok, a.question or DEFAULT_QUESTION)
    store = ActStore(a.data_dir, a.split, device="cpu")
    vp = pq.read_table(os.path.join(a.data_dir, f"pairs_{a.split}.parquet")).to_pandas(); vp = vp[vp["pos_idx"].isin(store.row_of)].iloc[: a.n_pairs].reset_index(drop=True)
    rows = store.rows_for(vp["pos_idx"].values); I = torch.as_tensor(vp["i"].values).long(); J = torch.as_tensor(vp["j"].values).long()
    acts = torch.stack([store.gather(rows, I), store.gather(rows, J)], 1).float(); N = len(vp)
    policy = load_policy(a.base, a.init, device="cuda"); policy.eval()
    llm = make_engine(a.base, tokenizer=a.base, gpu_mem=a.vllm_gpu_mem, max_len=a.vllm_max_len, seed=a.seed)
    if a.init != "base":
        from nla.train_rl_vllm import sync_actor_to_vllm
        sync_actor_to_vllm(policy, llm)
    del policy; torch.cuda.empty_cache()
    t0 = time.time(); out_rows = []
    nm = min(a.n_multi, N)
    if nm > 0 and a.group_multi > 0:
        res, info = rollout(llm, spec, acts[:nm], a.group_multi, a.max_new_tokens, a.temperature, seed=a.seed)
        for r in res: out_rows.append((vp["pair_id"][r["prompt_idx"]], r["text"].strip(), r["n_resp"], r["group_idx"], r["truncated"]))
        print(f"[dump] multi: {len(res)} rollouts for {nm} pairs, {info}", flush=True)
    if N > nm:
        res, info = rollout(llm, spec, acts[nm:], 1, a.max_new_tokens, a.temperature, seed=a.seed + 1)
        for r in res: out_rows.append((vp["pair_id"][nm + r["prompt_idx"]], r["text"].strip(), r["n_resp"], 0, r["truncated"]))
        print(f"[dump] single: {len(res)} rollouts for {N - nm} pairs, {info}", flush=True)
    tbl = pa.table({"pair_id": [r[0] for r in out_rows], "text": [r[1] for r in out_rows], "n_tokens": pa.array([int(r[2]) for r in out_rows], pa.int32()),
                    "verbosity": pa.array([a.verbosity] * len(out_rows), pa.int32()), "source": [a.source] * len(out_rows), "sample_idx": pa.array([int(r[3]) for r in out_rows], pa.int32())})
    os.makedirs(os.path.dirname(a.out), exist_ok=True); pq.write_table(tbl, a.out)
    ntok = np.array([r[2] for r in out_rows]); summ = {"rows": len(out_rows), "pairs": N, "n_multi": nm, "group_multi": a.group_multi, "temperature": a.temperature, "init": a.init, "source": a.source,
                                                       "tokens_mean": float(ntok.mean()), "tokens_median": float(np.median(ntok)), "empty_frac": float(np.mean([len(r[1]) == 0 for r in out_rows])),
                                                       "truncated_frac": float(np.mean([r[4] for r in out_rows])), "seconds": time.time() - t0}
    json.dump(summ, open(a.out.replace(".parquet", ".summary.json"), "w"), indent=1); print(json.dumps(summ), flush=True)
    for r in out_rows[:6]: print(f"  [{r[0]}] {r[1][:200]!r}", flush=True)
    print(f"[dump] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
