"""Step-0 signal test (board #22/#26 criteria, DECISIONS D4): does the initial policy already produce within-group exact-bits variance
above the scoring noise, per band, and are its bits about the pair (vs the depth-matched shuffle)?

  python -m nlt.rl.step0 --data-dir /vol/data/qwen3_8b --out /vol/rl/step0/ao.json --init ao --critic /vol/critic/text_v1/ckpt_latest.pt --n-pairs 512 --group 8

Per band (pre j<=13 / workspace 14-32 / motor >=33): mean bits, within-group std, scoring noise (same texts rescored with another
probe/eps seed: std(b0 - b1)/sqrt2), ratio, bits(z_dm) (another pair's text, same (i, j)), bits(z_rp) (random pair's text), share of
bits <= 0, tokens, next-token mention rate, copy, regex, corr(bits, tokens). PASS (workspace): std >= 3x noise, bits >= 3x bits_dm,
mention < 20 %.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np, torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--out", required=True); p.add_argument("--tag", default="step0")
    p.add_argument("--base", default="Qwen/Qwen3-8B"); p.add_argument("--init", default="ao"); p.add_argument("--question", default=None)
    p.add_argument("--critic", default=None); p.add_argument("--stub-critic", action="store_true"); p.add_argument("--ode-steps", type=int, default=32)
    p.add_argument("--probes", type=int, default=1); p.add_argument("--score-batch", type=int, default=64); p.add_argument("--enc-model", default=None); p.add_argument("--enc-layer", type=int, default=None)
    p.add_argument("--n-pairs", type=int, default=512); p.add_argument("--group", type=int, default=8); p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0); p.add_argument("--vllm-gpu-mem", type=float, default=0.40); p.add_argument("--vllm-max-len", type=int, default=512)
    p.add_argument("--lam", type=float, default=0.1); p.add_argument("--seed", type=int, default=0); p.add_argument("--skip", type=int, default=0, help="skip the first rows of pairs_val")
    a = p.parse_args(); torch.manual_seed(a.seed)
    import pyarrow.parquet as pq
    from nlt.data.dataset import ActStore
    from nlt.evals.common import band
    from nlt.verbalizer.prompt import build_prompt, DEFAULT_QUESTION
    from nlt.verbalizer.model import load_tokenizer, load_policy
    from nlt.verbalizer.vllm_rollout import make_engine, rollout
    from nlt.rl.filters import ViolationChecker
    from nlt.rl.reward import make_scorer, within_group_std, corr
    n_gpu = torch.cuda.device_count(); dev = "cuda:0"; cdev = "cuda:1" if n_gpu > 1 else "cuda:0"
    tok = load_tokenizer(a.base); spec = build_prompt(tok, a.question or DEFAULT_QUESTION)
    store = ActStore(a.data_dir, "val", device="cpu"); vc = ViolationChecker(store, a.data_dir)
    vp = pq.read_table(os.path.join(a.data_dir, "pairs_val.parquet")).to_pandas(); vp = vp[vp["pos_idx"].isin(store.row_of)].iloc[a.skip: a.skip + a.n_pairs].reset_index(drop=True)
    rows = store.rows_for(vp["pos_idx"].values); I = torch.as_tensor(vp["i"].values).long(); J = torch.as_tensor(vp["j"].values).long(); N = len(vp)
    h_i = store.gather(rows, I).float(); h_j = store.gather(rows, J).float(); acts = torch.stack([h_i, h_j], 1)
    bands = np.array([band(int(j)) for j in J.tolist()]); print(f"[step0] {N} pairs: " + ", ".join(f"{b}={int((bands == b).sum())}" for b in ("pre", "workspace", "motor")), flush=True)
    policy = load_policy(a.base, a.init, device=dev); policy.eval()
    llm = make_engine(a.base, tokenizer=a.base, gpu_mem=a.vllm_gpu_mem, max_len=a.vllm_max_len, seed=a.seed)
    if a.init != "base":
        from nla.train_rl_vllm import sync_actor_to_vllm
        sync_actor_to_vllm(policy, llm)
    res, info = rollout(llm, spec, acts, a.group, a.max_new_tokens, a.temperature, seed=a.seed); print(f"[step0] rollout {info}", flush=True)
    del policy; torch.cuda.empty_cache()
    n = len(res); groups = torch.tensor([r["prompt_idx"] for r in res]); texts = [r["text"].strip() for r in res]; n_tok = torch.tensor([r["n_resp"] for r in res], dtype=torch.float32)
    viol = vc.check(texts, [r["full_ids"][r["prompt_len"]:].tolist() for r in res], [int(vp["pos_idx"][g]) for g in groups.tolist()], [[int(vp["next_token_id"][g])] for g in groups.tolist()])
    scorer = make_scorer(a, cdev)
    def score(txts, seed):
        return scorer.score(h_i[groups], h_j[groups], [z if z else None for z in txts], groups.tolist(), seed=seed)["exact_bits"].float()
    t0 = time.time(); b0 = score(texts, 0); t_score = time.time() - t0; b1 = score(texts, 1)
    # depth-matched shuffle: for each prompt, texts of another prompt with the same (i, j) (roll inside the (i,j) class); random-pair shuffle
    rng = np.random.default_rng(a.seed); perm_dm = np.arange(N); perm_rp = rng.permutation(N)
    for key, idx in vp.groupby(["i", "j"]).groups.items():
        idx = np.asarray(list(idx));
        if len(idx) > 1: perm_dm[idx] = np.roll(idx, 1)
    gl = groups.tolist(); by_prompt = {}
    for k, g in enumerate(gl): by_prompt.setdefault(g, []).append(k)
    def shuffled(perm):
        out = list(texts)
        for k, g in enumerate(gl):
            src = by_prompt[int(perm[g])]; out[k] = texts[src[k - by_prompt[g][0]] if k - by_prompt[g][0] < len(src) else src[0]]
        return out
    b_dm = score(shuffled(perm_dm), 0); b_rp = score(shuffled(perm_rp), 0)
    noise_row = (b0 - b1) / np.sqrt(2)
    def stats(mask):
        m = torch.as_tensor(mask); g = groups[m]
        if m.sum() == 0: return {}
        wg = within_group_std(b0[m], g); nz = float(noise_row[m].std()); bm = float(b0[m].mean()); dm = float(b_dm[m].mean()); rp = float(b_rp[m].mean())
        bpt = b0[m] / n_tok[m].clamp_min(1)
        return {"n": int(m.sum()), "bits_mean": bm, "bits_median": float(b0[m].median()), "bits_per_token_median": float(bpt.median()), "bits_per_token_mean": float(bpt.mean()),
                "lambda_max": 0.5 * float(bpt.median()), "within_group_std": wg, "scoring_noise": nz, "std_over_noise": wg / nz if nz > 0 else float("nan"),
                "bits_dm": dm, "bits_rp": rp, "bits_over_dm": bm / dm if dm > 0 else float("inf"), "frac_nonpos": float((b0[m] <= 0).float().mean()), "tokens_mean": float(n_tok[m].mean()),
                "mention_next": float(viol["mention_next"][mask].mean()), "copy_rate": float(viol["copy_rate"][mask].mean()), "regex": float(viol["regex"][mask].mean()), "empty": float(viol["empty"][mask].mean()),
                "corr_bits_tokens": corr(b0[m], n_tok[m]), "reward_mean": float((b0[m] - a.lam * n_tok[m]).mean())}
    out = {"init": a.init, "n_pairs": N, "group": a.group, "critic": a.critic or "stub", "ode_steps": a.ode_steps, "probes": a.probes, "score_s_per_row": t_score / n,
           "all": stats(np.ones(n, bool)), "bands": {b: stats(bands[gl] == b) for b in ("pre", "workspace", "motor")}, "rollout": info}
    ws = out["bands"].get("workspace", {})
    out["pass_workspace"] = bool(ws and ws["std_over_noise"] >= 3 and ws["bits_over_dm"] >= 3 and ws["mention_next"] < 0.2)
    samp = []
    for k in np.argsort(-b0.numpy())[:8].tolist() + np.argsort(b0.numpy())[:4].tolist():
        g = gl[k]; samp.append({"i": int(I[g]), "j": int(J[g]), "bits": float(b0[k]), "bits_dm": float(b_dm[k]), "tokens": int(n_tok[k]), "text": texts[k][:300]})
    out["samples"] = samp
    print(json.dumps({k: v for k, v in out.items() if k != "samples"}, indent=1), flush=True)
    for s in samp: print(f"  [{s['i']}->{s['j']}] bits {s['bits']:+.2f} (dm {s['bits_dm']:+.2f}) tok {s['tokens']}: {s['text']!r}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(out, open(a.out, "w"), indent=1); print(f"[step0] PASS(workspace)={out['pass_workspace']} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
