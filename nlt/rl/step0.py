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
from nlt.evals.common import band
from nlt.rl.reward import make_scorer, within_group_std, corr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--out", required=True); p.add_argument("--tag", default="step0")
    p.add_argument("--base", default="Qwen/Qwen3-8B"); p.add_argument("--init", default="ao"); p.add_argument("--question", default=None)
    p.add_argument("--critic", default=None); p.add_argument("--stub-critic", action="store_true"); p.add_argument("--ode-steps", type=int, default=32)
    p.add_argument("--probes", type=int, default=1); p.add_argument("--score-batch", type=int, default=64); p.add_argument("--enc-model", default=None); p.add_argument("--enc-layer", type=int, default=None)
    p.add_argument("--n-pairs", type=int, default=512); p.add_argument("--group", type=int, default=8); p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0); p.add_argument("--vllm-gpu-mem", type=float, default=0.40); p.add_argument("--vllm-max-len", type=int, default=512)
    p.add_argument("--lam", type=float, default=0.1); p.add_argument("--seed", type=int, default=0); p.add_argument("--skip", type=int, default=0, help="skip the first rows of pairs_val")
    p.add_argument("--dump-dir", default=None, help="write the rollouts in the board #31 format here (e.g. /vol/z/ref_v1_0/val)")
    a = p.parse_args(); torch.manual_seed(a.seed)
    import pyarrow.parquet as pq
    from nlt.data.dataset import ActStore
    from nlt.verbalizer.prompt import build_prompt, DEFAULT_QUESTION
    from nlt.verbalizer.model import load_tokenizer, load_policy
    from nlt.verbalizer.vllm_rollout import make_engine, rollout
    from nlt.rl.filters import ViolationChecker
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
    del policy, llm; torch.cuda.empty_cache()
    n = len(res); groups = torch.tensor([r["prompt_idx"] for r in res]); texts = [r["text"].strip() for r in res]; n_tok = torch.tensor([r["n_resp"] for r in res], dtype=torch.float32)
    viol = vc.check(texts, [r["full_ids"][r["prompt_len"]:].tolist() for r in res], [int(vp["pos_idx"][g]) for g in groups.tolist()], next_words=[tok.decode([int(vp["next_token_id"][g])]) for g in groups.tolist()])
    if a.dump_dir:                                        # the policy's own samples on the fixed pairs, board #31 format (redteam Y2 bar = the step-0 dump)
        import pyarrow as pa
        os.makedirs(a.dump_dir, exist_ok=True); src = os.path.basename(os.path.dirname(a.dump_dir.rstrip("/"))) or a.tag
        pq.write_table(pa.table({"pair_id": [vp["pair_id"][g] for g in groups.tolist()], "text": texts, "n_tokens": pa.array([int(x) for x in n_tok.tolist()], pa.int32()), "verbosity": pa.array([1] * n, pa.int32()),
                                 "source": [src] * n, "sample_idx": pa.array([r["group_idx"] for r in res], pa.int32())}), os.path.join(a.dump_dir, f"part_0000000_{N:07d}.parquet"))
        print(f"[step0] dumped {n} rollouts -> {a.dump_dir}", flush=True)
    critics = [(c.split(":", 1) if ":" in c else (os.path.basename(os.path.dirname(c)), c)) for c in (a.critic.split(",") if a.critic else ["stub:stub"])]
    all_out = {}
    for cname, cpath in critics:
        a.critic = None if cpath == "stub" else cpath
        scorer = make_scorer(a, cdev)
        out = _evaluate(a, scorer, h_i, h_j, groups, texts, n_tok, viol, vp, I, J, N, bands, info, cname)
        all_out[cname] = out; print(f"[step0] {cname}: PASS(ws)={out['pass_workspace']} signal={out['signal_workspace']} healthy(ws)={out['critic_healthy_workspace']} content_ws={out['bands']['workspace'].get('bits_minus_dm', float('nan')):+.3f} P(z>dm)_ws={out['bands']['workspace'].get('p_own_gt_dm', float('nan')):.3f} lambda_content={out.get('lambda_content')}", flush=True)
        del scorer; torch.cuda.empty_cache()
    final = all_out[critics[0][0]] if len(critics) == 1 else {"init": a.init, "n_pairs": N, "group": a.group, "critics": all_out, "rollout": info}
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(final, open(a.out, "w"), indent=1); print(f"[step0] -> {a.out}", flush=True)


def _evaluate(a, scorer, h_i, h_j, groups, texts, n_tok, viol, vp, I, J, N, bands, info, cname):
    n = len(texts)
    def score(txts, seed):
        return scorer.score(h_i[groups], h_j[groups], [z if z else None for z in txts], groups.tolist(), seed=seed)["exact_bits"].float()
    t0 = time.time(); b0 = score(texts, 0); t_score = time.time() - t0; b1 = score(texts, 1)
    # depth-matched shuffle: for each prompt, texts of another prompt with the same (i, j) (roll inside the (i,j) class); random-pair shuffle
    rng = np.random.default_rng(a.seed); perm_dm = np.arange(N); perm_rp = rng.permutation(N)
    # depth-matched partner: another prompt with the same (i, j); singleton classes fall back to the nearest class (same j, closest i;
    # then same gap, closest j) so no prompt is ever paired with itself
    ii = vp["i"].values.astype(int); jj = vp["j"].values.astype(int)
    for key, idx in vp.groupby(["i", "j"]).groups.items():
        idx = np.asarray(list(idx))
        if len(idx) > 1: perm_dm[idx] = np.roll(idx, 1)
    n_approx = 0
    for g in range(N):
        if perm_dm[g] != g: continue
        cand = np.where((jj == jj[g]) & (np.arange(N) != g))[0]
        if len(cand) == 0: cand = np.where(((jj - ii) == (jj[g] - ii[g])) & (np.arange(N) != g))[0]
        if len(cand) == 0: cand = np.where(np.arange(N) != g)[0]
        dist = np.abs(ii[cand] - ii[g]) * 1000 + np.abs(jj[cand] - jj[g]); perm_dm[g] = int(cand[np.argmin(dist)]); n_approx += 1
    print(f"[step0] depth-matched partners: {N - n_approx} exact (same i,j), {n_approx} nearest-class", flush=True)
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
        bpt = b0[m] / n_tok[m].clamp_min(1); nn = int(m.sum())
        d_dm = (b0[m] - b_dm[m]); d_rp = (b0[m] - b_rp[m]); wg_tok = within_group_std(n_tok[m], g); p_dm = float((d_dm > 0).float().mean()); p_rp = float((d_rp > 0).float().mean())
        return {"n": nn, "bits_mean": bm, "bits_median": float(b0[m].median()), "bits_per_token_median": float(bpt.median()), "bits_per_token_mean": float(bpt.mean()),
                "lambda_max": 0.5 * float(bpt.median()), "lambda_wg": 0.25 * wg / wg_tok if wg_tok > 0 else float("nan"),
                "lambda_content": 0.25 * max(0.0, float(d_dm.mean())) / wg_tok if wg_tok > 0 else float("nan"),   # scaled by the CONTENT signal (bits - z_dm), not the critic's spread
                "within_group_std": wg, "within_group_std_tokens": wg_tok,
                "scoring_noise": nz, "std_over_noise": wg / nz if nz > 0 else float("nan"),
                "bits_dm": dm, "bits_rp": rp, "bits_minus_dm": float(d_dm.mean()), "bits_minus_dm_sem": float(d_dm.std() / nn ** 0.5), "bits_minus_rp": float(d_rp.mean()), "bits_minus_rp_sem": float(d_rp.std() / nn ** 0.5),
                "p_own_gt_dm": p_dm, "p_own_gt_rp": p_rp, "p_own_gt_null": float((b0[m] > 0).float().mean()),
                "bits_over_dm": bm / dm if dm > 0 else float("inf"), "critic_presence_offset_over_noise": abs(rp) / nz if nz > 0 else float("nan"),
                "frac_nonpos": float((b0[m] <= 0).float().mean()), "tokens_mean": float(n_tok[m].mean()),
                "mention_next": float(viol["mention_next"][mask].mean()), "copy_rate": float(viol["copy_rate"][mask].mean()), "regex": float(viol["regex"][mask].mean()), "empty": float(viol["empty"][mask].mean()), "junk": float(viol["junk"][mask].mean()),
                "corr_bits_tokens": corr(b0[m], n_tok[m]), "reward_mean": float((b0[m] - a.lam * n_tok[m]).mean())}
    out = {"init": a.init, "n_pairs": N, "group": a.group, "critic": a.critic or "stub", "dm_exact_partners": int(N - n_approx), "ode_steps": a.ode_steps, "probes": a.probes, "score_s_per_row": t_score / n,
           "all": stats(np.ones(n, bool)), "bands": {b: stats(bands[gl] == b) for b in ("pre", "workspace", "motor")}, "rollout": info}
    ws = out["bands"].get("workspace", {})
    # critic health (DECISIONS v1.5): per band with n >= 100 rows, |bits(z_rp)| AND |bits(z_dm)| within 3x the scoring noise (text about
    # another pair must buy ~nothing); signal: within-group std >= 3x noise AND paired gain over the depth-matched shuffle > 3 SEM.
    health = {}
    for bname, st in out["bands"].items():
        if st and st["n"] >= 100 and st["scoring_noise"] > 0:
            health[bname] = {"rp_over_noise": abs(st["bits_rp"]) / st["scoring_noise"], "dm_over_noise": abs(st["bits_dm"]) / st["scoring_noise"]}
            health[bname]["ok"] = bool(health[bname]["rp_over_noise"] <= 3 and health[bname]["dm_over_noise"] <= 3)
    out["critic_health"] = health; out["critic_healthy_workspace"] = bool(health.get("workspace", {}).get("ok", False)); out["critic_healthy_all"] = bool(health) and all(h["ok"] for h in health.values())
    out["signal_workspace"] = bool(ws and ws["std_over_noise"] >= 3 and ws["bits_minus_dm"] > 3 * ws["bits_minus_dm_sem"] and ws["mention_next"] < 0.2)
    out["pass_workspace"] = bool(out["signal_workspace"] and out["critic_healthy_workspace"])
    out["lambda_recommended"] = ws.get("lambda_wg") if ws else None            # DECISIONS v1.5: 0.25 x wg std(bits) / wg std(tokens), WORKSPACE band
    out["lambda_content"] = ws.get("lambda_content") if ws else None            # 0.25 x (bits - z_dm) / wg std(tokens): a length difference costs a quarter of the CONTENT signal
    samp = []
    for k in np.argsort(-b0.numpy())[:8].tolist() + np.argsort(b0.numpy())[:4].tolist():
        g = gl[k]; samp.append({"i": int(I[g]), "j": int(J[g]), "bits": float(b0[k]), "bits_dm": float(b_dm[k]), "tokens": int(n_tok[k]), "text": texts[k][:300]})
    out["samples"] = samp; out["critic_name"] = cname
    print(json.dumps({k: v for k, v in out.items() if k != "samples"}, indent=1), flush=True)
    for s in samp: print(f"  [{s['i']}->{s['j']}] bits {s['bits']:+.2f} (dm {s['bits_dm']:+.2f}) tok {s['tokens']}: {s['text']!r}", flush=True)
    return out


if __name__ == "__main__":
    main()
