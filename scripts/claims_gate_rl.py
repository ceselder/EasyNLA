"""120-row stage-0 gates THROUGH the RL reward: FlowCritic.score_claims_composed(reward="singles_red") = sum_i single PMI(c_i) - R_LM(C) - cost*|C|
(nla.flow.claim_lm redundancy). Per benchmark row, one call scores (shared eps across the row's variants):
  true set T | shuffled = another row's true claims (same size) | T with one claim swapped for its minimal false twin (every pair) |
  T + 1 / 2 / 4 paraphrases of claims already in T | T minus the same 1 / 2 / 4 claims (distinct gain)
Greedy frontier k = 1..8 is exact from the returned singles + ClaimLM (the reward is additive in singles minus R_LM). Metrics at claim cost 0 and at
--costs (lambda table of the set reward: the share of rows whose best set beats the empty set, mean |argmax set|).
  python scripts/claims_gate_rl.py --adapter <path> --tag <tag>   -> /vol_glp/cond/compnla/gate_rl_<tag>.json"""
import argparse, json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
OUT = "/vol_glp/cond/compnla"; TS = (0.1, 0.3, 0.5, 0.7, 0.9)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--adapter", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--costs", default="10,20,30,40"); ap.add_argument("--lm", default="Qwen/Qwen3-8B-Base"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); dev = "cuda:0"; rng = np.random.default_rng(a.seed)
    import pyarrow.parquet as pq
    from nla.flow.rl_critic import FlowCritic
    from nla.flow.claims import format_claims
    from nla.flow.claim_lm import ClaimLM
    C = json.load(open(f"{OUT}/claims.json"))["items"]; PP = {x["row"]: x["paraphrases"] for x in json.load(open(f"{OUT}/paraphrases.json"))["items"]}
    if a.limit: C = C[: a.limit]
    t = pq.read_table("/vol_q36/data/sft/av_sft_val_clean1.parquet", columns=["activation_vector"]); N = t.num_rows
    acts = torch.tensor(np.asarray(t.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(N, -1))
    aa = torch.load(a.adapter, map_location="cpu")["args"]; pov = os.path.join(os.path.dirname(a.adapter), "prior_cotrained_latest.pt"); pov = pov if os.path.exists(pov) else None
    fc = FlowCritic(aa["prior"], a.adapter, aa["stats"], None, None, torch.device(dev), enc_layer=aa.get("enc_layer", 42), t_grid=TS, eps_per_t=a.D, train_adapter=False,
                    prior_override=pov, ar_ckpt=aa.get("ar_ckpt", "/vol/ckpts/qwen36_27b/ar_sft_merged"), enc_device=torch.device(dev))
    lm = ClaimLM(a.lm, dev); rows = []
    for n, it in enumerate(C):
        row = it["row"]; tc = [c["claim"] for c in it["true_claims"]]; fps = it["false_pairs"]; pp = PP.get(row) or []
        other = C[(n + 1 + int(rng.integers(len(C) - 1))) % len(C)]; oc = [c["claim"] for c in other["true_claims"]][: len(tc)]
        sets = {"T": tc, "O": oc}
        for q, p in enumerate(fps): sets[f"swap{q}"] = [c for j, c in enumerate(tc) if j != p["true_index"]] + [p["false_claim"]]
        pad_src = list(rng.permutation(min(len(tc), len(pp))))
        for k in (1, 2, 4):
            if k <= len(pad_src): sets[f"pad{k}"] = tc + [pp[j] for j in pad_src[:k]]; sets[f"drop{k}"] = [c for j, c in enumerate(tc) if j not in set(pad_src[:k])]
        names = list(sets); out = fc.score_claims_composed([format_claims(sets[k]) for k in names], [acts[row]] * len(names), [n] * len(names), seed=a.seed, cost=0.0,
                                                          claim_max=64, reward="singles_red", lm=lm)
        R = {k: out["reward"][i] for i, k in enumerate(names)}; sg = out["singles"][names.index("T")]
        order, rem, path = [], list(range(len(tc))), []                        # greedy frontier from the RL singles + ClaimLM (exact for singles_red)
        for _ in range(min(8, len(tc))):
            vals = [sum(sg[j] for j in order + [c]) - lm.redundancy([tc[j] for j in order + [c]]) for c in rem]; b = int(np.argmax(vals)); order.append(rem.pop(b)); path.append(vals[b])
        rec = {"row": row, "n_true": len(tc), "set": R["T"], "shuffled": R["O"], "swaps": [R[k] for k in names if k.startswith("swap")], "best_single": max(sg), "singles": sg, "greedy": path,
               "pad": {str(k): {"paraphrase_gain": R[f"pad{k}"] - R["T"], "distinct_gain": R["T"] - R[f"drop{k}"]} for k in (1, 2, 4) if f"pad{k}" in R}}
        rows.append(rec)
        if n % 10 == 0: print(f"[gate-rl {a.tag}] row {n + 1}/{len(C)}: set {rec['set']:.0f} best single {rec['best_single']:.0f} shuffled {rec['shuffled']:.0f} greedy {[round(x) for x in path]}", flush=True)
    gm = [float(np.mean([r["greedy"][k] for r in rows if len(r["greedy"]) > k])) for k in range(8)]
    s = {"adapter": a.adapter, "tag": a.tag, "D": a.D, "n_rows": len(rows), "reward": "singles_red (FlowCritic.score_claims_composed, claim cost 0)",
         "set_mean": float(np.mean([r["set"] for r in rows])), "best_single_mean": float(np.mean([r["best_single"] for r in rows])),
         "set_ge_best_single": float(np.mean([r["set"] >= r["best_single"] for r in rows])), "true_beats_shuffled": float(np.mean([r["set"] > r["shuffled"] for r in rows])),
         "true_beats_one_twin_swap": float(np.mean([r["set"] > x for r in rows for x in r["swaps"]])), "greedy_mean_by_k": gm,
         "median_single": float(np.median([x for r in rows for x in r["singles"]])), "padding": {}, "costs": {}}
    for k in ("1", "2", "4"):
        pg = [r["pad"][k]["paraphrase_gain"] for r in rows if k in r["pad"]]; dg = [r["pad"][k]["distinct_gain"] for r in rows if k in r["pad"]]
        if pg: s["padding"][k] = {"paraphrase_gain_mean": float(np.mean(pg)), "distinct_gain_mean": float(np.mean(dg)), "distinct_beats_paraphrase": float(np.mean([x > y for x, y in zip(dg, pg)])), "n": len(pg)}
    for lam in [float(x) for x in a.costs.split(",") if x]:
        best = [max([0.0] + [g - lam * (k + 1) for k, g in enumerate(r["greedy"])]) for r in rows]; kk = [int(np.argmax([0.0] + [g - lam * (k + 1) for k, g in enumerate(r["greedy"])])) for r in rows]
        s["costs"][str(lam)] = {"rows_with_nonempty_best_set": float(np.mean([x > 0 for x in best])), "mean_claims_in_best_set": float(np.mean(kk)), "mean_best_reward": float(np.mean(best))}
    os.makedirs(OUT, exist_ok=True); json.dump({"summary": s, "rows": rows}, open(f"{OUT}/gate_rl_{a.tag}.json", "w"), indent=1)
    print(json.dumps(s, indent=1), flush=True)


if __name__ == "__main__":
    main()
