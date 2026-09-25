"""Redundancy terms for the claims reward, compared on the 120-row stage-0 benchmark THROUGH the RL scorer (FlowCritic.score_claims_composed).

One critic pass per row scores every variant set with the SAME noise: true set T, shuffled (another row's claims), T with one claim swapped for its
minimal false twin (every pair), T + 1/2/4 paraphrases, T minus the same 1/2/4 claims. Saved per set: its claims, single-claim PMIs, composed PMI.
Set scores (all at claim cost 0; nla.flow.claim_redundancy is the shared implementation the RL reward uses):
  singles      sum_i v_i                                   (no redundancy term)
  neg_lm       -R_LM(C)                                    (the text-LM redundancy alone)
  lm / lm1.5 / lm2 / lm3   sum_i v_i - alpha * R_LM(C)
  semdup_emb / semdup_nli  sum_i v_i * (1 - max_{j<i} sim(c_i, c_j)), sim = embedding cosine / NLI entailment (max of both directions),
                           and *_t variants with sim rescaled above a floor (clip((sim - s0)/(1 - s0), 0, 1))
  min_composed min(sum_i v_i, PMI_composed(C))             (the critic's own redundancy; check only)
Metrics: set >= best single, true > shuffled, true > one-twin swap, paraphrase gain vs distinct gain at +1/+2/+4.
  python scripts/claims_redundancy_eval.py --adapter <path> --tag <tag>   -> /vol_glp/cond/compnla/redundancy_<tag>.json"""
import argparse, json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
OUT = "/vol_glp/cond/compnla"; TS = (0.1, 0.3, 0.5, 0.7, 0.9)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--adapter", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--lm", default="Qwen/Qwen3-8B-Base"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--emb", default="sentence-transformers/all-MiniLM-L6-v2"); ap.add_argument("--nli", default="cross-encoder/nli-deberta-v3-base")
    ap.add_argument("--floors", default="0.5,0.7")
    ap.add_argument("--from-json", default="", help="skip the critic pass: re-score the saved per-set singles of a previous run (redundancy_<tag>.json); output redundancy_<tag>_v2.json")
    a = ap.parse_args(); dev = "cuda:0"; rng = np.random.default_rng(a.seed)
    import pyarrow.parquet as pq
    from nla.flow.rl_critic import FlowCritic
    from nla.flow.claims import format_claims
    from nla.flow.claim_lm import ClaimLM
    from nla.flow.claim_redundancy import EmbSim, NLISim, LexSim, MaxSim, semdup_score
    C = json.load(open(f"{OUT}/claims.json"))["items"]; PP = {x["row"]: x["paraphrases"] for x in json.load(open(f"{OUT}/paraphrases.json"))["items"]}
    if a.limit: C = C[: a.limit]
    t = pq.read_table("/vol_q36/data/sft/av_sft_val_clean1.parquet", columns=["activation_vector"]); N = t.num_rows
    acts = torch.tensor(np.asarray(t.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(N, -1))
    aa = torch.load(a.adapter, map_location="cpu")["args"]; pov = os.path.join(os.path.dirname(a.adapter), "prior_cotrained_latest.pt"); pov = pov if os.path.exists(pov) else None
    lm = ClaimLM(a.lm, dev); emb = EmbSim(a.emb, dev); nli = NLISim(a.nli, dev); lex = LexSim(4)
    rows = json.load(open(a.from_json))["rows"] if a.from_json else []
    fc = None if a.from_json else FlowCritic(aa["prior"], a.adapter, aa["stats"], None, None, torch.device(dev), enc_layer=aa.get("enc_layer", 42), t_grid=TS, eps_per_t=a.D, train_adapter=False,
                                             prior_override=pov, ar_ckpt=aa.get("ar_ckpt", "/vol/ckpts/qwen36_27b/ar_sft_merged"), enc_device=torch.device(dev))
    for n, it in enumerate([] if a.from_json else C):                                                            # ---- critic pass: singles + composed PMI of every variant set
        row = it["row"]; tc = [c["claim"] for c in it["true_claims"]]; fps = it["false_pairs"]; pp = PP.get(row) or []
        other = C[(n + 1 + int(rng.integers(len(C) - 1))) % len(C)]; oc = [c["claim"] for c in other["true_claims"]][: len(tc)]
        sets = {"T": tc, "O": oc}
        for q, p in enumerate(fps): sets[f"swap{q}"] = [c for j, c in enumerate(tc) if j != p["true_index"]] + [p["false_claim"]]
        pad_src = list(rng.permutation(min(len(tc), len(pp))))
        for k in (1, 2, 4):
            if k <= len(pad_src): sets[f"pad{k}"] = tc + [pp[j] for j in pad_src[:k]]; sets[f"drop{k}"] = [c for j, c in enumerate(tc) if j not in set(pad_src[:k])]
        names = list(sets); out = fc.score_claims_composed([format_claims(sets[k]) for k in names], [acts[row]] * len(names), [n] * len(names), seed=a.seed, cost=0.0, claim_max=64, reward="set")
        rows.append({"row": row, "sets": {k: {"claims": out["claims"][i], "singles": out["singles"].get(i), "pmi_composed": out["pmi"][i]} for i, k in enumerate(names)}})
        if n % 20 == 0: print(f"[redundancy {a.tag}] critic pass row {n + 1}/{len(C)}", flush=True)
    floors = [float(x) for x in a.floors.split(",") if x]
    def scores(S):
        cl, v = S["claims"], S["singles"]; sv = sum(v); R = lm.redundancy(cl)
        o = {"singles": sv, "neg_lm": -R, "lm": sv - R, "lm1.5": sv - 1.5 * R, "lm2": sv - 2 * R, "lm3": sv - 3 * R, "min_composed": min(sv, S["pmi_composed"])}
        for nm, sim in (("emb", emb), ("nli", nli), ("lex", lex), ("max", MaxSim(nli, emb)), ("max3", MaxSim(nli, emb, lex))):
            o[f"semdup_{nm}"] = semdup_score(cl, v, sim)
            for s0 in floors: o[f"semdup_{nm}_t{s0:g}"] = semdup_score(cl, v, sim, floor=s0)
        return o
    for n, r in enumerate(rows):
        for k, S in r["sets"].items(): S["scores"] = scores(S)
        if n % 20 == 0: print(f"[redundancy {a.tag}] scoring row {n + 1}/{len(rows)}", flush=True)
    V = list(rows[0]["sets"]["T"]["scores"]); summ = {"adapter": a.adapter, "tag": a.tag, "n_rows": len(rows), "variants": {}}
    for var in V:
        sT = [r["sets"]["T"]["scores"][var] for r in rows]; best = [max(r["sets"]["T"]["singles"]) for r in rows]
        sw = [(r["sets"]["T"]["scores"][var], r["sets"][k]["scores"][var]) for r in rows for k in r["sets"] if k.startswith("swap")]
        s = {"set_mean": float(np.mean(sT)), "set_ge_best_single": float(np.mean([x >= b for x, b in zip(sT, best)])),
             "true_beats_shuffled": float(np.mean([r["sets"]["T"]["scores"][var] > r["sets"]["O"]["scores"][var] for r in rows])),
             "true_beats_one_twin_swap": float(np.mean([x > y for x, y in sw])), "padding": {}}
        for k in ("1", "2", "4"):
            pg = [r["sets"][f"pad{k}"]["scores"][var] - r["sets"]["T"]["scores"][var] for r in rows if f"pad{k}" in r["sets"]]
            dg = [r["sets"]["T"]["scores"][var] - r["sets"][f"drop{k}"]["scores"][var] for r in rows if f"drop{k}" in r["sets"]]
            if pg: s["padding"][k] = {"paraphrase_gain_mean": float(np.mean(pg)), "distinct_gain_mean": float(np.mean(dg)), "distinct_beats_paraphrase": float(np.mean([x > y for x, y in zip(dg, pg)])), "n": len(pg)}
        summ["variants"][var] = s
        p = s["padding"]; print(f"[redundancy {a.tag}] {var:18s} set>=best {s['set_ge_best_single']:.2f} >shuf {s['true_beats_shuffled']:.2f} >twin {s['true_beats_one_twin_swap']:.2f} | "
                                + " ".join(f"+{k}: para {p[k]['paraphrase_gain_mean']:+.0f} distinct {p[k]['distinct_gain_mean']:+.0f}" for k in p), flush=True)
    # twin-swap decomposition: which term moves when a true claim is replaced by its twin
    dsv = [r["sets"]["T"]["scores"]["singles"] - r["sets"][k]["scores"]["singles"] for r in rows for k in r["sets"] if k.startswith("swap")]
    dR = [r["sets"]["T"]["scores"]["neg_lm"] - r["sets"][k]["scores"]["neg_lm"] for r in rows for k in r["sets"] if k.startswith("swap")]
    summ["twin_swap_decomposition"] = {"singles_drop_mean": float(np.mean(dsv)), "singles_drop_median": float(np.median(dsv)), "neg_lm_drop_mean": float(np.mean(dR)), "neg_lm_drop_median": float(np.median(dR)),
                                       "share_R_LM_lower_for_swap": float(np.mean([x < 0 for x in dR]))}
    os.makedirs(OUT, exist_ok=True); json.dump({"summary": summ, "rows": rows}, open(f"{OUT}/redundancy_{a.tag}{'_v2' if a.from_json else ''}.json", "w"), indent=1)
    print(json.dumps(summ["twin_swap_decomposition"]), flush=True)


if __name__ == "__main__":
    main()
