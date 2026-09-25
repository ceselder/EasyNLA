"""Validate duplicate-discount rules on real rollouts (score_claims_peek.py output: claims + single-claim PMIs per row).

Per rollout: reward = sum_i v_i (1 - d_i) - cost * n under sim in {nli, max(nli, emb), max(nli, emb, lex)} (floor 0.5, nla.flow.claim_redundancy),
the quote-repetition rate, and for rollouts that re-quote a >= 4-word span: the reward of the deduped version (first quote-bullet + all other
bullets). Also the per-bullet discount factor (1 - d_i) distribution (for normal SFT samples: how much diverse bullets are charged).
  python scripts/validate_dedup.py --scored <peek_scored.json> [--cost 6.93] [--device cpu] --out <json>"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scored", required=True); ap.add_argument("--cost", type=float, default=6.93); ap.add_argument("--device", default="cpu")
    ap.add_argument("--floor", type=float, default=0.5); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from nla.flow.claim_redundancy import EmbSim, NLISim, LexSim, MaxSim, semdup_score, _discounts, quote_rep_rate, dedup_quotes
    S = json.load(open(a.scored)); rows = {}
    for r in S["all"]: rows.setdefault(r["row"], []).append((r["claim"], r["pmi"]))
    nli = NLISim(device=a.device); emb = EmbSim(device=a.device); lex = LexSim(4)
    sims = {"nli": nli, "max_nli_emb": MaxSim(nli, emb), "max3_nli_emb_lex": MaxSim(nli, emb, lex)}
    allc = [[c for c, _ in v] for v in rows.values()]
    for s_ in (nli, emb): s_.prefetch(allc + [dedup_quotes(c) for c in allc])
    res = {"n_rollouts": len(rows), "cost": a.cost, "floor": a.floor, "quote_rep_rate": quote_rep_rate(allc), "sims": {}}
    for nm, sim in sims.items():
        rw, dd, fac, ex = [], [], [], 0
        for cl, v in zip(allc, [[p for _, p in vv] for vv in rows.values()]):
            val = semdup_score(cl, v, sim, a.floor); rw.append(val - a.cost * len(cl))
            fac += [1 - d for d in _discounts(cl, sim, a.floor)]
            dq = dedup_quotes(cl)
            if len(dq) < len(cl):
                ex += 1; pm = dict(zip(cl, v)); vd = [pm[c] for c in dq]
                dd.append((semdup_score(dq, vd, sim, a.floor) - a.cost * len(dq)) >= (val - a.cost * len(cl)) - 1e-9)
        res["sims"][nm] = {"reward_mean": float(np.mean(rw)), "discount_factor_median": float(np.median(fac)), "discount_factor_mean": float(np.mean(fac)),
                           "share_bullets_discounted_gt_20pct": float(np.mean([f < 0.8 for f in fac])), "exploit_rollouts": ex,
                           "deduped_ge_exploit": float(np.mean(dd)) if dd else None}
        print(f"[dedup] {nm:18s} reward mean {np.mean(rw):8.1f} | discount factor median {np.median(fac):.3f} mean {np.mean(fac):.3f}, bullets discounted > 20% {np.mean([f < 0.8 for f in fac]):.2f} | "
              f"exploit rollouts {ex}, deduped >= exploit {res['sims'][nm]['deduped_ge_exploit']}", flush=True)
    print(f"[dedup] quote-repetition rate {res['quote_rep_rate']:.3f} over {len(rows)} rollouts", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
