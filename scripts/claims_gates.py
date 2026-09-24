"""Stage-1 GATES for a claim conditioner, on the fixed stage-0 benchmark (/vol_glp/cond/compnla/claims.json: 120 clean1 held-out rows x ~10
Sonnet-written atomic true claims + 5 minimal false twins; scripts/claims_gen.py). Same estimator as scripts/claims_stage0.py: PMI proxy
(d/2) * mean_{t, draws}[L_uncond - L_cond] in nats, D noise draws x t in {0.1,0.3,0.5,0.7,0.9}, eps shared by every condition of a row.

Condition format follows the adapter: --set-encode adapters get concatenated per-claim memories; claim-set text adapters (--claim-subsets) get
"• c1\\n• c2 ..."; paragraph critics (sw_tokar, ...) get the claims joined by newlines (the stage-0 format).

--compose (default auto = on for single-claim critics, train_cond --claim-subsets 1): claims are composed in VELOCITY space,
v = v0 + w * sum_i (v(c_i) - v0), and every set quantity below is computed for w in {1/m (mean), m^-0.5 (sqrt), 1 (sum)} from the per-claim
velocity deltas (one conditional pass per claim): composed PMI vs set size (nested random order), true set vs the same set with one claim
swapped for its false twin, true set vs a SHUFFLED set (the claims of another row, same size), and the greedy frontier under each w.
Gates (compositionality-nla notes/DESIGN.md, 2026-09-23): paired detection >= 62 % overall and > 55 % on entity and on number_date; median
single-claim PMI > 0; greedy frontier non-decreasing on average (mean PMI after k claims does not fall with k); set PMI >= best single claim.
  python scripts/claims_gates.py --adapter /vol_glp/cond/<tag>/adapter_latest.pt --tag <tag>   -> /vol_glp/cond/compnla/gates_<tag>.json"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TS = [0.1, 0.3, 0.5, 0.7, 0.9]
OUT = "/vol_glp/cond/compnla"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--adapter", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--greedy-draws", type=int, default=4); ap.add_argument("--kmax", type=int, default=8); ap.add_argument("--chunk", type=int, default=8); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--compose", choices=["auto", "on", "off"], default="auto")
    a = ap.parse_args(); dev = "cuda:0"
    import pyarrow.parquet as pq
    from nla.flow.scoring import FlowBundle
    from nla.flow.claims import format_claims
    from nla.schema import extract_explanation
    C = json.load(open(f"{OUT}/claims.json"))["items"]
    if a.limit: C = C[: a.limit]
    t = pq.read_table("/vol_q36/data/sft/av_sft_val_clean1.parquet", columns=["activation_vector", "response"]); N = t.num_rows
    acts = torch.tensor(np.asarray(t.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(N, -1))
    gold = [extract_explanation(r) or r for r in t.column("response").to_pylist()]
    aa = torch.load(a.adapter, map_location="cpu")["args"]
    fb = FlowBundle(aa["prior"], a.adapter, aa["stats"], dev, base="Qwen/Qwen3.6-27B", enc_layer=aa.get("enc_layer", 42), ar_ckpt=aa.get("ar_ckpt", "/vol/ckpts/qwen36_27b/ar_sft_merged"), prior_override=(os.path.join(os.path.dirname(a.adapter), "prior_cotrained_latest.pt") if os.path.exists(os.path.join(os.path.dirname(a.adapter), "prior_cotrained_latest.pt")) else None))
    fb.model.eval(); d = acts.shape[1]; T = len(TS); tt = torch.tensor(TS, device=dev)
    mode = "set" if fb.set_encode else ("bullets" if aa.get("claim_subsets", 0) > 0 else "paragraph")
    compose = a.compose == "on" or (a.compose == "auto" and aa.get("claim_subsets", 0) == 1)
    if compose: return run_compose(a, fb, C, acts, gold, mode, dev)
    def cond(sets):
        if mode == "set": return fb.cond_sets(sets)
        return fb.cond([format_claims(s) if mode == "bullets" else "\n".join(s) for s in sets])
    def noisy(x0, E):
        Dn = E.shape[0]
        xt = ((1 - tt)[None, :, None] * x0[None] + tt[None, :, None] * E[:, None, :]).reshape(Dn * T, d)
        return xt, tt.repeat(Dn), (E[:, None, :] - x0[None]).expand(Dn, T, d).reshape(Dn * T, d)
    @torch.no_grad()
    def L(x0, E, sets):
        xt, tv, tgt = noisy(x0, E); R = xt.shape[0]
        if sets is None:
            with torch.autocast("cuda", dtype=torch.bfloat16): v = fb.model(xt, tv).float()
            return ((v - tgt) ** 2).mean(-1).view(1, E.shape[0], T)
        out = []
        for i in range(0, len(sets), a.chunk):
            ss = sets[i:i + a.chunk]; G = len(ss); enc, mk, cv = cond(ss)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = fb.model(xt.repeat(G, 1), tv.repeat(G), enc.repeat_interleave(R, 0), mk.repeat_interleave(R, 0), cv.repeat_interleave(R, 0) if cv is not None else None).float()
            out.append(((v - tgt.repeat(G, 1)) ** 2).mean(-1).view(G, E.shape[0], T))
        return torch.cat(out)
    pmi = lambda Lu, Lc: ((d / 2) * (Lu - Lc).mean(-1)).mean(-1).cpu().numpy()          # [n_sets] nats, averaged over draws

    rows, t0 = [], time.time()
    for n, it in enumerate(C):
        row = it["row"]; x0 = fb.norm.normalize(acts[row][None].to(dev)).float()
        E = torch.randn(a.D, d, device=dev, generator=torch.Generator(device=dev).manual_seed(1_000_003 + row)); Lu = L(x0, E, None)
        tc = [c["claim"] for c in it["true_claims"]]; ty = [c["type"] for c in it["true_claims"]]; fps = it["false_pairs"]
        sets = [[c] for c in tc] + [[p["false_claim"]] for p in fps] + [tc, [gold[row]]]
        P = pmi(Lu, L(x0, E, sets)); nt, nf = len(tc), len(fps)
        rec = {"row": row, "single": P[:nt].tolist(), "types": ty, "false": P[nt:nt + nf].tolist(), "true_index": [p["true_index"] for p in fps],
               "set": float(P[nt + nf]), "gold": float(P[nt + nf + 1])}
        Eg, Lug = E[: a.greedy_draws], Lu[:, : a.greedy_draws]; chosen, rem, path = [], list(range(nt)), []
        for k in range(min(a.kmax, nt)):
            Pg = pmi(Lug, L(x0, Eg, [[tc[j] for j in chosen + [c]] for c in rem])); b = int(np.argmax(Pg)); chosen.append(rem.pop(b)); path.append(float(Pg[b]))
        rec["greedy"] = path; rows.append(rec)
        if n % 20 == 0: print(f"[gates {a.tag}] row {n + 1}/{len(C)} {time.time() - t0:.0f}s | single median {np.median(P[:nt]):.1f} set {rec['set']:.1f} greedy {[round(x) for x in path]}", flush=True)
    # summary
    wins = {}; allw = []
    for r in rows:
        for fi, ti in enumerate(r["true_index"]):
            w = r["single"][ti] > r["false"][fi]; allw.append(w); wins.setdefault(r["types"][ti], []).append(w)
    singles = np.array([x for r in rows for x in r["single"]]); best = np.array([max(r["single"]) for r in rows]); setp = np.array([r["set"] for r in rows])
    K = max(len(r["greedy"]) for r in rows); gm = [float(np.mean([r["greedy"][k] for r in rows if len(r["greedy"]) > k])) for k in range(K)]
    by_type_single = {}
    for r in rows:
        for x, t_ in zip(r["single"], r["types"]): by_type_single.setdefault(t_, []).append(x)
    s = {"adapter": a.adapter, "tag": a.tag, "cond_format": mode, "n_rows": len(rows), "D": a.D,
         "paired_detection": float(np.mean(allw)), "paired_n": len(allw), "paired_by_type": {k: [float(np.mean(v)), len(v)] for k, v in wins.items()},
         "single_pmi_median": float(np.median(singles)), "single_pmi_mean": float(singles.mean()), "single_frac_negative": float((singles < 0).mean()),
         "single_median_by_type": {k: float(np.median(v)) for k, v in by_type_single.items()},
         "set_pmi_mean": float(setp.mean()), "best_single_mean": float(best.mean()), "set_ge_best_frac": float((setp >= best).mean()), "gold_pmi_mean": float(np.mean([r["gold"] for r in rows])),
         "greedy_mean_by_k": gm, "greedy_increments": [gm[k] - gm[k - 1] for k in range(1, K)]}
    ent, num = s["paired_by_type"].get("entity", [0, 0])[0], s["paired_by_type"].get("number_date", [0, 0])[0]
    s["gates"] = {"paired>=0.62": s["paired_detection"] >= 0.62, "entity>0.55": ent > 0.55, "number>0.55": num > 0.55, "median_single>0": s["single_pmi_median"] > 0,
                  "greedy_nondecreasing": float(np.mean(s["greedy_increments"])) >= 0 if s["greedy_increments"] else False, "set>=best_single": s["set_pmi_mean"] >= s["best_single_mean"]}
    s["all_gates_pass"] = all(s["gates"].values())
    os.makedirs(OUT, exist_ok=True); json.dump({"summary": s, "rows": rows}, open(f"{OUT}/gates_{a.tag}.json", "w"), indent=1)
    print(json.dumps(s, indent=1), flush=True)


WS = {"mean": lambda m: 1.0 / m, "sqrt": lambda m: m ** -0.5, "sum": lambda m: 1.0}


def run_compose(a, fb, C, acts, gold, mode, dev):
    from nla.flow.claims import format_claims
    d = acts.shape[1]; T = len(TS); tt = torch.tensor(TS, device=dev)
    def enc1(claims):   # one claim per condition, in the critic's training format
        if mode == "set": return fb.cond_sets([[c] for c in claims])
        return fb.cond([format_claims([c]) if mode == "bullets" else c for c in claims])
    @torch.no_grad()
    def deltas(xt, tv, v0, claims):
        R = xt.shape[0]; out = []
        for i in range(0, len(claims), a.chunk):
            cc = claims[i:i + a.chunk]; G = len(cc); enc, mk, cv = enc1(cc)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = fb.model(xt.repeat(G, 1), tv.repeat(G), enc.repeat_interleave(R, 0), mk.repeat_interleave(R, 0), cv.repeat_interleave(R, 0) if cv is not None else None).float()
            out.append(v.view(G, R, d) - v0[None])
        return torch.cat(out)                                                         # [n_claims, R, d]
    rows, t0 = [], time.time(); rng = np.random.default_rng(0)
    for n, it in enumerate(C):
        row = it["row"]; x0 = fb.norm.normalize(acts[row][None].to(dev)).float()
        E = torch.randn(a.D, d, device=dev, generator=torch.Generator(device=dev).manual_seed(1_000_003 + row))
        xt = ((1 - tt)[None, :, None] * x0[None] + tt[None, :, None] * E[:, None, :]).reshape(a.D * T, d); tv = tt.repeat(a.D)
        tgt = (E[:, None, :] - x0[None]).expand(a.D, T, d).reshape(a.D * T, d)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): v0 = fb.model(xt, tv).float()
        Lu = ((v0 - tgt) ** 2).mean(-1).mean()
        tc = [c["claim"] for c in it["true_claims"]]; ty = [c["type"] for c in it["true_claims"]]; fps = it["false_pairs"]; nt = len(tc)
        other = C[(n + 1 + int(rng.integers(len(C) - 1))) % len(C)]; oc = [c["claim"] for c in other["true_claims"]][:nt]
        Dl = deltas(xt, tv, v0, tc + [p["false_claim"] for p in fps] + oc)
        Dt, Df, Do = Dl[:nt], Dl[nt:nt + len(fps)], Dl[nt + len(fps):]
        def P(S, w):   # composed PMI (nats) of a stack of deltas [m, R, d]
            v = v0 + WS[w](S.shape[0]) * S.sum(0); return float((d / 2) * (Lu - ((v - tgt) ** 2).mean(-1).mean()))
        single = [P(Dt[j:j + 1], "sum") for j in range(nt)]; false = [P(Df[j:j + 1], "sum") for j in range(len(fps))]
        order = list(rng.permutation(nt)); rec = {"row": row, "types": ty, "single": single, "false": false, "true_index": [p["true_index"] for p in fps]}
        for w in WS:
            rec[f"nested_{w}"] = [P(Dt[order[:k]], w) for k in range(1, nt + 1)]
            rec[f"set_{w}"] = P(Dt, w); rec[f"shuffled_{w}"] = P(Do, w) if len(oc) else float("nan")
            rec[f"swap_{w}"] = [P(torch.cat([Dt[[j for j in range(nt) if j != p["true_index"]]], Df[q:q + 1]]), w) for q, p in enumerate(fps)]
            ch, rem, path = [], list(range(nt)), []
            for k in range(min(a.kmax, nt)):
                vals = [P(Dt[ch + [c]], w) for c in rem]; b = int(np.argmax(vals)); ch.append(rem.pop(b)); path.append(vals[b])
            rec[f"greedy_{w}"] = path
        rows.append(rec)
        if n % 20 == 0: print(f"[gates-compose {a.tag}] row {n + 1}/{len(C)} {time.time() - t0:.0f}s | single median {np.median(single):.1f} | set mean/sqrt/sum "
                              f"{rec['set_mean']:.1f}/{rec['set_sqrt']:.1f}/{rec['set_sum']:.1f} | shuffled {rec['shuffled_mean']:.1f}/{rec['shuffled_sqrt']:.1f}/{rec['shuffled_sum']:.1f}", flush=True)
    wins = {}; allw = []
    for r in rows:
        for fi, ti in enumerate(r["true_index"]):
            w_ = r["single"][ti] > r["false"][fi]; allw.append(w_); wins.setdefault(r["types"][ti], []).append(w_)
    singles = np.array([x for r in rows for x in r["single"]]); best = np.array([max(r["single"]) for r in rows])
    s = {"adapter": a.adapter, "tag": a.tag, "cond_format": mode, "compose": True, "n_rows": len(rows), "D": a.D,
         "paired_detection": float(np.mean(allw)), "paired_n": len(allw), "paired_by_type": {k: [float(np.mean(v)), len(v)] for k, v in wins.items()},
         "single_pmi_median": float(np.median(singles)), "single_pmi_mean": float(singles.mean()), "single_frac_negative": float((singles < 0).mean()),
         "best_single_mean": float(best.mean()), "by_w": {}}
    for w in WS:
        K = max(len(r[f"nested_{w}"]) for r in rows)
        nest = [float(np.mean([r[f"nested_{w}"][k] for r in rows if len(r[f"nested_{w}"]) > k])) for k in range(K)]
        gm = [float(np.mean([r[f"greedy_{w}"][k] for r in rows if len(r[f"greedy_{w}"]) > k])) for k in range(min(a.kmax, K))]
        setp = np.array([r[f"set_{w}"] for r in rows]); shuf = np.array([r[f"shuffled_{w}"] for r in rows])
        sw = [(r[f"set_{w}"] > x) for r in rows for x in r[f"swap_{w}"]]
        s["by_w"][w] = {"set_pmi_mean": float(setp.mean()), "shuffled_pmi_mean": float(np.nanmean(shuf)), "true_beats_shuffled": float(np.nanmean(setp > shuf)),
                        "true_beats_one_twin_swap": float(np.mean(sw)), "set_ge_best_single_frac": float((setp >= best).mean()),
                        "nested_mean_by_k": nest, "greedy_mean_by_k": gm, "greedy_increments": [gm[k] - gm[k - 1] for k in range(1, len(gm))]}
    ent, num = s["paired_by_type"].get("entity", [0, 0])[0], s["paired_by_type"].get("number_date", [0, 0])[0]
    bm = s["by_w"]["mean"]
    s["gates"] = {"paired>=0.62": s["paired_detection"] >= 0.62, "entity>0.55": ent > 0.55, "number>0.55": num > 0.55, "median_single>0": s["single_pmi_median"] > 0,
                  "greedy_nondecreasing(mean)": float(np.mean(bm["greedy_increments"])) >= 0 if bm["greedy_increments"] else False,
                  "set>=best_single(mean)": bm["set_pmi_mean"] >= s["best_single_mean"]}
    s["all_gates_pass"] = all(s["gates"].values())
    os.makedirs(OUT, exist_ok=True); json.dump({"summary": s, "rows": rows}, open(f"{OUT}/gates_{a.tag}.json", "w"), indent=1)
    print(json.dumps(s, indent=1), flush=True)


if __name__ == "__main__":
    main()
