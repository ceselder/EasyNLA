"""Composition alternatives for a SINGLE-CLAIM critic on the fixed 120-row stage-0 benchmark (eval only).

Every set score is built from the same cached per-claim velocity deltas (one conditional pass per claim, D noise draws x t grid, eps shared
by every condition of a row) and reported in nats:
  mean        v = v0 + (1/m) sum_i (v_i - v0)
  sum         v = v0 + sum_i (v_i - v0)
  lin         v = v0 + w(t) sum_i (v_i - v0),  w(t) = (1 - t) * 1 + t / m      (full sum at low noise, average at high noise)
  sw03, sw05  hard switch: w = 1 for t < t0, 1/m otherwise, t0 in {0.3, 0.5}
  singles_red sum_i PMI(h; c_i) - R(C),  R(C) = log p_LM(C) - sum_i log p_LM(c_i)   (text-LM total correlation of the claims;
              log p_LM(C) averaged over the given and the reversed order; LM = Qwen3-8B-Base over "Claims about a text:\\n• c1\\n• c2 ...")
Metrics per variant: full true set vs best single claim, greedy frontier k = 1..8, true set > shuffled set (another row's true claims, same
size), true set > the set with one claim swapped for its minimal false twin, PARAPHRASE PADDING (add 1 / 2 / 4 paraphrases of claims already in
the set; the score should not rise) next to the gain of the same number of DISTINCT true claims, and the per-row Spearman correlation of the
score with the number of distinct true claims (nested random order).
  python scripts/claims_compose_variants.py --adapter /vol_glp/cond/c1_synth_p2/adapter_latest.pt --tag c1_synth_p2"""
import argparse, json, math, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TS = [0.1, 0.3, 0.5, 0.7, 0.9]
OUT = "/vol_glp/cond/compnla"
VEL = {"mean": lambda t, m: 1.0 / m, "sum": lambda t, m: 1.0, "lin": lambda t, m: (1 - t) + t / m,
       "sw03": lambda t, m: 1.0 if t < 0.3 else 1.0 / m, "sw05": lambda t, m: 1.0 if t < 0.5 else 1.0 / m}
HEAD = "Claims about a text:\n"


class LM:
    def __init__(self, name, dev):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name, token=os.environ.get("HF_TOKEN")); self.dev = dev
        self.m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, token=os.environ.get("HF_TOKEN")).to(dev).eval()
        self.nh = len(self.tok(HEAD, add_special_tokens=False)["input_ids"]); self.cache = {}

    @torch.no_grad()
    def logp(self, sets):
        """sets: list of tuples of claims -> log p (nats) of the bullet lines after the header, cached"""
        todo = [s for s in dict.fromkeys(sets) if s not in self.cache]
        for i in range(0, len(todo), 48):
            ch = todo[i:i + 48]; texts = [HEAD + "".join(f"• {c}\n" for c in s) for s in ch]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.dev)
            lp = torch.log_softmax(self.m(**enc).logits[:, :-1].float(), -1).gather(-1, enc["input_ids"][:, 1:, None])[..., 0]
            mask = enc["attention_mask"][:, 1:].clone(); mask[:, : self.nh - 1] = 0
            for s, v in zip(ch, (lp * mask).sum(1).tolist()): self.cache[s] = v
        return [self.cache[s] for s in sets]


def spearman(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(x, y).correlation
    return float(r) if r == r else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--adapter", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--kmax", type=int, default=8); ap.add_argument("--chunk", type=int, default=8); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lm", default="Qwen/Qwen3-8B-Base"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); dev = "cuda:0"; rng = np.random.default_rng(a.seed)
    import pyarrow.parquet as pq
    from nla.flow.scoring import FlowBundle
    from nla.flow.claims import format_claims
    C = json.load(open(f"{OUT}/claims.json"))["items"]; PP = {x["row"]: x["paraphrases"] for x in json.load(open(f"{OUT}/paraphrases.json"))["items"]}
    if a.limit: C = C[: a.limit]
    t = pq.read_table("/vol_q36/data/sft/av_sft_val_clean1.parquet", columns=["activation_vector"]); N = t.num_rows
    acts = torch.tensor(np.asarray(t.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(N, -1))
    aa = torch.load(a.adapter, map_location="cpu")["args"]
    fb = FlowBundle(aa["prior"], a.adapter, aa["stats"], dev, base="Qwen/Qwen3.6-27B", enc_layer=aa.get("enc_layer", 42), ar_ckpt=aa.get("ar_ckpt", "/vol/ckpts/qwen36_27b/ar_sft_merged"), prior_override=(os.path.join(os.path.dirname(a.adapter), "prior_cotrained_latest.pt") if os.path.exists(os.path.join(os.path.dirname(a.adapter), "prior_cotrained_latest.pt")) else None))
    fb.model.eval(); lm = LM(a.lm, dev); d = acts.shape[1]; T = len(TS); tt = torch.tensor(TS, device=dev)
    bullet = aa.get("claim_subsets", 0) > 0
    variants = list(VEL) + ["singles_red"]; rows = []; t0 = time.time()
    for n, it in enumerate(C):
        row = it["row"]; x0 = fb.norm.normalize(acts[row][None].to(dev)).float()
        E = torch.randn(a.D, d, device=dev, generator=torch.Generator(device=dev).manual_seed(1_000_003 + row))
        xt = ((1 - tt)[None, :, None] * x0[None] + tt[None, :, None] * E[:, None, :]).reshape(a.D * T, d); tv = tt.repeat(a.D)
        tgt = (E[:, None, :] - x0[None]).expand(a.D, T, d).reshape(a.D * T, d)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): v0 = fb.model(xt, tv).float()
        Lu = ((v0 - tgt) ** 2).mean(-1)                                                                      # [D*T]
        tc = [c["claim"] for c in it["true_claims"]]; fps = it["false_pairs"]; pp = PP.get(row) or []
        other = C[(n + 1 + int(rng.integers(len(C) - 1))) % len(C)]; oc = [c["claim"] for c in other["true_claims"]][: len(tc)]
        pool = tc + [p["false_claim"] for p in fps] + oc + pp; nt, nf, no = len(tc), len(fps), len(oc)
        I_T = list(range(nt)); I_F = list(range(nt, nt + nf)); I_O = list(range(nt + nf, nt + nf + no)); I_P = list(range(nt + nf + no, len(pool)))
        Dl = []
        with torch.no_grad():
            for i in range(0, len(pool), a.chunk):
                cc = pool[i:i + a.chunk]; G = len(cc); enc, mk, cv = fb.cond([format_claims([c]) if bullet else c for c in cc]); R_ = xt.shape[0]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    v = fb.model(xt.repeat(G, 1), tv.repeat(G), enc.repeat_interleave(R_, 0), mk.repeat_interleave(R_, 0), cv.repeat_interleave(R_, 0) if cv is not None else None).float()
                Dl.append((v.view(G, R_, d) - v0[None]).half())
        Dl = torch.cat(Dl)                                                                                   # [pool, D*T, d] fp16
        def vel(idx, var):
            if not idx: return 0.0
            S = Dl[idx].float().sum(0); m = len(idx); w = torch.tensor([VEL[var](float(x), m) for x in tv.tolist()], device=dev)
            L = ((v0 + w[:, None] * S - tgt) ** 2).mean(-1); return float((d / 2) * (Lu - L).mean())
        single = [vel([j], "mean") for j in range(len(pool))]
        lps = dict(zip([(c,) for c in pool], lm.logp([(c,) for c in pool])))
        def red(idx):
            if len(idx) <= 1: return 0.0
            s = tuple(pool[j] for j in idx); j1, j2 = lm.logp([s, s[::-1]])
            return 0.5 * (j1 + j2) - sum(lps[(pool[j],)] for j in idx)
        def score(idx, var): return (sum(single[j] for j in idx) - red(idx)) if var == "singles_red" else vel(idx, var)
        best = max(single[j] for j in I_T); rec = {"row": row, "n_true": nt, "single_true": [single[j] for j in I_T], "best_single": best, "by": {}}
        order = list(rng.permutation(nt)); pad_src = list(rng.permutation(min(nt, len(I_P))))
        for var in variants:
            r = {"set": score(I_T, var), "shuffled": score(I_O, var) if I_O else float("nan"),
                 "swaps": [score([j for j in I_T if j != p["true_index"]] + [I_F[q]], var) for q, p in enumerate(fps)],
                 "nested": [score([I_T[j] for j in order[:k]], var) for k in range(1, nt + 1)]}
            ch, rem, path = [], list(I_T), []
            for k in range(min(a.kmax, nt)):
                vals = [score(ch + [c], var) for c in rem]; b = int(np.argmax(vals)); ch.append(rem.pop(b)); path.append(vals[b])
            r["greedy"] = path
            r["pad"] = {}
            for p in (1, 2, 4):
                if p > len(pad_src): continue
                add = [I_P[j] for j in pad_src[:p]]; drop = [I_T[j] for j in pad_src[:p]]
                r["pad"][str(p)] = {"paraphrase_gain": score(I_T + add, var) - r["set"], "distinct_gain": r["set"] - score([j for j in I_T if j not in drop], var)}
            r["spearman_k"] = spearman(list(range(1, nt + 1)), r["nested"])
            rec["by"][var] = r
        rows.append(rec)
        if n % 10 == 0:
            print(f"[variants {a.tag}] row {n + 1}/{len(C)} {time.time() - t0:.0f}s | best single {best:.0f} | " + " ".join(f"{v} {rec['by'][v]['set']:.0f}" for v in variants), flush=True)
    summ = {"adapter": a.adapter, "tag": a.tag, "n_rows": len(rows), "D": a.D, "t_grid": TS, "lm": a.lm, "best_single_mean": float(np.mean([r["best_single"] for r in rows])), "variants": {}}
    for var in variants:
        R = [r["by"][var] for r in rows]; K = min(a.kmax, min(len(r["greedy"]) for r in R))
        gm = [float(np.mean([r["greedy"][k] for r in R if len(r["greedy"]) > k])) for k in range(max(len(r["greedy"]) for r in R))]
        s = {"set_mean": float(np.mean([r["set"] for r in R])), "set_ge_best_single_frac": float(np.mean([r["set"] >= rw["best_single"] for r, rw in zip(R, rows)])),
             "shuffled_mean": float(np.nanmean([r["shuffled"] for r in R])), "true_beats_shuffled": float(np.nanmean([r["set"] > r["shuffled"] for r in R])),
             "true_beats_one_twin_swap": float(np.mean([r["set"] > x for r in R for x in r["swaps"]])),
             "greedy_mean_by_k": gm[: a.kmax], "greedy_nondecreasing": bool(all(gm[k + 1] >= gm[k] - 1e-9 for k in range(min(a.kmax, len(gm)) - 1))),
             "spearman_with_n_distinct_true_mean": float(np.mean([r["spearman_k"] for r in R])), "spearman_positive_frac": float(np.mean([r["spearman_k"] > 0 for r in R])),
             "padding": {}}
        for p in ("1", "2", "4"):
            pg = [r["pad"][p]["paraphrase_gain"] for r in R if p in r["pad"]]; dg = [r["pad"][p]["distinct_gain"] for r in R if p in r["pad"]]
            if pg: s["padding"][p] = {"paraphrase_gain_mean": float(np.mean(pg)), "paraphrase_gain_positive_frac": float(np.mean([x > 0 for x in pg])),
                                      "distinct_gain_mean": float(np.mean(dg)), "distinct_beats_paraphrase_frac": float(np.mean([x > y for x, y in zip(dg, pg)])), "n": len(pg)}
        summ["variants"][var] = s
    os.makedirs(OUT, exist_ok=True); json.dump({"summary": summ, "rows": rows}, open(f"{OUT}/compose_variants_{a.tag}.json", "w"), indent=1)
    print(json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()
