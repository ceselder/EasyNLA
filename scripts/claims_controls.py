"""Controls for claim critics: are twin metrics driven by the activation or by claim-only cues? + the cue-immune primary metric.

(1) Twin detection with a WRONG-ACTIVATION control: P(PMI(h; true) > PMI(h; twin)) with the row's own h, and the same pair scored against h' of a
    different random row (should be 50 %; the excess over 50 % is claim-only cue use). On the 120-row stage-0 benchmark (Sonnet twins) and on
    synthetic held-out anchors (Gemma F-field / slot-swap twins), by claim type / family.
(2) CLAIM-ONLY baseline: Qwen3-8B-Base log p(claim) (total and per token, "Claims about a text:\\n• <claim>\\n") picks true vs twin without any
    activation: how guessable the twins are from text alone.
(3) SAME-TEMPLATE RETRIEVAL on held-out synthetic anchors (primary, immune to claim-only cues): for a template (family:type), N held-out
    activations with DISTINCT answers; score matrix PMI(h_i; c_j) (FM proxy, the same noise per activation across claims, PMI subtracts each
    activation's unconditional loss so columns compare); top-1 accuracy activation->claim (row) and claim->activation (column), N = 16 / 64 / 256
    (smaller N = random sub-blocks of the 256 x 256 matrix, averaged).
  python scripts/claims_controls.py --adapter <path> --tag <tag>   -> /vol_glp/cond/compnla/controls_<tag>.json"""
import argparse, json, math, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TS = [0.1, 0.3, 0.5, 0.7, 0.9]
OUT = "/vol_glp/cond/compnla"
HEAD = "Claims about a text:\n"


class Scorer:
    def __init__(self, fb, fmt, dev, D):
        self.fb, self.fmt, self.dev, self.D = fb, fmt, dev, D; self.tt = torch.tensor(TS, device=dev); self.T = len(TS)

    def noisy(self, x0, seed):   # x0 [1, d] -> xt [D*T, d], tv, tgt
        d = x0.shape[1]; E = torch.randn(self.D, d, device=self.dev, generator=torch.Generator(device=self.dev).manual_seed(seed))
        xt = ((1 - self.tt)[None, :, None] * x0[None] + self.tt[None, :, None] * E[:, None, :]).reshape(self.D * self.T, d)
        return xt, self.tt.repeat(self.D), (E[:, None, :] - x0[None]).expand(self.D, self.T, d).reshape(self.D * self.T, d)

    @torch.no_grad()
    def enc(self, claims):
        return self.fb.cond([self.fmt(c) for c in claims])

    @torch.no_grad()
    def pmi_matrix(self, X, claims, seeds, chunk_rows=12000):
        """X [n, d] standardised activations, claims [m] -> PMI [n, m] (nats), noise per activation fixed by seeds[i]"""
        fb = self.fb; d = X.shape[1]; n, m = X.shape[0], len(claims); enc, mk, cv = self.enc(claims); out = torch.zeros(n, m)
        R = self.D * self.T; per = max(1, chunk_rows // (R * m)) if R * m <= chunk_rows else 1
        for i0 in range(0, n, per):
            ii = list(range(i0, min(n, i0 + per))); XT, TV, TG, LU = [], [], [], []
            for i in ii:
                xt, tv, tgt = self.noisy(X[i:i + 1], seeds[i])
                with torch.autocast("cuda", dtype=torch.bfloat16): v0 = fb.model(xt, tv).float()
                XT.append(xt); TV.append(tv); TG.append(tgt); LU.append(((v0 - tgt) ** 2).mean(-1).mean())
            for j0 in range(0, m, max(1, chunk_rows // (R * len(ii)))):
                jj = list(range(j0, min(m, j0 + max(1, chunk_rows // (R * len(ii))))))
                xs = torch.cat([XT[k].repeat(len(jj), 1) for k in range(len(ii))]); ts = torch.cat([TV[k].repeat(len(jj)) for k in range(len(ii))])
                tg = torch.cat([TG[k].repeat(len(jj), 1) for k in range(len(ii))])
                sel = torch.tensor(jj, device=self.dev).repeat_interleave(R).repeat(len(ii))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    v = fb.model(xs, ts, enc[sel], mk[sel], cv[sel] if cv is not None else None).float()
                L = ((v - tg) ** 2).mean(-1).view(len(ii), len(jj), R).mean(-1)
                for k, i in enumerate(ii): out[i, jj] = ((d / 2) * (LU[k] - L[k])).cpu()
        return out


class LM:
    def __init__(self, name, dev):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name, token=os.environ.get("HF_TOKEN")); self.dev = dev
        self.m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, token=os.environ.get("HF_TOKEN")).to(dev).eval()
        self.nh = len(self.tok(HEAD, add_special_tokens=False)["input_ids"])

    @torch.no_grad()
    def logp(self, claims):
        """-> (total log p, per-token log p) of each claim line"""
        tot, per = [], []
        for i in range(0, len(claims), 64):
            ch = claims[i:i + 64]; enc = self.tok([HEAD + f"• {c}\n" for c in ch], return_tensors="pt", padding=True, add_special_tokens=False).to(self.dev)
            lp = torch.log_softmax(self.m(**enc).logits[:, :-1].float(), -1).gather(-1, enc["input_ids"][:, 1:, None])[..., 0]
            mask = enc["attention_mask"][:, 1:].clone(); mask[:, : self.nh - 1] = 0
            s = (lp * mask).sum(1); k = mask.sum(1).clamp_min(1); tot += s.tolist(); per += (s / k).tolist()
        return tot, per


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--adapter", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--D", type=int, default=4)
    ap.add_argument("--claims-glob", default="/vol_glp/claims/final/final_v2_0[0-4]*.parquet"); ap.add_argument("--n-val", type=int, default=3000)
    ap.add_argument("--templates", type=int, default=14); ap.add_argument("--Nmax", type=int, default=256); ap.add_argument("--lm", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--no-lm", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); dev = "cuda:0"; rng = np.random.default_rng(a.seed); t0 = time.time()
    import pyarrow.parquet as pq
    from nla.flow.scoring import FlowBundle
    from nla.flow.claims import format_claims
    from nla.flow.train_cond import load_claims_dir
    aa = torch.load(a.adapter, map_location="cpu")["args"]
    fb = FlowBundle(aa["prior"], a.adapter, aa["stats"], dev, base="Qwen/Qwen3.6-27B", enc_layer=aa.get("enc_layer", 42), ar_ckpt=aa.get("ar_ckpt", "/vol/ckpts/qwen36_27b/ar_sft_merged")); fb.model.eval()
    fmt = (lambda c: format_claims([c])) if aa.get("claim_subsets", 0) > 0 else (lambda c: c)
    sc = Scorer(fb, fmt, dev, a.D); res = {"adapter": a.adapter, "tag": a.tag, "D": a.D, "t_grid": TS}
    # ---------------- (1)+(2) benchmark twins
    C = json.load(open(f"{OUT}/claims.json"))["items"]
    t = pq.read_table("/vol_q36/data/sft/av_sft_val_clean1.parquet", columns=["activation_vector"]); N = t.num_rows
    acts = torch.tensor(np.asarray(t.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(N, -1))
    pairs = [(it["row"], it["true_claims"][p["true_index"]]["claim"], p["false_claim"], it["true_claims"][p["true_index"]]["type"]) for it in C for p in it["false_pairs"]]
    rows = sorted({r for r, *_ in pairs}); wrong = {r: rows[(k + 1 + int(rng.integers(len(rows) - 1))) % len(rows)] for k, r in enumerate(rows)}
    bench = []
    for r in rows:
        pr = [p for p in pairs if p[0] == r]; cl = [x for p in pr for x in (p[1], p[2])]
        X = fb.norm.normalize(torch.stack([acts[r], acts[wrong[r]]]).to(dev)).float()
        M = sc.pmi_matrix(X, cl, [1_000_003 + r, 1_000_003 + wrong[r]])
        for k, p in enumerate(pr): bench.append({"row": r, "type": p[3], "true": p[1], "twin": p[2], "own": [float(M[0, 2 * k]), float(M[0, 2 * k + 1])], "wrong": [float(M[1, 2 * k]), float(M[1, 2 * k + 1])]})
    print(f"[controls {a.tag}] benchmark twins scored ({len(bench)} pairs, {time.time() - t0:.0f}s)", flush=True)
    # ---------------- synthetic held-out anchors
    *_, val = load_claims_dir("/vol_glp/claims", 0, a.n_val, glob_pat=a.claims_glob)
    Xv = fb.norm.normalize(torch.stack([v[0] for v in val]).float().to(dev)).float(); nv = len(val)
    syn = []; wr = rng.permutation(nv)
    for i, v in enumerate(val[:1024]):
        tw = v[4] or [None] * len(v[1]); j = int(wr[i]) if int(wr[i]) != i else (i + 1) % nv
        for fam in ("internal", "text", "semantic"):
            cand = [(c, t_, x) for c, g, t_, x in zip(v[1], v[2], v[3], tw) if g == fam and x]
            if not cand: continue
            c, t_, x = cand[int(rng.integers(len(cand)))]
            M = sc.pmi_matrix(torch.stack([Xv[i], Xv[j]]), [c, x], [2_000_003 + i, 2_000_003 + j])
            syn.append({"i": i, "family": fam, "type": f"{fam}:{(t_ or '').split('/')[0]}", "true": c, "twin": x,
                        "own": [float(M[0, 0]), float(M[0, 1])], "wrong": [float(M[1, 0]), float(M[1, 1])]})
    print(f"[controls {a.tag}] synthetic twins scored ({len(syn)} pairs, {time.time() - t0:.0f}s)", flush=True)
    if not a.no_lm:
        lm = LM(a.lm, dev)
        for P in (bench, syn):
            tt_, tp_ = lm.logp([p["true"] for p in P]); ft_, fp_ = lm.logp([p["twin"] for p in P])
            for p, a1, a2, b1, b2 in zip(P, tt_, tp_, ft_, fp_): p["lm_total"] = [a1, b1]; p["lm_per_token"] = [a2, b2]
        del lm; torch.cuda.empty_cache()
    def summ(P, key):
        out = {}
        groups = {"all": P}
        for p in P: groups.setdefault(p[key], []).append(p)
        for g, ps in groups.items():
            o = {"n": len(ps), "own": float(np.mean([p["own"][0] > p["own"][1] for p in ps])), "wrong_activation": float(np.mean([p["wrong"][0] > p["wrong"][1] for p in ps]))}
            o["own_minus_wrong"] = o["own"] - o["wrong_activation"]
            if "lm_total" in ps[0]:
                o["lm_total"] = float(np.mean([p["lm_total"][0] > p["lm_total"][1] for p in ps])); o["lm_per_token"] = float(np.mean([p["lm_per_token"][0] > p["lm_per_token"][1] for p in ps]))
            out[g] = o
        return out
    res["benchmark_twins"] = summ(bench, "type"); res["synthetic_twins_by_family"] = summ(syn, "family"); res["synthetic_twins_by_type"] = summ(syn, "type")
    # ---------------- (3) same-template retrieval
    by = {}
    for i, v in enumerate(val):
        seen = set()
        for c, g, t_ in zip(v[1], v[2], v[3]):
            k = f"{g}:{(t_ or '').split('/')[0]}"
            if k in seen or (t_ or "").endswith("/paraphrase"): continue
            seen.add(k); by.setdefault(k, []).append((i, c))
    tmpls = []
    for k, lst in sorted(by.items(), key=lambda x: -len(x[1])):
        uniq, used = [], set()
        for i, c in lst:
            if c not in used: used.add(c); uniq.append((i, c))
        if len(uniq) >= 64: tmpls.append((k, uniq[: a.Nmax]))
        if len(tmpls) >= a.templates: break
    retr = {}
    for k, lst in tmpls:
        ii = [i for i, _ in lst]; cl = [c for _, c in lst]; M = sc.pmi_matrix(Xv[ii], cl, [3_000_003 + i for i in ii]).numpy(); n = len(ii)
        r = {"n_max": n}
        for Nsub in (16, 64, 256):
            if Nsub > n: continue
            accr, accc = [], []
            for _ in range(1 if Nsub == n else 20):
                s_ = rng.choice(n, Nsub, replace=False); S = M[np.ix_(s_, s_)]; ar = np.arange(Nsub)
                accr.append(float((S.argmax(1) == ar).mean())); accc.append(float((S.argmax(0) == ar).mean()))
            r[str(Nsub)] = {"act_to_claim": float(np.mean(accr)), "claim_to_act": float(np.mean(accc)), "chance": 1.0 / Nsub}
        retr[k] = r; print(f"[controls {a.tag}] retrieval {k}: " + " ".join(f"N={q} a->c {r[q]['act_to_claim']:.3f} c->a {r[q]['claim_to_act']:.3f}" for q in ("16", "64", "256") if q in r), flush=True)
    res["retrieval"] = retr
    for q in ("16", "64", "256"):
        v_ = [r[q] for r in retr.values() if q in r]
        if v_: res.setdefault("retrieval_mean", {})[q] = {"act_to_claim": float(np.mean([x["act_to_claim"] for x in v_])), "claim_to_act": float(np.mean([x["claim_to_act"] for x in v_])),
                                                           "chance": 1.0 / int(q), "n_templates": len(v_)}
    os.makedirs(OUT, exist_ok=True); json.dump({"summary": res, "benchmark_pairs": bench, "synthetic_pairs": syn}, open(f"{OUT}/controls_{a.tag}.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("benchmark_twins", "synthetic_twins_by_family", "retrieval_mean") if k in res}, indent=1), flush=True)


if __name__ == "__main__":
    main()
