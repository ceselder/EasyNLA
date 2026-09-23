"""Stage-1 evaluation of a claim-SET conditioner: is the joint condition better than composing single claims?

On N held-out rows (clean1 val: gold explanation split into claims; optionally synthetic val anchors from --claims-dir), EXACT
log p(h | .) - log p(h) in bits (probability-flow ODE, Hutchinson, same probes for every variant of a row) for
  joint     : the claim set as ONE condition ("• c1\\n• c2 ...", first --k claims)       <- what the stage-2 reward uses
  compose_1 : velocity composition v = v0 + sum_i (v(c_i) - v0)        (w = 1, product of experts)
  compose_m : velocity composition v = v0 + (1/m) sum_i (v(c_i) - v0)  (w = 1/m, averaging)
  shuffled  : the claim set of another row (mismatch control)
  sum_single: sum_i PMI(h; c_i) of the single-claim conditions (the likelihood-domain independence approximation)
  gold      : the raw gold paragraph (the claims unsplit)
  -> {out}/compose_<tag>.json (per-row values + means/sems)"""
import argparse, json, math, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nla.flow.claims import split_claims, format_claims


class Composed(torch.nn.Module):
    """velocity composition of per-claim conditions: v = v0 + w_b * sum_{i in b} (v(x_b, t, c_i) - v0(x_b, t))"""
    def __init__(self, fb, claim_sets, w_mode):
        super().__init__(); self.fb = fb; flat = [c for s in claim_sets for c in s]
        self.owner = torch.tensor([b for b, s in enumerate(claim_sets) for _ in s], device=fb.dev)
        self.enc, self.mk, self.cv = fb.cond_sets([[c] for c in flat]) if fb.set_encode else fb.cond([format_claims([c]) for c in flat])
        m = torch.tensor([len(s) for s in claim_sets], device=fb.dev, dtype=torch.float32)
        self.w = (1.0 / m) if w_mode == "mean" else torch.ones_like(m)

    def forward(self, x, t, *_):
        v0 = self.fb.model(x, t); xi, ti = x[self.owner], t[self.owner]
        vi = self.fb.model(xi, ti, self.enc, self.mk, self.cv)
        delta = torch.zeros_like(v0).index_add_(0, self.owner, (vi - v0[self.owner]).to(v0.dtype))
        return v0 + self.w[:, None].to(v0.dtype) * delta


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--adapter", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--k", type=int, default=8); ap.add_argument("--steps", type=int, default=32); ap.add_argument("--xb", type=int, default=16)
    ap.add_argument("--val-parquet", default="/vol_q36/data/sft/av_sft_val_clean1.parquet"); ap.add_argument("--claims-dir", default=None)
    ap.add_argument("--out", default="/vol_glp/cond/compose"); ap.add_argument("--no-sum-single", action="store_true")
    a = ap.parse_args(); dev = "cuda"
    import pyarrow.parquet as pq
    from nla.flow.scoring import FlowBundle
    from nla.flow.eval_cond import exact_logp
    from nla.schema import extract_explanation
    sets = {}
    t = pq.read_table(a.val_parquet, columns=["activation_vector", "response"]).slice(0, a.n)
    acts = torch.tensor(np.stack(t.column("activation_vector").to_pylist()), dtype=torch.float32)
    gold = [extract_explanation(r) or r for r in t.column("response").to_pylist()]
    sets["gold_split"] = (acts, [split_claims(z)[: a.k] or [z] for z in gold], gold)
    if a.claims_dir:
        from nla.flow.train_cond import load_claims_dir
        *_, val = load_claims_dir(a.claims_dir, 0, a.n); rng = np.random.default_rng(0)
        sets["synthetic"] = (torch.stack([v[0] for v in val]).float(), [list(rng.permutation(v[1])[: a.k]) for v in val], None)
    aa = torch.load(a.adapter, map_location="cpu")["args"]
    fb = FlowBundle(aa["prior"], a.adapter, aa["stats"], dev, base="Qwen/Qwen3.6-27B", enc_layer=aa.get("enc_layer", 42), ar_ckpt=aa.get("ar_ckpt", "/vol/ckpts/qwen36_27b/ar_sft_merged"))
    res = {"adapter": a.adapter, "tag": a.tag, "k": a.k, "ode_steps": a.steps, "set_encode": fb.set_encode, "sets": {}}
    condC = (lambda sets: fb.cond_sets(sets)) if fb.set_encode else (lambda sets: fb.cond([format_claims(s_) for s_ in sets]))
    for name, (A, C, G) in sets.items():
        N = len(C); x0 = fb.norm.normalize(A.to(dev)); perm = np.random.default_rng(1).permutation(N).tolist()
        lp = {k: [] for k in ("uncond", "joint", "compose_1", "compose_m", "shuffled", "sum_single") + (("gold",) if G else ())}
        for c0 in range(0, N, a.xb):
            sl = list(range(c0, min(N, c0 + a.xb))); xx = x0[sl]; seed = 11 + c0
            def run(model, enc=None, mk=None, cv=None):
                return exact_logp(model, xx, enc, mk, n_steps=a.steps, probes=1, gen=torch.Generator(device=dev).manual_seed(seed), cvec=cv).cpu()
            u = run(fb.model); lp["uncond"].append(u)
            e, m, c = condC([C[i] for i in sl]); lp["joint"].append(run(fb.model, e, m, c))
            e, m, c = condC([C[perm[i]] for i in sl]); lp["shuffled"].append(run(fb.model, e, m, c))
            if G: e, m, c = (fb.cond_sets([[G[i]] for i in sl]) if fb.set_encode else fb.cond([G[i] for i in sl])); lp["gold"].append(run(fb.model, e, m, c))
            for mode, key in (("sum", "compose_1"), ("mean", "compose_m")):
                lp[key].append(run(Composed(fb, [C[i] for i in sl], mode)))
            if not a.no_sum_single:                                                       # sum_i PMI(c_i): one pass per claim position
                tot = torch.zeros(len(sl)); mm = max(len(C[i]) for i in sl)
                for j in range(mm):
                    rows = [r for r, i in enumerate(sl) if j < len(C[i])]
                    if not rows: continue
                    e, m, c = condC([[C[sl[r]][j]] for r in rows])
                    lj = exact_logp(fb.model, xx[rows], e, m, n_steps=a.steps, probes=1, gen=torch.Generator(device=dev).manual_seed(seed), cvec=c).cpu()
                    tot[rows] += lj - u[rows]
                lp["sum_single"].append(u + tot)
            print(f"[compose {a.tag}] {name} rows {sl[-1] + 1}/{N}", flush=True)
        lp = {k: torch.cat(v) for k, v in lp.items() if v}
        pmi = {k: ((v - lp["uncond"]) / math.log(2)) for k, v in lp.items() if k != "uncond"}
        res["sets"][name] = {"n": N, "claims_per_row": float(np.mean([len(c) for c in C])),
                             "mean_bits": {k: v.mean().item() for k, v in pmi.items()}, "sem_bits": {k: (v.std() / math.sqrt(N)).item() for k, v in pmi.items()},
                             "per_row_bits": {k: v.tolist() for k, v in pmi.items()}}
        print(f"[compose {a.tag}] {name}: " + " | ".join(f"{k} {v.mean().item():.1f}±{(v.std() / math.sqrt(N)).item():.1f}" for k, v in pmi.items()) + " bits", flush=True)
    os.makedirs(a.out, exist_ok=True); json.dump(res, open(f"{a.out}/compose_{a.tag}.json", "w"), indent=1); print("[compose] wrote", f"{a.out}/compose_{a.tag}.json", flush=True)


if __name__ == "__main__":
    main()
