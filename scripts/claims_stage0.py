"""compositionality-nla stage 0: do the EXISTING flow critics score single atomic claims sensibly? (no training)

For each held-out row (claims from scripts/claims_gen.py, uploaded to /vol_glp/cond/compnla/claims.json) and one flow critic, with the noise SHARED
across every condition of the row (common random numbers: D draws x t in {0.1,0.3,0.5,0.7,0.9}, same eps at every t within a draw):
  PMI proxy  i(h; c) = (d/2) * mean_{t, draws} [ L_uncond - L_cond ]      (nats; d = 5120; L = per-dimension MSE of the velocity)
computed for (a) every single true claim and every paired false claim, (b) the joint = all true claims of the row concatenated, (c) the gold Opus
explanation, and (d) GREEDY forward selection over the true claims (condition = the chosen claims concatenated in selection order), up to --kmax,
using the first --greedy-draws draws. Per-(row, condition, draw) values are stored so the analysis can bootstrap.

  python scripts/claims_stage0.py --critic sw_tokar          ->  /vol_glp/cond/compnla/stage0_sw_tokar.json
"""
import argparse, json, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
CRITICS = {"sw_tokar": "/vol_glp/cond/sw_tokar/adapter_latest.pt", "trunk_dn64": "/vol_glp/cond/trunk_dn64/adapter_latest.pt"}
TS = [0.1, 0.3, 0.5, 0.7, 0.9]
OUT = "/vol_glp/cond/compnla"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--critic", required=True); ap.add_argument("--D", type=int, default=8); ap.add_argument("--greedy-draws", type=int, default=4)
    ap.add_argument("--kmax", type=int, default=8); ap.add_argument("--text-chunk", type=int, default=4); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--set-test", action="store_true", help="cross-read critics only: encode every claim ALONE and concatenate the token memories (exactly permutation-invariant) -> stage0set_<critic>.json")
    ap.add_argument("--order-test", action="store_true", help="permutation/format test of the joint condition -> stage0order_<critic>.json (needs stage0_<critic>.json)")
    a = ap.parse_args(); dev = "cuda:0"
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    from nla.flow.scoring import FlowBundle
    from nla.schema import extract_explanation
    snap = snapshot_download("Qwen/Qwen3.6-27B", token=os.environ.get("HF_TOKEN"), local_dir="/root/base_snap", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.py", "*.model", "*.tiktoken"])
    C = json.load(open(f"{OUT}/claims.json"))["items"]
    if a.limit: C = C[: a.limit]
    t = pq.read_table("/vol_q36/data/sft/av_sft_val_clean1.parquet", columns=["activation_vector", "response"]); N = t.num_rows
    acts = torch.tensor(np.asarray(t.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(N, -1))
    gold = [extract_explanation(r) or r for r in t.column("response").to_pylist()]
    ap_ = CRITICS[a.critic]; aa = torch.load(ap_, map_location="cpu")["args"]; pco = os.path.join(os.path.dirname(ap_), "prior_cotrained_latest.pt")
    fb = FlowBundle(aa["prior"], ap_, aa["stats"], dev, base=snap, enc_layer=aa.get("enc_layer", 42), ar_ckpt=aa.get("ar_ckpt", "/vol/ckpts/qwen36_27b/ar_sft_merged"), prior_override=pco if os.path.exists(pco) else None)
    fb.model.eval(); d = acts.shape[1]; T = len(TS); tt = torch.tensor(TS, device=dev)

    def noisy(x0, E):                                   # E [D, d] -> x_t [D*T, d], t [D*T], target [D*T, d]  (row order: draw-major, t-minor)
        Dn = E.shape[0]
        xt = ((1 - tt)[None, :, None] * x0[None] + tt[None, :, None] * E[:, None, :]).reshape(Dn * T, d)
        tgt = (E[:, None, :] - x0[None]).expand(Dn, T, d).reshape(Dn * T, d)
        return xt, tt.repeat(Dn), tgt

    @torch.no_grad()
    def L_uncond(x0, E):
        xt, tv, tgt = noisy(x0, E)
        with torch.autocast("cuda", dtype=torch.bfloat16): v = fb.model(xt, tv).float()
        return ((v - tgt) ** 2).mean(-1).view(E.shape[0], T)          # [D, T]

    @torch.no_grad()
    def L_cond(x0, E, texts):
        xt, tv, tgt = noisy(x0, E); R = xt.shape[0]; out = []
        for i in range(0, len(texts), a.text_chunk):
            tx = texts[i:i + a.text_chunk]; G = len(tx); enc, mk, cv = fb.cond(tx)
            xB = xt.repeat(G, 1); tB = tv.repeat(G); tgB = tgt.repeat(G, 1)
            encB = enc.repeat_interleave(R, 0) if enc is not None else None; mkB = mk.repeat_interleave(R, 0) if mk is not None else None
            cvB = cv.repeat_interleave(R, 0) if cv is not None else None
            with torch.autocast("cuda", dtype=torch.bfloat16): v = fb.model(xB, tB, encB, mkB, cvB).float()
            out.append(((v - tgB) ** 2).mean(-1).view(G, E.shape[0], T))
        return torch.cat(out)                                             # [n_texts, D, T]

    def pmi(Lu, Lc):                                                      # -> per-draw PMI proxy in nats [n, D]
        return (d / 2) * (Lu[None] - Lc).mean(-1)

    res = {"critic": a.critic, "adapter": ap_, "ts": TS, "D": a.D, "greedy_draws": a.greedy_draws, "kmax": a.kmax, "rows": []}; t0 = time.time()
    if a.set_test:                                                        # set-encoded condition: CrossRead has no key positions -> order-free by construction
        assert fb.cond_mode in ("tokens_ar", "tokens_base"), "--set-test needs a cross-read (token-memory) conditioner"
        prev = {r["row"]: r for r in json.load(open(f"{OUT}/stage0_{a.critic}.json"))["rows"]}

        @torch.no_grad()
        def L_mem(x0, E, encC, mkC, sets):                               # sets: list of index lists into the row's per-claim memories -> [n_sets, D, T]
            xt, tv, tgt = noisy(x0, E); R = xt.shape[0]; out = []
            for i in range(0, len(sets), a.text_chunk):
                ss = sets[i:i + a.text_chunk]; G = len(ss); es = [encC[s][mkC[s]] for s in ss]; Tm = max(e.shape[0] for e in es)
                enc = encC.new_zeros(G, Tm, encC.shape[-1]); mk = torch.zeros(G, Tm, dtype=torch.bool, device=encC.device)
                for g, e in enumerate(es): enc[g, :e.shape[0]] = e; mk[g, :e.shape[0]] = True
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    v = fb.model(xt.repeat(G, 1), tv.repeat(G), enc.repeat_interleave(R, 0), mk.repeat_interleave(R, 0), None).float()
                out.append(((v - tgt.repeat(G, 1)) ** 2).mean(-1).view(G, E.shape[0], T))
            return torch.cat(out)

        res.update(rows=[])
        for n, it in enumerate(C):
            row = it["row"]; x0 = fb.norm.normalize(acts[row][None].to(dev)).float()
            g_ = torch.Generator(device=dev).manual_seed(a.seed * 1_000_003 + row); E = torch.randn(a.D, d, device=dev, generator=g_); Lu = L_uncond(x0, E)
            tc = [c["claim"] for c in it["true_claims"]]; fps = it["false_pairs"]; nt = len(tc)
            encC, mkC, cv = fb.cond(tc + [p["false_claim"] for p in fps]); assert cv is None
            full = list(range(nt)); swaps = [[j for j in full if j != p["true_index"]] + [nt + q] for q, p in enumerate(fps)]
            sets = [[j] for j in full] + [full] + swaps
            P = pmi(Lu, L_mem(x0, E, encC, mkC, sets)).cpu().numpy()
            rec = {"row": row, "single": P[:nt].round(3).tolist(), "set_joint": P[nt].round(3).tolist(), "set_swap": P[nt + 1:].round(3).tolist(),
                   "single_repro_maxdiff": float(np.abs(P[:nt].mean(1) - np.array(prev[row]["true"]).mean(1)).max())}
            Eg = E[: a.greedy_draws]; Lug = Lu[: a.greedy_draws]; chosen, path, remaining = [], [], list(range(nt))
            for k in range(min(a.kmax, nt)):
                Pg = pmi(Lug, L_mem(x0, Eg, encC, mkC, [chosen + [c] for c in remaining])).mean(-1).cpu().numpy()
                b = int(np.argmax(Pg)); chosen.append(remaining.pop(b)); path.append({"k": k + 1, "added": chosen[-1], "pmi": float(Pg[b]), "cands_pmi": Pg.round(3).tolist()})
            rec["greedy"] = path; res["rows"].append(rec)
            if n % 20 == 0: print(f"[set {a.critic}] row {n + 1}/{len(C)} {time.time() - t0:.0f}s | single {P[:nt].mean():.1f} (repro maxdiff {rec['single_repro_maxdiff']:.2f}) set {P[nt].mean():.1f} swaps {P[nt + 1:].mean():.1f} greedy@{len(path)} {path[-1]['pmi']:.1f}", flush=True)
        json.dump(res, open(f"{OUT}/stage0set_{a.critic}.json", "w")); print(f"[set] wrote stage0set_{a.critic}.json in {time.time() - t0:.0f}s", flush=True); return
    if a.order_test:                                                      # is the joint condition a SET? same eps as stage 0 -> "orig" reproduces stage 0's joint
        prev = {r["row"]: r for r in json.load(open(f"{OUT}/stage0_{a.critic}.json"))["rows"]}
        conds = ["orig", "prose", "reversed", "greedy", "shuffle1", "shuffle2", "shuffle3"]; res.update(conds=conds, rows=[])
        for n, it in enumerate(C):
            row = it["row"]; x0 = fb.norm.normalize(acts[row][None].to(dev)).float()
            g_ = torch.Generator(device=dev).manual_seed(a.seed * 1_000_003 + row); E = torch.randn(a.D, d, device=dev, generator=g_)
            tc = [c["claim"] for c in it["true_claims"]]; gi = [g["added"] for g in prev[row]["greedy"]]; go = gi + [j for j in range(len(tc)) if j not in gi]
            rs = np.random.default_rng(row); sh = [list(rs.permutation(len(tc))) for _ in range(3)]
            texts = ["\n".join(tc), " ".join(tc), "\n".join(tc[::-1]), "\n".join(tc[j] for j in go)] + ["\n".join(tc[j] for j in s) for s in sh]
            P = pmi(L_uncond(x0, E), L_cond(x0, E, texts)).cpu().numpy()
            res["rows"].append({"row": row, "pmi": P.round(3).tolist(), "greedy_order": go, "shuffles": [[int(j) for j in s] for s in sh]})
            if n % 20 == 0: print(f"[order {a.critic}] row {n + 1}/{len(C)} {time.time() - t0:.0f}s | " + " ".join(f"{c} {v:.1f}" for c, v in zip(conds, P.mean(1))), flush=True)
        json.dump(res, open(f"{OUT}/stage0order_{a.critic}.json", "w")); print(f"[order] wrote stage0order_{a.critic}.json in {time.time() - t0:.0f}s", flush=True); return
    for n, it in enumerate(C):
        row = it["row"]; x0 = fb.norm.normalize(acts[row][None].to(dev)).float()
        g_ = torch.Generator(device=dev).manual_seed(a.seed * 1_000_003 + row); E = torch.randn(a.D, d, device=dev, generator=g_)
        Lu = L_uncond(x0, E)
        tc = [c["claim"] for c in it["true_claims"]]; fc = [p["false_claim"] for p in it["false_pairs"]]
        base_texts = tc + fc + ["\n".join(tc), gold[row]]
        P = pmi(Lu, L_cond(x0, E, base_texts)).cpu().numpy()             # [n_texts, D]
        rec = {"row": row, "true": P[:len(tc)].round(3).tolist(), "false": P[len(tc):len(tc) + len(fc)].round(3).tolist(),
               "joint": P[len(tc) + len(fc)].round(3).tolist(), "gold": P[-1].round(3).tolist()}
        # greedy forward selection over true claims, condition = concatenation of the chosen claims (first greedy_draws draws)
        Eg = E[: a.greedy_draws]; Lug = Lu[: a.greedy_draws]; chosen, path = [], []
        remaining = list(range(len(tc)))
        for k in range(min(a.kmax, len(tc))):
            cands = ["\n".join([tc[j] for j in chosen] + [tc[c]]) for c in remaining]
            Pg = pmi(Lug, L_cond(x0, Eg, cands)).mean(-1).cpu().numpy()
            b = int(np.argmax(Pg)); chosen.append(remaining.pop(b)); path.append({"k": k + 1, "added": chosen[-1], "pmi": float(Pg[b]), "cands_pmi": Pg.round(3).tolist()})
        rec["greedy"] = path
        res["rows"].append(rec)
        if n % 10 == 0: print(f"[stage0 {a.critic}] row {n + 1}/{len(C)} {time.time() - t0:.0f}s | true mean {P[:len(tc)].mean():.1f} false mean {P[len(tc):len(tc)+len(fc)].mean():.1f} joint {P[len(tc)+len(fc)].mean():.1f} gold {P[-1].mean():.1f} greedy@{len(path)} {path[-1]['pmi']:.1f}", flush=True)
    os.makedirs(OUT, exist_ok=True); json.dump(res, open(f"{OUT}/stage0_{a.critic}.json", "w")); print(f"[stage0] wrote stage0_{a.critic}.json in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
