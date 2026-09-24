"""Offline scoring for the unCLIP critic comparison, on the SAME held-out rows and items every other critic was scored on
(the item definitions are copied from scripts/clip_eval.py so the analysis in scripts/plot_unclip.py can put all critics side by side).

--critic unclip:<encoder.json>:<prior_dir>   UnclipCritic (nla/unclip/critic.py): logp_cond = log p(e|z), pmi = log p(e|z) - log p(e) (nats)
                                              tests: wd, detector, groups, twins, deletions, ladder, pmi
--critic flow:<adapter_path>                  flow critic, FM proxy of its RL reward (t in {0.3, 0.5, 0.7}, eps shared per item and t, the train_cond eval
                                              protocol), per-item scores for the wrong-detail items only (test wd; every other test already has stored scores)
--critic clip:<ckpt_dir>                      contrastive critic (nla.contrastive.model.ClipCritic), raw scaled cosine, per-item wd scores
--critic mse                                  the SFT MSE reconstructor (-MSE of its prediction, NLA units), per-item wd scores

wd        av_sft_val rows 0-1023, nla.flow.negatives.make_negative with random.Random(2) and pool = the same 1,024 explanations (1,023 items)
detector  /vol_glp/cond/halluc_classify_numbers_sw_tokar.json: orig / near / far / hedge / removed variants of 512 grounded-number rows
groups    /vol_glp/cond/flow_noise/gen.json (40 clean1 activations x 8 sampled explanations; warm start + step-400 policy)
twins     /vol_glp/cond/flow_noise/twin_acts.pt, deletions /vol_glp/cond/flow_noise/deletions.json
ladder    /vol_glp/scale/g2pilot/g2_pilot_v3.parquet Opus-overlap positions (nla.contrastive.ladders template)
pmi       256 clean1 rows: exact PMI of the gold and of a shuffled explanation; e-space code length -log p(e|z)
Output: JSON (default /vol_glp/unclip/evals/<tag>.json)."""
import argparse, json, math, os, random, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))


def load_val(n=1024):
    import pyarrow.parquet as pq
    from nla.schema import extract_explanation
    vt = pq.read_table("/vol_q36/data/sft/av_sft_val.parquet", columns=["activation_vector", "response"]).slice(0, n)
    A = torch.tensor(np.asarray(vt.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(vt.num_rows, -1))
    Z = [(extract_explanation(r) or r or "").strip() for r in vt.column("response").to_pylist()]
    return A, Z


def wd_items(Z):
    from nla.flow.negatives import make_negative
    rng = random.Random(2); negs = [make_negative(z, rng, Z) for z in Z]
    return [(i, negs[i][0], negs[i][1]) for i in range(len(Z)) if negs[i][0]]


class Scorer:
    """uniform interface: score(H [N, 5120] raw float, texts [N]) -> dict of np arrays ('lp' = higher is better; 'pmi' where defined)"""
    def __init__(self, spec, dev="cuda"):
        self.kind, _, rest = spec.partition(":"); self.dev = dev; self.spec = spec
        if self.kind == "unclip":
            enc_json, _, prior_dir = rest.partition(":")
            from nla.unclip.critic import UnclipCritic
            self.C = UnclipCritic(enc_json, prior_dir, dev)
        elif self.kind == "flow":
            from nla.flow.scoring import FlowBundle
            aa = torch.load(rest, map_location="cpu")["args"]; pco = os.path.join(os.path.dirname(rest), "prior_cotrained_latest.pt")
            self.fb = FlowBundle(aa["prior"], rest, aa["stats"], dev, base="Qwen/Qwen3.6-27B", enc_layer=aa.get("enc_layer", 42),
                                 ar_ckpt=aa.get("ar_ckpt", "/vol/ckpts/qwen36_27b/ar_sft_merged"), prior_override=pco if os.path.exists(pco) else None)
            self.fb.model.eval()
        elif self.kind == "clip":
            from nla.contrastive.model import ClipCritic
            self.C = ClipCritic(rest, "Qwen/Qwen3.6-27B", dev)
        elif self.kind == "mse":
            import intervene_playground_app as pg
            from nla.schema import normalize_activation
            pg._load(); self.pg = pg; self.norm = normalize_activation; self.critic = pg._critic("SFT reconstructor (ar_sft_merged, pre-RL)")
        else:
            raise ValueError(spec)

    @torch.no_grad()
    def score(self, H, texts, bs=32, **kw):
        out = {}
        for i in range(0, len(texts), bs):
            h = H[i:i + bs].float(); tx = texts[i:i + bs]
            if self.kind == "unclip":
                r = self.C.score(h, tx, **kw); d = {"lp": r["logp_cond"], "pmi": r["pmi"], "lp_uncond": r["logp_uncond"]}
            elif self.kind == "clip":
                Ae = self.C.act_emb(h); T = self.C.text_emb(tx); d = {"lp": self.C.heads.scale().item() * (Ae * T).sum(-1)}
            elif self.kind == "mse":
                msf = self.pg.S["msf"]; lp = []
                for hh, z in zip(h, tx):
                    p = self.pg.ar_pred(self.critic, z or "(empty)")
                    lp.append(-float(((self.norm(p[None], msf) - self.norm(hh[None].to(p.device), msf)) ** 2).mean()))
                d = {"lp": torch.tensor(lp)}
            for k, v in d.items(): out.setdefault(k, []).append(torch.as_tensor(v).float().cpu())
        return {k: torch.cat(v).numpy() for k, v in out.items()}

    @torch.no_grad()
    def flow_gap(self, h, z_true, z_neg, seed):
        """flow FM-proxy per item: loss(neg) - loss(true) at t in {0.3, 0.5, 0.7} with eps shared by the pair (train_cond eval protocol)"""
        fb = self.fb; x0 = fb.norm.normalize(h[None].to(self.dev)).float(); g_ = torch.Generator(device=self.dev).manual_seed(seed); gaps = []
        enc_t, mk_t, cv_t = fb.cond([z_true]); enc_n, mk_n, cv_n = fb.cond([z_neg])
        for t in (0.3, 0.5, 0.7):
            eps = torch.randn(x0.shape, device=self.dev, generator=g_); xt = (1 - t) * x0 + t * eps; tv = torch.full((1,), t, device=self.dev); L = []
            for enc, mk, cv in ((enc_t, mk_t, cv_t), (enc_n, mk_n, cv_n)):
                with torch.autocast("cuda", dtype=torch.bfloat16): v = fb.model(xt, tv, enc, mk, cv)
                L.append(float(fb.fm_err(v, eps - x0)[0]))
            gaps.append(L[1] - L[0])
        return gaps


def main():
    p = argparse.ArgumentParser(); p.add_argument("--critic", required=True); p.add_argument("--tag", required=True); p.add_argument("--out", default=None)
    p.add_argument("--tests", default="wd,detector,groups,twins,deletions,ladder,pmi"); p.add_argument("--score-kw", default="{}", help="JSON kwargs for UnclipCritic.score (e.g. exact/proxy mode, n_steps)")
    a = p.parse_args(); tests = set(a.tests.split(",")); kw = json.loads(a.score_kw); t0 = time.time()
    S = Scorer(a.critic); res = {"critic": a.critic, "tag": a.tag, "score_kw": kw, "tests": sorted(tests)}
    A, Z = load_val(1024); items = wd_items(Z)
    print(f"[unclip-eval] {a.critic}: {len(items)} wrong-detail items ({time.time() - t0:.0f}s)", flush=True)

    if "wd" in tests:
        if S.kind == "flow":
            wd = [{"row": i, "kind": k, "gaps": S.flow_gap(A[i], Z[i], zn, 1000 + i)} for i, zn, k in items]
        else:
            H = torch.stack([A[i] for i, _, _ in items]); st = S.score(H, [Z[i] for i, _, _ in items], **kw); sn = S.score(H, [zn for _, zn, _ in items], **kw)
            wd = [{"row": i, "kind": k, **{f"true_{m}": float(st[m][j]) for m in st}, **{f"neg_{m}": float(sn[m][j]) for m in sn}} for j, (i, _, k) in enumerate(items)]
        res["wd"] = wd; print(f"[unclip-eval] wd done ({time.time() - t0:.0f}s)", flush=True)
    if S.kind != "unclip":
        tests &= {"wd"}

    if "detector" in tests:
        import pyarrow.parquet as pq
        cj = json.load(open("/vol_glp/cond/halluc_classify_numbers_sw_tokar.json")); its = cj["items"][:512]; modes = ["orig"] + [m for m in cj["modes"] if m != "orig"]
        vf = pq.read_table("/vol_q36/data/sft/av_sft_val.parquet", columns=["activation_vector"]); col = vf.column(0)
        det = []
        for it in its:
            h = torch.tensor(np.asarray(col[it["row"]].as_py(), dtype=np.float32))[None].repeat(len(modes), 1)
            sc = S.score(h, [it["variants"][m]["text"] for m in modes], **kw)
            det.append({"row": it["row"], "lp": dict(zip(modes, sc["lp"].tolist())), "pmi": dict(zip(modes, sc["pmi"].tolist()))})
        res["detector"] = {"modes": modes, "items": det}; print(f"[unclip-eval] detector done ({time.time() - t0:.0f}s)", flush=True)

    if tests & {"groups", "twins", "deletions", "pmi"}:
        import pyarrow.parquet as pq
        from nla.schema import extract_explanation
        ct = pq.read_table("/vol_q36/data/sft/av_sft_val_clean1.parquet", columns=["activation_vector", "response"])
        CA = torch.tensor(np.asarray(ct.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(ct.num_rows, -1))
        CZ = [(extract_explanation(r) or r or "").strip() for r in ct.column("response").to_pylist()]

    if tests & {"groups", "twins", "deletions"}:
        gen = json.load(open("/vol_glp/cond/flow_noise/gen.json")); TA = torch.load("/vol_glp/cond/flow_noise/twin_acts.pt") if tests & {"twins", "deletions"} else {}
        dele = json.load(open("/vol_glp/cond/flow_noise/deletions.json"))["items"] if "deletions" in tests else []
        gr = {av: {"lp": [], "pmi": []} for av in gen["avs"]}; tw = {}; dl = []
        for g, row in enumerate(gen["rows"]):
            for av, v in gen["avs"].items():
                texts = [z if z else "(empty)" for z in v["explanations"][g]]
                if "groups" in tests:
                    sc = S.score(CA[row][None].repeat(len(texts), 1), texts, **kw); gr[av]["lp"].append(sc["lp"].tolist()); gr[av]["pmi"].append(sc["pmi"].tolist())
                if "twins" in tests and row in TA:
                    e = TA[row]; acts = [e["h_stored"], e["h_recap"]] + list(e["twins"]) + ([e["placebo"]] if e["placebo"] is not None else [])
                    M_lp, M_pmi = [], []
                    for z in texts:
                        sc = S.score(torch.stack(acts).float(), [z] * len(acts), **kw); M_lp.append(sc["lp"].tolist()); M_pmi.append(sc["pmi"].tolist())
                    tw.setdefault(str(row), {"n_twins": len(e["twins"]), "has_placebo": e["placebo"] is not None, "lp": {}, "pmi": {}})
                    tw[str(row)]["lp"][av] = M_lp; tw[str(row)]["pmi"][av] = M_pmi        # [nZ, nA]
            for it in [d for d in dele if d["row"] == row]:
                sc = S.score(CA[row][None].repeat(3, 1), [it["z"], it["remove_false"] or "(empty)", it["remove_true"] or "(empty)"], **kw)
                dl.append({"av": it["av"], "g": it["g"], "i": it["i"], "row": row, "n_removed": it["n_removed"], "n_false": it["n_false"], "lp": sc["lp"].tolist(), "pmi": sc["pmi"].tolist()})
        res["groups"] = gr; res["twins"] = tw; res["deletions"] = dl; print(f"[unclip-eval] groups/twins/deletions done ({time.time() - t0:.0f}s)", flush=True)

    if "ladder" in tests and os.path.exists("/vol_glp/scale/g2pilot/g2_pilot_v3.parquet"):
        import pyarrow.parquet as pq
        from nla.contrastive.ladders import sentence, z0_of, ORDER
        P_ = [r for r in pq.read_table("/vol_glp/scale/g2pilot/g2_pilot_v3.parquet").to_pylist() if r["src"] == "opus_overlap" and r["facts"] and r["fact_ladders"]]
        vf = pq.read_table("/vol_q36/data/sft/av_sft_val.parquet", columns=["activation_vector"]); col = vf.column(0); lad = []
        for r in P_:
            z0 = z0_of(r["facts"]); facts = [x for x in json.loads(r["fact_ladders"]) if x.get("twin")][:3]
            if not z0 or not facts: continue
            h = torch.tensor(np.asarray(col[r["row"]].as_py(), dtype=np.float32))[None]
            for x in facts:
                rungs = {k: (z0 + " " + sentence(x, k)) if sentence(x, k) else (z0 if k == "omit" else None) for k in ORDER}; names = [k for k, v in rungs.items() if v]
                sc = S.score(h.repeat(len(names), 1), [rungs[k] for k in names], **kw)
                lad.append({"row": r["row"], "type": x["type"], "lp": dict(zip(names, sc["lp"].tolist())), "pmi": dict(zip(names, sc["pmi"].tolist()))})
        res["ladder"] = lad; print(f"[unclip-eval] ladder: {len(lad)} items ({time.time() - t0:.0f}s)", flush=True)

    if "pmi" in tests:
        n = 256; perm = torch.randperm(n, generator=torch.Generator().manual_seed(1)).tolist(); zs = CZ[:n]; zsh = [zs[i] for i in perm]
        sc = S.score(CA[:n], zs, **kw); ss = S.score(CA[:n], zsh, **kw); L2 = math.log(2)
        res["pmi"] = {"n": n, "pmi_bits_mean": float(sc["pmi"].mean() / L2), "pmi_bits_sem": float(sc["pmi"].std() / math.sqrt(n) / L2),
                      "shuf_bits_mean": float(ss["pmi"].mean() / L2), "frac_positive": float((sc["pmi"] > 0).mean()),
                      "e_code_bits_cond": float(-sc["lp"].mean() / L2), "e_code_bits_uncond": float(-sc["lp_uncond"].mean() / L2)}
        print("[unclip-eval] pmi", json.dumps(res["pmi"]), flush=True)

    out = a.out or f"/vol_glp/unclip/evals/{a.tag}.json"; os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w")); print(f"[unclip-eval] wrote {out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
