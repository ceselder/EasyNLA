"""Offline scoring of a CLIP-style critic (nla.contrastive) on the SAME held-out rows and items the flow / MSE critics were scored on.
Writes raw scores; the analysis (and the comparison with the flow critics' stored scores) is scripts/clip_analyze.py on the box.

  val        /vol_q36/data/sft/av_sft_val.parquet (first --val-n rows): retrieval top-1/5 at N = 1k / 10k, same-document match among 5 cuts,
             1-of-8 source match (the flow critics' eval/source_match_acc protocol: 7 random distractor explanations), wrong-detail detection by
             type (nla.flow.negatives.make_negative, random.Random(2), first 1024 rows = train_cond's eval negatives)
  detector   /vol_glp/cond/halluc_classify_numbers_sw_tokar.json: orig / near / far / hedge / removed variants of 512 grounded-number rows
  groups     /vol_glp/cond/flow_noise/gen.json: 40 clean1 activations x 8 sampled explanations (warm start + step-400 policy)
  twins      /vol_glp/cond/flow_noise/twin_acts.pt: the same explanations scored against h_stored, h_recap, detail twins and the placebo
  deletions  /vol_glp/cond/flow_noise/deletions.json: (z, z minus false claims, z minus true claims) at h_stored
  pmi        256 clean1 rows (the exact-PMI rows): discriminative PMI of the gold explanation and of a shuffled one
Every score is reported raw (scaled cosine s) and bank-normalised: s(h, z) - log mean_j exp s(h_j, z) over a fixed bank of --bank-n val
activations from documents not in clean1 (the discriminative PMI estimate; the z-dependent normaliser charges generic explanations).
"""
import argparse, json, math, os, random, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--ckpt", required=True); p.add_argument("--base", required=True); p.add_argument("--out", default=None)
    p.add_argument("--val-n", type=int, default=14711); p.add_argument("--bank-n", type=int, default=4096); p.add_argument("--tests", default="val,detector,groups,twins,deletions,pmi,ladder")
    a = p.parse_args(); tests = set(a.tests.split(",")); t0 = time.time()
    import pyarrow.parquet as pq
    from nla.contrastive.model import ClipCritic
    from nla.flow.negatives import make_negative
    from nla.schema import extract_explanation
    C = ClipCritic(a.ckpt, a.base, "cuda"); dev = "cuda"; res = {"ckpt": a.ckpt, "args": C.args}
    vt = pq.read_table("/vol_q36/data/sft/av_sft_val.parquet", columns=["activation_vector", "response", "doc_id"]).slice(0, a.val_n)
    VA = torch.tensor(np.asarray(vt.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(vt.num_rows, -1))
    VZ = [(extract_explanation(r) or r or "").strip() for r in vt.column("response").to_pylist()]; VD = vt.column("doc_id").to_pylist()
    ct = pq.read_table("/vol_q36/data/sft/av_sft_val_clean1.parquet", columns=["activation_vector", "response", "doc_id"])
    CA = torch.tensor(np.asarray(ct.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(ct.num_rows, -1))
    CZ = [(extract_explanation(r) or r or "").strip() for r in ct.column("response").to_pylist()]; cdocs = set(ct.column("doc_id").to_pylist())
    with torch.no_grad():
        VAe = torch.cat([C.act_emb(VA[i:i + 4096]) for i in range(0, len(VA), 4096)])
        bank_idx = [i for i in range(len(VD)) if VD[i] not in cdocs][: a.bank_n]; Bk = VAe[bank_idx]; s_ = C.heads.scale().item(); res["scale"] = s_
    print(f"[clip-eval] {a.ckpt}: scale {s_:.1f}, val {len(VZ)} rows, bank {len(bank_idx)} activations ({time.time() - t0:.0f}s)", flush=True)
    def lse_bank(T):   # [nT, D] -> log mean_j exp s(h_j, z) over the bank, per text
        return (torch.logsumexp(s_ * C.sim(Bk, T), 0) - math.log(Bk.shape[0]))
    @torch.no_grad()
    def score(Ae, texts):   # Ae [nA, D] embeddings; -> raw [nA, nT], normaliser [nT]
        T = C.text_emb(texts); return (s_ * C.sim(Ae, T)), lse_bank(T), T

    if "val" in tests:
        with torch.no_grad(): VT = C.text_emb(VZ)
        out = {}
        for n_ in (1000, 10000):
            n_ = min(n_, len(VZ)); L = C.sim(VAe[:n_], VT[:n_]); ar = torch.arange(n_, device=dev)
            for nm, M in (("a2t", L), ("t2a", L.T)):
                top = M.topk(5, dim=1).indices; out[f"ret_{nm}_top1_n{n_}"] = (top[:, 0] == ar).float().mean().item(); out[f"ret_{nm}_top5_n{n_}"] = (top == ar[:, None]).any(1).float().mean().item()
        by = {}
        for i, d_ in enumerate(VD): by.setdefault(d_, []).append(i)
        ok_r = ok_c = tot = 0
        for g in by.values():
            if len(g) < 5: continue
            g = sorted(g)[:5]; M = C.sim(VAe[g], VT[g]); ar = torch.arange(5, device=dev); ok_r += (M.argmax(1) == ar).sum().item(); ok_c += (M.argmax(0) == ar).sum().item(); tot += 5
        out["samedoc5_a2t"] = ok_r / tot; out["samedoc5_t2a"] = ok_c / tot; out["samedoc5_n"] = tot
        rng = np.random.default_rng(3); ok = 0; M_ = 1024
        for i in range(M_):
            cand = [i] + list(rng.choice([j for j in range(M_) if j != i], 7, replace=False)); ok += int(C.sim(VAe[i:i + 1], VT[[int(c) for c in cand]])[0].argmax().item() == 0)
        out["source_match_1of8"] = ok / M_
        nrng = random.Random(2); negs = [make_negative(z, nrng, VZ[:1024]) for z in VZ[:1024]]; rows_ = [i for i, (zn, _) in enumerate(negs) if zn]
        with torch.no_grad(): TN = C.text_emb([negs[i][0] for i in rows_])
        kinds = {}
        dT = C.diag(VAe[rows_], VT[rows_]); dN = C.diag(VAe[rows_], TN)
        for j, i in enumerate(rows_): kinds.setdefault(negs[i][1], []).append(bool(dT[j] > dN[j]))
        out["neg_detect_acc"] = float(np.mean([w for v in kinds.values() for w in v])); out.update({f"neg_detect_acc_{k}": float(np.mean(v)) for k, v in kinds.items()}); out.update({f"neg_n_{k}": len(v) for k, v in kinds.items()})
        res["val"] = out; print("[clip-eval] val", json.dumps({k: round(v, 4) for k, v in out.items()}), flush=True)

    if "detector" in tests:
        cj = json.load(open("/vol_glp/cond/halluc_classify_numbers_sw_tokar.json")); items = cj["items"][:512]; modes = ["orig"] + [m for m in cj["modes"] if m != "orig"]
        va_full = pq.read_table("/vol_q36/data/sft/av_sft_val.parquet", columns=["activation_vector"]); NV = va_full.num_rows
        ACTF = torch.tensor(np.asarray(va_full.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(NV, -1))
        det = []
        for it in items:
            Ae = C.act_emb(ACTF[it["row"]][None]); raw, lz, _ = score(Ae, [it["variants"][m]["text"] for m in modes])
            det.append({"row": it["row"], "raw": dict(zip(modes, raw[0].tolist())), "pmi": dict(zip(modes, (raw[0] - lz).tolist()))})
        res["detector"] = {"modes": modes, "items": det}; print(f"[clip-eval] detector: {len(det)} items", flush=True)

    if tests & {"groups", "twins", "deletions"}:
        gen = json.load(open("/vol_glp/cond/flow_noise/gen.json")); TA = torch.load("/vol_glp/cond/flow_noise/twin_acts.pt") if tests & {"twins", "deletions"} else {}
        dele = json.load(open("/vol_glp/cond/flow_noise/deletions.json"))["items"] if "deletions" in tests else []
        gr = {av: {"raw": [], "pmi": []} for av in gen["avs"]}; tw = {}; dl = []
        for g, row in enumerate(gen["rows"]):
            Ah = C.act_emb(CA[row][None])
            for av, v in gen["avs"].items():
                texts = [z if z else "(empty)" for z in v["explanations"][g]]; raw, lz, T = score(Ah, texts)
                gr[av]["raw"].append(raw[0].tolist()); gr[av]["pmi"].append((raw[0] - lz).tolist())
                if row in TA:
                    e = TA[row]; acts = [e["h_stored"], e["h_recap"]] + list(e["twins"]) + ([e["placebo"]] if e["placebo"] is not None else [])
                    Aa = C.act_emb(torch.stack(acts).float()); tw.setdefault(str(row), {"n_twins": len(e["twins"]), "has_placebo": e["placebo"] is not None, "S": {}})
                    tw[str(row)]["S"][av] = (s_ * C.sim(Aa, T)).T.tolist()           # [nZ, nA]
            for it in [d for d in dele if d["row"] == row]:
                raw, lz, _ = score(Ah, [it["z"], it["remove_false"] or "(empty)", it["remove_true"] or "(empty)"])
                dl.append({"av": it["av"], "g": it["g"], "i": it["i"], "row": row, "n_removed": it["n_removed"], "n_false": it["n_false"], "raw": raw[0].tolist(), "pmi": (raw[0] - lz).tolist()})
        res["groups"] = gr; res["twins"] = tw; res["deletions"] = dl; print(f"[clip-eval] groups/twins/deletions done ({time.time() - t0:.0f}s)", flush=True)

    if "ladder" in tests and os.path.exists("/vol_glp/scale/g2pilot/g2_pilot_v3.parquet"):
        # held-out hedge ladders (g2 pilot Opus-overlap positions = av_sft_val rows; same items + template as scripts/hedge_ladder_eval.py)
        from nla.contrastive.ladders import sentence, z0_of, ORDER
        P_ = [r for r in pq.read_table("/vol_glp/scale/g2pilot/g2_pilot_v3.parquet").to_pylist() if r["src"] == "opus_overlap" and r["facts"] and r["fact_ladders"]]
        vfull = pq.read_table("/vol_q36/data/sft/av_sft_val.parquet", columns=["activation_vector"]); lad = []
        for r in P_:
            z0 = z0_of(r["facts"]); facts = [x for x in json.loads(r["fact_ladders"]) if x.get("twin")][:3]
            if not z0 or not facts: continue
            Ae = C.act_emb(torch.tensor(np.asarray(vfull.column(0)[r["row"]].as_py(), dtype=np.float32))[None])
            for x in facts:
                rungs = {k: (z0 + " " + sentence(x, k)) if sentence(x, k) else (z0 if k == "omit" else None) for k in ORDER}; names = [k for k, v in rungs.items() if v]
                raw, lz, _ = score(Ae, [rungs[k] for k in names])
                lad.append({"row": r["row"], "type": x["type"], "raw": dict(zip(names, raw[0].tolist())), "pmi": dict(zip(names, (raw[0] - lz).tolist()))})
        res["ladder"] = lad; print(f"[clip-eval] ladder: {len(lad)} items", flush=True)

    if "pmi" in tests:
        n = 256; perm = torch.randperm(n, generator=torch.Generator().manual_seed(1)).tolist(); zs = CZ[:n]; zsh = [zs[i] for i in perm]
        with torch.no_grad():
            Ae = C.act_emb(CA[:n]); T = C.text_emb(zs); Ts = C.text_emb(zsh)
            pmi = (s_ * C.diag(Ae, T) - lse_bank(T)); pms = (s_ * C.diag(Ae, Ts) - lse_bank(Ts))
        res["pmi"] = {"n": n, "pmi_nats_mean": pmi.mean().item(), "pmi_bits_mean": pmi.mean().item() / math.log(2), "shuf_bits_mean": pms.mean().item() / math.log(2),
                      "cap_bits": math.log(len(bank_idx)) / math.log(2), "frac_positive": (pmi > 0).float().mean().item()}
        print("[clip-eval] pmi", json.dumps(res["pmi"]), flush=True)
    out = a.out or os.path.join(a.ckpt, "offline_eval.json"); json.dump(res, open(out, "w")); print(f"[clip-eval] wrote {out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
