"""Pooling diagnostic: does a critic's CONDITIONING representation even carry a swapped detail?

Pairs (true text, copy with ONE detail swapped):
  make_negative  the 1,023 held-out Opus-explanation negatives every critic's wrong-detail eval uses (av_sft_val, random.Random(2); number / quote / name)
  g2_twin        g2 renderings with one exact fact value replaced by its wrong-exact twin (a value of the same subtype from another document;
                 nla.contrastive.ladders.twin_negative) on a g2 shard NO critic trained on
  ladder         held-out g2-pilot Opus-overlap ladders (hedge_ladder_eval template): exact vs twin, exact vs category (hedge), exact vs omitted
Representations of a text z:
  AR(z)                    the MSE reconstructor's prediction of h (ar_sft_merged value head)                              [5120]
  trunk tokens, frozen     AR-SFT trunk layer-42 token states (the unCLIP prior's per-token input; the frozen tokens_ar memory)
  trunk tokens, 644-bit    the same trunk with the 644-bit flow conditioner's LoRA (sw_tokar ar_encoder_latest.pt) = that critic's memory
      each token representation summarised three ways: mean-pooled | the swapped tokens only (the span between the common prefix and
      suffix, mean) | per-token (min cosine over the tokens after the common prefix, suffix-aligned)
  g(z) <run>               pooled contrastive text embeddings (attention pooling heads over the frozen trunk tokens) [1024]
Measures per pair set x representation (and by detail type and by distance of the true value from the activation's position):
  cos           cosine between the pair's representations
  rel_dist      (1 - cos(pair)) / median(1 - cos) over random pairs of different explanations: the swap as a fraction of a typical
                between-explanation distance (0 = the detail is invisible, 1 = as different as two unrelated explanations)
  probe_acc     held-out-document linear probe (logistic regression, no bias) on r(a) - r(b) for a random order of (true, swapped):
                can a fixed linear readout tell which text is the true one (text-only, so it reads plausibility / consistency cues)
-> /vol_glp/clip/pooling_diagnostic.json"""
import argparse, json, math, os, random, sys, time, zlib
import numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
BINS = [(12, "<=12 chars back"), (60, "12-60"), (250, "60-250"), (10 ** 9, ">250")]


def dist_bucket(src, val):
    if not src or not val: return "absent"
    v = val.strip().strip('"“”').strip()
    if len(v) < 2: return "absent"
    i = src.rfind(v)
    if i < 0: i = src.lower().rfind(v.lower())
    if i < 0: return "absent"
    back = len(src) - (i + len(v))
    for b, name in BINS:
        if back <= b: return name


def diff_span(a, b):
    p = 0
    while p < min(len(a), len(b)) and a[p] == b[p]: p += 1
    s = 0
    while s < min(len(a), len(b)) - p and a[len(a) - 1 - s] == b[len(b) - 1 - s]: s += 1
    return a[p:len(a) - s], b[p:len(b) - s]


def main():
    p = argparse.ArgumentParser(); p.add_argument("--base", required=True); p.add_argument("--out", default="/vol_glp/clip/pooling_diagnostic.json")
    p.add_argument("--clip-runs", default="plain=/vol_glp/clip/clipQ_opus_frozen_plain/latest,edit_negatives=/vol_glp/clip/clipQ_opus_frozen/latest")
    p.add_argument("--g2-shard", default=""); p.add_argument("--g2-n", type=int, default=2500); p.add_argument("--bs", type=int, default=64)
    a = p.parse_args(); t0 = time.time(); dev = "cuda"; rng = random.Random(0)
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from nla.schema import extract_explanation
    from nla.flow.negatives import make_negative
    from nla.flow.train_cond import ARVecEncoder, _qc_ok
    from nla.contrastive.ladders import twin_negative, sentence, z0_of
    from nla.contrastive.model import ClipHeads
    tok = AutoTokenizer.from_pretrained(a.base); tok.padding_side = "right"
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    # ---------------- pairs
    pairs = []   # dict(set, type, bucket, doc, a=true text, b=swapped text)
    vt = pq.read_table("/vol_q36/data/sft/av_sft_val.parquet", columns=["response", "doc_id", "detokenized_text_truncated"])
    VZ = [(extract_explanation(r) or r or "").strip() for r in vt.column("response").to_pylist()]; VD = vt.column("doc_id").to_pylist(); VS = vt.column("detokenized_text_truncated").to_pylist()
    nrng = random.Random(2)
    for i, z in enumerate(VZ[:1024]):
        zn, kind = make_negative(z, nrng, VZ[:1024])
        if zn: tv, _ = diff_span(z, zn); pairs.append(dict(set="make_negative", type=kind, bucket=dist_bucket(VS[i], tv), doc=str(VD[i]), a=z, b=zn))
    if a.g2_shard:
        cols = ["explanation", "explanations", "qc", "fact_ladders", "doc_id", "text"]
        t = pq.read_table(a.g2_shard, columns=cols).slice(0, a.g2_n).to_pylist(); trng = random.Random(1)
        for r in t:
            ok = [e for e, q in zip(r["explanations"] or [], r["qc"] or []) if e and _qc_ok(q)]; z = ok[0] if ok else (r["explanation"] or "")
            zn, ty = twin_negative(z, r["fact_ladders"], trng)
            if zn: tv, _ = diff_span(z, zn); pairs.append(dict(set="g2_twin", type=ty, bucket=dist_bucket(r["text"], tv), doc=str(r["doc_id"]), a=z, b=zn))
    lp = "/vol_glp/scale/g2pilot/g2_pilot_v3.parquet"
    if os.path.exists(lp):
        for r in pq.read_table(lp).to_pylist():
            if r["src"] != "opus_overlap" or not r["facts"] or not r["fact_ladders"]: continue
            z0 = z0_of(r["facts"])
            if not z0: continue
            for x in [x for x in json.loads(r["fact_ladders"]) if x.get("twin")][:3]:
                ex = sentence(x, "exact"); bk = dist_bucket(VS[r["row"]], str(x["value"]))
                if not ex: continue
                for other in ("twin", "category", "omit"):
                    so = sentence(x, other)
                    if other == "omit": pairs.append(dict(set=f"ladder_exact_vs_omit", type=x["type"], bucket=bk, doc=str(VD[r["row"]]), a=z0 + " " + ex, b=z0))
                    elif so: pairs.append(dict(set=f"ladder_exact_vs_{other}", type=x["type"], bucket=bk, doc=str(VD[r["row"]]), a=z0 + " " + ex, b=z0 + " " + so))
    rand_txt = VZ[1024:3024]
    print(f"[diag] {len(pairs)} pairs: " + json.dumps({s: sum(1 for q in pairs if q["set"] == s) for s in sorted(set(q["set"] for q in pairs))}) + f" ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- representations
    def run_encoder(enc, texts, want_ar=False):
        """-> list of token-state tensors [T_i, d] (masked tokens only, fp16, cpu), list of token id lists, AR predictions [n, 5120] or None"""
        toks, ids_all, ars = [], [], []
        for i in range(0, len(texts), a.bs):
            ch = [z if z else "(empty)" for z in texts[i:i + a.bs]]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                e, m = enc.tokens(ch)
                if want_ar: enc.forward(ch); ars.append(enc.last_pred_raw.float().cpu())
            ids = enc.tok([enc.tmpl.format(explanation=z) for z in ch], padding=True, truncation=True, max_length=256, add_special_tokens=False)["input_ids"]
            for j in range(len(ch)):
                mk = m[j]; toks.append(e[j][mk].to(torch.float16).cpu()); ids_all.append([t for t, keep in zip(ids[j], mk.tolist()) if keep])
        return toks, ids_all, (torch.cat(ars) if want_ar else None)

    all_texts = [q["a"] for q in pairs] + [q["b"] for q in pairs] + rand_txt; NP = len(pairs)
    frozen = ARVecEncoder("/vol/ckpts/qwen36_27b/ar_sft_merged", tok, dev, trainable=False, enc_layer=42)
    TF, IDS, AR = run_encoder(frozen, all_texts, want_ar=True); print(f"[diag] frozen trunk + AR done ({time.time() - t0:.0f}s)", flush=True)
    del frozen; torch.cuda.empty_cache()
    lora = ARVecEncoder("/vol/ckpts/qwen36_27b/ar_sft_merged", tok, dev, trainable=True, grad_ckpt=False, enc_layer=42)
    lora.load_saved(torch.load("/vol_glp/cond/sw_tokar/ar_encoder_latest.pt", map_location="cpu")); (lora.crit if lora.crit is not None else lora.lm).eval()
    TL, _, _ = run_encoder(lora, all_texts); print(f"[diag] 644-bit conditioner memory done ({time.time() - t0:.0f}s)", flush=True)
    del lora; torch.cuda.empty_cache()
    G = {}
    for spec in a.clip_runs.split(","):
        name, _, d = spec.partition("=")
        st = torch.load(os.path.join(d, "heads.pt"), map_location="cpu"); h = ClipHeads(st["args"]["act_arch"], 5120, st["args"]["d_out"]).to(dev); h.load_state_dict(st["heads"]); h.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(TF), 256):
                ch = TF[i:i + 256]; L = max(x.shape[0] for x in ch); E = torch.zeros(len(ch), L, 5120, device=dev); M = torch.zeros(len(ch), L, dtype=torch.bool, device=dev)
                for j, x in enumerate(ch): E[j, : x.shape[0]] = x.to(dev).float(); M[j, : x.shape[0]] = True
                out.append(F.normalize(h.pool(E, M).float(), dim=-1).cpu())
        G[f"g(z) contrastive: {name}"] = torch.cat(out)

    # ---------------- per-pair features and similarities
    def span(ids_a, ids_b):
        pp = 0
        while pp < min(len(ids_a), len(ids_b)) and ids_a[pp] == ids_b[pp]: pp += 1
        ss = 0
        while ss < min(len(ids_a), len(ids_b)) - pp and ids_a[-1 - ss] == ids_b[-1 - ss]: ss += 1
        return pp, ss
    def tok_reprs(T):
        """-> dict summary -> (vectors for all texts [N, d] or None, per-pair cosine list)"""
        mean = torch.stack([x.float().mean(0) for x in T])
        sw_a, sw_b, pt_cos, sw_cos = [], [], [], []
        for k in range(NP):
            ia, ib = IDS[k], IDS[NP + k]; A_, B_ = T[k].float(), T[NP + k].float(); pp, ss = span(ia, ib)
            ea = A_[pp:len(ia) - ss] if len(ia) - ss > pp else A_[-1:]; eb = B_[pp:len(ib) - ss] if len(ib) - ss > pp else B_[-1:]
            sw_a.append(ea.mean(0)); sw_b.append(eb.mean(0)); sw_cos.append(F.cosine_similarity(ea.mean(0), eb.mean(0), dim=0).item())
            cs = [F.cosine_similarity(ea.mean(0), eb.mean(0), dim=0).item()]
            if ss > 0: cs += F.cosine_similarity(A_[len(ia) - ss:], B_[len(ib) - ss:], dim=-1).tolist()   # suffix-aligned tokens after the swap
            pt_cos.append(min(cs))
        return {"mean-pooled": (mean, None), "swapped tokens only": (torch.stack(sw_a), torch.stack(sw_b), sw_cos), "per-token (most-changed token)": (None, None, pt_cos)}
    R = {}   # name -> (vec_a [NP, d] | None, vec_b | None, cos list, random-pair baseline 1-cos median)
    def rand_base(V):
        V = F.normalize(V[2 * NP:].float(), dim=-1); ii = torch.randperm(V.shape[0], generator=torch.Generator().manual_seed(0))
        return float(torch.median(1 - (V * V[ii]).sum(-1)))
    def add_vec(name, V):
        Vn = F.normalize(V.float(), dim=-1); cos = (Vn[:NP] * Vn[NP:2 * NP]).sum(-1).tolist(); R[name] = (V[:NP], V[NP:2 * NP], cos, rand_base(V))
    add_vec("AR(z) (MSE reconstructor prediction)", AR)
    for nm, T in (("trunk tokens, frozen AR-SFT (unCLIP prior input)", TF), ("trunk tokens, 644-bit flow conditioner LoRA", TL)):
        tr = tok_reprs(T); add_vec(f"{nm}: mean-pooled", tr["mean-pooled"][0])
        base_sw = R[f"{nm}: mean-pooled"][3]   # per-token summaries have no single-text vector: normalise by the pooled random-pair distance
        R[f"{nm}: swapped tokens only"] = (tr["swapped tokens only"][0], tr["swapped tokens only"][1], tr["swapped tokens only"][2], None)
        R[f"{nm}: per-token (most-changed token)"] = (None, None, tr["per-token (most-changed token)"][2], None)
    for nm, V in G.items(): add_vec(nm, V)

    # ---------------- aggregate + probes
    def fit_lr(X, y, l2=1e-2, iters=300):
        Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32); w = torch.zeros(Xt.shape[1], requires_grad=True)
        opt = torch.optim.LBFGS([w], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad(); l = F.binary_cross_entropy_with_logits(Xt @ w, yt) + l2 * (w * w).sum(); l.backward(); return l
        opt.step(closure); return w.detach()
    sets = sorted(set(q["set"] for q in pairs)); res = {"n_pairs": {s: sum(1 for q in pairs if q["set"] == s) for s in sets}, "reprs": {}}
    def probe(Va, Vb, idx):
        if Va is None or len(idx) < 60: return None
        docs = sorted(set(pairs[k]["doc"] for k in idx)); te_docs = set(d for d in docs if zlib.crc32(d.encode()) % 4 == 0)
        pr = random.Random(5); X, y, is_te = [], [], []
        for k in idx:
            d = (Va[k] - Vb[k]).float().numpy(); s_ = pr.random() < 0.5; X.append(d if s_ else -d); y.append(int(s_)); is_te.append(pairs[k]["doc"] in te_docs)
        X = np.stack(X); y = np.array(y); te = np.array(is_te); X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        if te.sum() < 20 or (~te).sum() < 40: return None
        w = fit_lr(X[~te], y[~te]); return float(((torch.tensor(X[te], dtype=torch.float32) @ w > 0).numpy().astype(int) == y[te]).mean())
    for name, (Va, Vb, cos, base) in R.items():
        rec = {"random_pair_1mcos_median": base, "sets": {}}
        for sname in sets:
            idx = [k for k, q in enumerate(pairs) if q["set"] == sname]; c = np.array([cos[k] for k in idx])
            ent = {"n": len(idx), "cos_mean": float(c.mean()), "rel_dist": (float(np.mean(1 - c)) / base) if base else None, "probe_acc": probe(Va, Vb, idx), "by_type": {}, "by_bucket": {}}
            for key, field in (("by_type", "type"), ("by_bucket", "bucket")):
                for v in sorted(set(pairs[k][field] for k in idx)):
                    jj = [k for k in idx if pairs[k][field] == v]; cc = np.array([cos[k] for k in jj])
                    ent[key][v] = {"n": len(jj), "cos_mean": float(cc.mean()), "rel_dist": (float(np.mean(1 - cc)) / base) if base else None}
            rec["sets"][sname] = ent
        res["reprs"][name] = rec
        print(f"[diag] {name[:60]:60s} " + " | ".join(f"{s}: cos {rec['sets'][s]['cos_mean']:.4f} rel {rec['sets'][s]['rel_dist'] if rec['sets'][s]['rel_dist'] is None else round(rec['sets'][s]['rel_dist'], 3)} probe {rec['sets'][s]['probe_acc']}" for s in sets), flush=True)
    res["examples"] = {s: [{k: q[k] for k in ("type", "bucket", "a", "b")} for q in pairs if q["set"] == s][:3] for s in sets}
    json.dump(res, open(a.out, "w"), indent=1); print(f"[diag] wrote {a.out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
