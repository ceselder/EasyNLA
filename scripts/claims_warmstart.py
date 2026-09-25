"""Verbalizer warm-start data for the compositional NLA: bullet lists of claims the critic can read from the activation (critic-filtered SFT).

  split  (Gemma container) the 728k gold Opus explanations (/vol_q36/data/acts_qwen36_L42/shard_*.parquet) split into atomic bullets
         -> /vol_glp/claims/warmstart/split/split_<shard>.parquet (row, doc_id, is_val, bullets)
  multi  (Gemma container) held-out synthetic anchors (is_val; no critic phase trains on them): 3 random aspects x 2 claims each, quote-checked
         -> /vol_glp/claims/warmstart/multi/multi_<name>.parquet (anchor_id, claims, types)
  score  (nla-glp GPU) single-claim PMI of every candidate bullet under a critic (FM proxy, D noise draws x 5 t, the same noise for all of an
         activation's bullets) -> /vol_glp/claims/warmstart/scores_<critic>/<part>.parquet
  build  (CPU) keep bullets with PMI > lambda, best-first, cap 6 -> SFT parquets in the verbalizer's schema (prompt, activation_vector, doc_id,
         response = "<explanation>\\n• c1\\n• c2 ...\\n</explanation>") + held-out split -> /vol_glp/claims/warmstart/sft_<critic>/{train,val}.parquet
"""
import argparse, glob, gzip, json, math, os, random, re, sys, time, zlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
WS = "/vol_glp/claims/warmstart"
GOLD = "/vol_q36/data/acts_qwen36_L42/shard_*.parquet"

SPLIT_SYS = """You split an explanation of a language model's internal state into ATOMIC CLAIMS.
The explanation describes what a language model is representing at one token of a document (topic, genre, entities, numbers, the current sentence, what comes next, ...).
Rewrite it as a list of short, self-contained claims:
- one fact per claim; split compound sentences; drop filler and hedging words but keep every specific (names, numbers, quoted words);
- each claim must stand alone (name the thing, never "it" or "this" without a referent);
- use only content that is in the explanation; do not add, infer or correct anything;
- keep section labels only when they carry meaning ("Genre: product review");
- at most 12 claims.
Output one claim per line, each starting with "• ", and nothing else."""


def split_items(files, rng):
    import pyarrow.parquet as pq
    out = []
    for f in files:
        name = os.path.basename(f)[:-8]; t = pq.read_table(f, columns=["explanation", "doc_id", "is_val"]).to_pydict()
        for r, (z, d, v) in enumerate(zip(t["explanation"], t["doc_id"], t["is_val"])):
            if z and z.strip(): out.append(({"shard": name, "row": r, "doc_id": d, "is_val": bool(v)}, [{"role": "system", "content": SPLIT_SYS}, {"role": "user", "content": z.strip()}]))
    return out


def run_split(root_out, files, commit=None):
    import pyarrow as pa, pyarrow.parquet as pq
    import claims_gemma as cg
    os.makedirs(f"{root_out}/split", exist_ok=True)
    todo = [f for f in files if not os.path.exists(f"{root_out}/split/split_{os.path.basename(f)[:-8]}.parquet")]
    if not todo: return 0
    proc, t_start = cg.start_server([], 512, 8000, open("/tmp/server_split.log", "w"))
    if t_start is None: raise SystemExit("gemma server did not start: " + open("/tmp/server_split.log").read()[-2000:])
    try:
        for f in todo:
            items = split_items([f], random.Random(0)); t0 = time.time()
            texts, tok = cg.run_items([(None, None, m) for _, m in items], 8000, 1024, max_tokens=400, temperature=0.3)
            recs = []
            for (meta, _), txt in zip(items, texts):
                b = [re.sub(r"^\s*[•\-*]\s*", "", l).strip() for l in (txt or "").splitlines() if re.match(r"^\s*[•\-*]\s+\S", l)]
                b = [x for x in b if len(x.split()) >= 3][:12]
                if b: recs.append(dict(meta, bullets=b))
            name = os.path.basename(f)[:-8]; tmp = f"{root_out}/split/split_{name}.parquet.tmp"
            pq.write_table(pa.Table.from_pylist(recs), tmp, compression="zstd"); os.replace(tmp, f"{root_out}/split/split_{name}.parquet")
            print(f"[ws-split] {name}: {len(items)} explanations -> {len(recs)} with bullets ({np.mean([len(r['bullets']) for r in recs]):.1f} each) in {time.time() - t0:.0f}s ({len(items) / (time.time() - t0):.0f}/s)", flush=True)
            if commit: commit()
    finally:
        cg.stop_server(proc)
    return len(todo)


def run_multi(root, root_out, names, commit=None):
    """3 random aspects x 2 claims for every held-out anchor of the given text shards (quote-checked, false twins kept for reference)"""
    import pyarrow as pa, pyarrow.parquet as pq
    import claims_gemma as cg
    from claims_semantic import SYSTEM_WIDE, sample_request_wide, user_msg
    os.makedirs(f"{root_out}/multi", exist_ok=True)
    todo = [n for n in names if not os.path.exists(f"{root_out}/multi/multi_{n}.parquet")]
    if not todo: return 0
    proc, t_start = cg.start_server([], 512, 8000, open("/tmp/server_multi.log", "w"))
    if t_start is None: raise SystemExit("gemma server did not start")
    try:
        for n in todo:
            rng = random.Random(zlib.crc32(n.encode())); items = []
            for l in gzip.open(f"{root}/text/text_{n}.jsonl.gz", "rt"):
                r = json.loads(l)
                if not r.get("is_val"): continue
                asp = []
                while len(asp) < 3:
                    q = sample_request_wide(r, rng)
                    if q["aspects"][0] not in asp: asp.append(q["aspects"][0])
                q = {"aspects": asp, "gran": q["gran"], "style": q["style"], "n": 6, "short": True}; r["_shard"] = n
                um = user_msg(r, q).replace("Aspect: ", "Aspects (write 2 claims for EACH of these three aspects): ", 1)
                items.append((r, q, [{"role": "system", "content": SYSTEM_WIDE}, {"role": "user", "content": um}]))
            t0 = time.time(); texts, tok = cg.run_items(items, 8000, 1024, max_tokens=500)
            recs, st = cg.verify(items, texts)
            tbl = pa.Table.from_pylist([{k: v for k, v in r.items() if k != "_shard"} for r in recs])
            tmp = f"{root_out}/multi/multi_{n}.parquet.tmp"; pq.write_table(tbl, tmp, compression="zstd"); os.replace(tmp, f"{root_out}/multi/multi_{n}.parquet")
            print(f"[ws-multi] {n}: {len(items)} held-out anchors -> {st['quote_ok']} claims (quote pass {st['quote_pass_rate']:.3f}) in {time.time() - t0:.0f}s", flush=True)
            if commit: commit()
    finally:
        cg.stop_server(proc)
    return len(todo)


# ---------------------------------------------------------------- critic scoring of candidate bullets (nla-glp GPU container)
def _score_block(fb, fmt, X, owner, claims, D, dev, seeds, rows_per_fwd=12000):
    """single-claim PMI (nats) of claims[k] against X[owner[k]]; noise per activation fixed by seeds (shared by all of its claims)"""
    import torch
    TS = [0.1, 0.3, 0.5, 0.7, 0.9]; tt = torch.tensor(TS, device=dev); T = len(TS); R = D * T; d = X.shape[1]
    uniq = sorted(set(owner)); XT, TV, TG, LU = {}, {}, {}, {}
    with torch.no_grad():
        for i in uniq:
            E = torch.randn(D, d, device=dev, generator=torch.Generator(device=dev).manual_seed(int(seeds[i])))
            x0 = X[i:i + 1]; xt = ((1 - tt)[None, :, None] * x0[None] + tt[None, :, None] * E[:, None, :]).reshape(R, d); tg = (E[:, None, :] - x0[None]).expand(D, T, d).reshape(R, d)
            with torch.autocast("cuda", dtype=torch.bfloat16): v0 = fb.model(xt, tt.repeat(D)).float()
            XT[i], TG[i], LU[i] = xt, tg, ((v0 - tg) ** 2).mean(-1).mean()
        out = [None] * len(claims); per = max(1, min(256, rows_per_fwd // R))   # <= 256 texts per encoder call (batch x heads must fit the Triton grid)
        for k0 in range(0, len(claims), per):
            kk = list(range(k0, min(len(claims), k0 + per))); enc, mk, cv = fb.cond([fmt(claims[k]) for k in kk])
            xs = torch.cat([XT[owner[k]] for k in kk]); ts = tt.repeat(D * len(kk)); tg = torch.cat([TG[owner[k]] for k in kk])
            sel = torch.arange(len(kk), device=dev).repeat_interleave(R)
            with torch.autocast("cuda", dtype=torch.bfloat16): v = fb.model(xs, ts, enc[sel], mk[sel], cv[sel] if cv is not None else None).float()
            L = ((v - tg) ** 2).mean(-1).view(len(kk), R).mean(-1)
            for j, k in enumerate(kk): out[k] = float((d / 2) * (LU[owner[k]] - L[j]))
    return out


def _load_part(part):
    """-> (recs, acts): candidate bullets of one part. gold:<shard> = Gemma-split gold explanations; syn:<text shard> = held-out synthetic anchors
    (internal + text claims of the final shard + Gemma multi-aspect claims). recs: id, doc_id, is_val, source, claims, types. Missing final shard -> ([], [])"""
    import pyarrow.parquet as pq
    if part.startswith("gold:"):   # gold:<shard_name>
        name = part[5:]; S = pq.read_table(f"{WS}/split/split_{name}.parquet").to_pylist()
        A = pq.read_table(f"/vol_q36/data/acts_qwen36_L42/{name}.parquet", columns=["activation_vector"]).column(0)
        recs = [dict(id=f"gold:{name}:{r['row']}", doc_id=r["doc_id"], is_val=r["is_val"], source="gold_split", claims=r["bullets"], types=["gold"] * len(r["bullets"])) for r in S]
        return recs, [A[r["row"]].values.to_numpy(zero_copy_only=False) for r in S]
    name = part[4:]
    if not os.path.exists(f"/vol_glp/claims/final/final_{name}.parquet"): print(f"[ws-score] {part}: no final shard, skipped", flush=True); return [], []
    F = pq.read_table(f"/vol_glp/claims/final/final_{name}.parquet", columns=["anchor_id", "doc_id", "is_val", "claims", "families", "types", "activation_vector"]).to_pylist()
    M = {r["anchor_id"]: r for r in (pq.read_table(f"{WS}/multi/multi_{name}.parquet").to_pylist() if os.path.exists(f"{WS}/multi/multi_{name}.parquet") else [])}
    recs, acts = [], []
    for r in F:
        if not r["is_val"]: continue
        cl = [c for c, g in zip(r["claims"], r["families"]) if g in ("internal", "text")]; ty = [f"{g}:{(t_ or '').split('/')[0]}" for c, g, t_ in zip(r["claims"], r["families"], r["types"]) if g in ("internal", "text")]
        m = M.get(r["anchor_id"])
        if m: cl += m["claims"]; ty += [f"semantic:{t_.split('/')[0]}" for t_ in m["types"]]
        if len(cl) < 2: continue
        recs.append(dict(id=r["anchor_id"], doc_id=r["doc_id"], is_val=True, source="synthetic_multi", claims=cl, types=ty)); acts.append(np.asarray(r["activation_vector"], dtype=np.float32))
    return recs, acts


def cmd_score(a):
    """score candidate bullets of one part (a gold shard or a synthetic text shard) with a critic -> scores parquet"""
    import torch, pyarrow as pa, pyarrow.parquet as pq
    from nla.flow.scoring import FlowBundle
    from nla.flow.claims import format_claims
    dev = "cuda:0"; aa = torch.load(a.adapter, map_location="cpu")["args"]
    fb = FlowBundle(aa["prior"], a.adapter, aa["stats"], dev, base="Qwen/Qwen3.6-27B", enc_layer=aa.get("enc_layer", 42), ar_ckpt=aa.get("ar_ckpt", "/vol/ckpts/qwen36_27b/ar_sft_merged"), prior_override=(os.path.join(os.path.dirname(a.adapter), "prior_cotrained_latest.pt") if os.path.exists(os.path.join(os.path.dirname(a.adapter), "prior_cotrained_latest.pt")) else None)); fb.model.eval()
    fmt = (lambda c: format_claims([c])) if aa.get("claim_subsets", 0) > 0 else (lambda c: c)
    out_dir = f"{WS}/scores_{a.critic_tag}"; os.makedirs(out_dir, exist_ok=True)
    for part in a.parts.split(","):
        dst = f"{out_dir}/{part.replace('/', '_')}.parquet"
        if os.path.exists(dst): continue
        t0 = time.time(); recs, acts = _load_part(part)
        if not recs: continue
        X = fb.norm.normalize(torch.tensor(np.stack(acts)).float().to(dev)).float()
        owner = [i for i, r in enumerate(recs) for _ in r["claims"]]; claims = [c for r in recs for c in r["claims"]]
        seeds = [zlib.crc32(r["id"].encode()) for r in recs]; pm = []
        for b0 in range(0, len(recs), 256):          # activations in blocks (noise tensors per block)
            ks = [k for k, o in enumerate(owner) if b0 <= o < b0 + 256]
            pm_b = _score_block(fb, fmt, X, [owner[k] for k in ks], [claims[k] for k in ks], a.D, dev, seeds); pm += pm_b
        o = 0
        for r in recs: r["pmi"] = pm[o: o + len(r["claims"])]; o += len(r["claims"])
        for r, v in zip(recs, acts): r["activation_vector"] = v.astype(np.float32).tolist()
        tmp = dst + ".tmp"; pq.write_table(pa.Table.from_pylist(recs), tmp, compression="zstd"); os.replace(tmp, dst)
        print(f"[ws-score] {part}: {len(recs)} activations, {len(claims)} bullets scored in {time.time() - t0:.0f}s", flush=True)


FUTURE_TYPES = {"text:next_word", "text:next_words", "text:next_sentence", "text:sentence_end", "text:line_break", "text:number_soon", "text:list_soon",
                "text:entity_soon", "semantic:upcoming", "semantic:upcoming_numbers"}   # state the TRUE continuation: critic training only, never verbalizer targets
T5, T2 = (0.1, 0.3, 0.5, 0.7, 0.9), (0.3, 0.7)


def _neg_sets(recs, K, seed):
    """per bullet: K negative activations from OTHER documents of the same part, drawn among the activations that carry a claim of the same
    template (synthetic; any activation of the part if fewer than 2K do, and always for gold bullets) -> [record][bullet] -> list of indices"""
    n = len(recs); doc = [r["doc_id"] for r in recs]; has = {}
    for i, r in enumerate(recs):
        for t_ in set(r["types"]): has.setdefault(t_, []).append(i)
    out = []
    for i, r in enumerate(recs):
        rng = random.Random(zlib.crc32(f"{seed}:{r['id']}".encode())); per = []
        for t_ in r["types"]:
            pool = has.get(t_) if t_ != "gold" else None
            if not pool or len(pool) < 2 * K: pool = None
            sel, tries = set(), 0
            while len(sel) < K and tries < 50 * K:
                j = rng.choice(pool) if pool else rng.randrange(n); tries += 1
                if doc[j] != doc[i]: sel.add(j)
            per.append(sorted(sel) or [(i + 1) % n])
        out.append(per)
    return out


def cmd_score_margin(a):
    """contrastive margin of every candidate bullet: single-claim PMI on its own activation (5 t, as `score`) and on K same-template activations of
    OTHER documents (t in T2); each activation keeps its own fixed noise (seeded by its id) and unconditional baseline, so PMIs are comparable across
    activations. margin_lse = pmi2_own - log mean_k exp(pmi2_neg_k), margin_mean = pmi2_own - mean_k pmi2_neg_k, rank = #negatives >= own"""
    import torch, pyarrow as pa, pyarrow.parquet as pq
    from nla.flow.scoring import FlowBundle
    from nla.flow.claims import format_claims
    dev = "cuda:0"; aa = torch.load(a.adapter, map_location="cpu")["args"]
    pov = os.path.join(os.path.dirname(a.adapter), "prior_cotrained_latest.pt")
    fb = FlowBundle(aa["prior"], a.adapter, aa["stats"], dev, base="Qwen/Qwen3.6-27B", enc_layer=aa.get("enc_layer", 42), ar_ckpt=aa.get("ar_ckpt", "/vol/ckpts/qwen36_27b/ar_sft_merged"),
                    prior_override=(pov if os.path.exists(pov) else None))
    fmt = (lambda c: format_claims([c])) if aa.get("claim_subsets", 0) > 0 else (lambda c: c)
    out_dir = f"{WS}/scores_margin_{a.critic_tag}"; os.makedirs(out_dir, exist_ok=True)
    tt5 = torch.tensor(T5, device=dev); i2 = [T5.index(t) for t in T2]; K = a.K
    for part in a.parts.split(","):
        dst = f"{out_dir}/{part.replace('/', '_')}.parquet"
        if os.path.exists(dst): continue
        t0 = time.time(); recs, acts = _load_part(part)
        if len(recs) < K + 2: continue
        n = len(recs); X = fb.norm.normalize(torch.tensor(np.stack(acts)).float().to(dev)).float(); d = X.shape[1]
        E = torch.cat([torch.randn(1, d, device=dev, generator=torch.Generator(device=dev).manual_seed(zlib.crc32(r["id"].encode()))) for r in recs])   # = `score` noise (D=1)
        L0 = torch.empty(n, len(T5), device=dev)
        with torch.no_grad():
            for b0 in range(0, n, 1024):                                   # unconditional baseline per (activation, t)
                xb, eb = X[b0: b0 + 1024], E[b0: b0 + 1024]; m = len(xb)
                xt = ((1 - tt5)[None, :, None] * xb[:, None] + tt5[None, :, None] * eb[:, None]).reshape(-1, d); tg = (eb - xb)[:, None].expand(m, len(T5), d).reshape(-1, d)
                with torch.autocast("cuda", dtype=torch.bfloat16): v0 = fb.model(xt, tt5.repeat(m)).float()
                L0[b0: b0 + m] = ((v0 - tg) ** 2).mean(-1).view(m, len(T5))
        negs = _neg_sets(recs, K, a.seed)
        B = [(i, k) for i, r in enumerate(recs) for k in range(len(r["claims"]))]
        cells = []                                                          # per bullet: (activation, t-index) rows; own at T5 then negatives at T2
        own_pm, neg_pm = [None] * len(B), [None] * len(B)
        per = max(1, min(256, a.rows_per_fwd // (len(T5) + K * len(T2))))
        with torch.no_grad():
            for c0 in range(0, len(B), per):
                bb = B[c0: c0 + per]; enc, mk, cv = fb.cond([fmt(recs[i]["claims"][k]) for i, k in bb])
                ai, ti, bi = [], [], []
                for q, (i, k) in enumerate(bb):
                    js = negs[i][k]
                    ai += [i] * len(T5) + [j for j in js for _ in T2]; ti += list(range(len(T5))) + [t for _ in js for t in i2]; bi += [q] * (len(T5) + len(js) * len(T2))
                ai_t = torch.tensor(ai, device=dev); ti_t = torch.tensor(ti, device=dev); bi_t = torch.tensor(bi, device=dev); L = torch.empty(len(ai), device=dev)
                for r0 in range(0, len(ai), a.rows_per_fwd):
                    s_ = slice(r0, r0 + a.rows_per_fwd); A_, T_, Q_ = ai_t[s_], ti_t[s_], bi_t[s_]; tv = tt5[T_]
                    xt = (1 - tv)[:, None] * X[A_] + tv[:, None] * E[A_]; tg = E[A_] - X[A_]
                    with torch.autocast("cuda", dtype=torch.bfloat16): v = fb.model(xt, tv, enc[Q_], mk[Q_], cv[Q_] if cv is not None else None).float()
                    L[s_] = ((v - tg) ** 2).mean(-1)
                pm = (d / 2) * (L0[ai_t, ti_t] - L)                        # per-row PMI proxy (nats)
                o = 0
                for q, (i, k) in enumerate(bb):
                    nk = len(negs[i][k]) * len(T2); own_pm[c0 + q] = pm[o: o + len(T5)].cpu(); neg_pm[c0 + q] = pm[o + len(T5): o + len(T5) + nk].view(-1, len(T2)).mean(-1).cpu(); o += len(T5) + nk
        o = 0
        for r in recs:
            m_ = len(r["claims"]); P5 = [own_pm[o + k] for k in range(m_)]; N_ = [neg_pm[o + k] for k in range(m_)]; o += m_
            r["pmi"] = [float(p.mean()) for p in P5]; p2 = [float(p[i2].mean()) for p in P5]; r["pmi2"] = p2
            r["neg_mean"] = [float(x.mean()) for x in N_]; r["neg_lse"] = [float(torch.logsumexp(x.double(), 0) - math.log(len(x))) for x in N_]
            r["neg_max"] = [float(x.max()) for x in N_]; r["rank"] = [int((x >= v).sum()) for x, v in zip(N_, p2)]
            r["margin_lse"] = [v - l for v, l in zip(p2, r["neg_lse"])]; r["margin_mean"] = [v - l for v, l in zip(p2, r["neg_mean"])]
        for r, v in zip(recs, acts): r["activation_vector"] = v.astype(np.float32).tolist()
        tmp = dst + ".tmp"; pq.write_table(pa.Table.from_pylist(recs), tmp, compression="zstd"); os.replace(tmp, dst)
        print(f"[ws-margin] {part}: {n} activations, {len(B)} bullets x (5 own + {K}x{len(T2)} negative rows) in {time.time() - t0:.0f}s", flush=True)


def _fam(t_):
    return "gold" if t_ == "gold" else t_.split(":")[0]


def cmd_margin_stats(a):
    """margin distributions by family / type and kept counts per threshold (with the absolute-lambda columns for reference) -> margin_stats.json"""
    import pyarrow.parquet as pq
    rows = []                                                               # (family, type, pmi, margin_lse, margin_mean, rank, future)
    for f in sorted(glob.glob(f"{WS}/scores_margin_{a.critic_tag}/*.parquet")):
        for r in pq.read_table(f, columns=["types", "pmi", "margin_lse", "margin_mean", "rank"]).to_pylist():
            for t_, p, ml, mm, rk in zip(r["types"], r["pmi"], r["margin_lse"], r["margin_mean"], r["rank"]): rows.append((_fam(t_), t_, p, ml, mm, rk, t_ in FUTURE_TYPES))
    out = {"critic": a.critic_tag, "K": None, "families": {}, "types": {}}
    def summ(sel):
        P = np.array([x[2] for x in sel]); ML = np.array([x[3] for x in sel]); MM = np.array([x[4] for x in sel]); RK = np.array([x[5] for x in sel])
        q = lambda v: {str(k): float(np.percentile(v, k)) for k in (10, 25, 50, 75, 90)}
        return {"n": len(sel), "margin_lse_pct": q(ML), "margin_mean_pct": q(MM), "pmi_pct": q(P), "rank0_share": float(np.mean(RK == 0)), "rank_le3_share": float(np.mean(RK <= 3)),
                "share_margin_lse_gt": {str(t): float(np.mean(ML > t)) for t in (-20, 0, 10, 20, 40)}, "share_pmi_gt": {str(t): float(np.mean(P > t)) for t in (20, 40, 80)}}
    fams = sorted({x[0] for x in rows})
    for fm in fams + ["all"]:
        sel = [x for x in rows if fm == "all" or x[0] == fm]
        if sel: out["families"][fm] = summ(sel)
    for t_ in sorted({x[1] for x in rows}):
        sel = [x for x in rows if x[1] == t_]
        if len(sel) >= 200: out["types"][t_] = dict(summ(sel), future=t_ in FUTURE_TYPES)
    json.dump(out, open(f"{WS}/scores_margin_{a.critic_tag}/margin_stats.json", "w"), indent=1)
    for fm, v in out["families"].items(): print(f"[margin] {fm}: n {v['n']}, margin_lse median {v['margin_lse_pct']['50']:.1f} (p25 {v['margin_lse_pct']['25']:.1f}, p75 {v['margin_lse_pct']['75']:.1f}), own beats all {100*v['rank0_share']:.0f}%, "
                                                  f"share margin>0 {v['share_margin_lse_gt']['0']:.2f}, pmi>20 {v['share_pmi_gt']['20']:.2f}", flush=True)


_NUM = re.compile(r"\d(?:[\d,.:/]*\d)?")


def _prefix_texts(part_file):
    """id -> document prefix up to the activation (gold: extraction shard `text`; synthetic: anchors_<name>.parquet `prefix_text`)"""
    import pyarrow.parquet as pq
    base = os.path.basename(part_file)[:-8]
    if base.startswith("gold:"):
        name = base[5:]; t = pq.read_table(f"/vol_q36/data/acts_qwen36_L42/{name}.parquet", columns=["text"]).column(0).to_pylist()
        return lambda id_: t[int(id_.rsplit(":", 1)[1])]
    name = base[4:]; f = f"/vol_glp/claims/anchors/anchors_{name}.parquet"
    if not os.path.exists(f): return lambda id_: None
    t = pq.read_table(f, columns=["anchor_id", "prefix_text"]).to_pydict(); m = dict(zip(t["anchor_id"], t["prefix_text"]))
    return m.get


def _number_ok(claim, type_, prefix, window):
    """decodability (diffusion-AR-nla probes): exact numbers are linearly readable from h42 only <= 1 token back. A claim quoting a number that
    occurs in the document prefix is kept only if that number sits in the last `window` characters; position claims and numbers that do not
    occur in the prefix (buckets, hedges) are exempt"""
    if window <= 0 or prefix is None or type_ == "text:position": return True
    tail = prefix[-window:]
    for n in _NUM.findall(claim):
        if n in prefix and n not in tail: return False
    return True


def cmd_build(a):
    """critic-filtered SFT sets: bullets with PMI > lambda, best-first, <= cap per activation, in the verbalizer's SFT schema"""
    import pyarrow as pa, pyarrow.parquet as pq
    prompt = pq.read_table("/vol_q36/data/sft/av_sft_train.parquet", columns=["prompt"]).slice(0, 1).column(0).to_pylist()[0]
    rows = {"train": [], "val": []}; st = {"activations": 0, "kept": 0, "bullets_in": 0, "bullets_kept": 0, "by_source": {}}
    st["number_dropped"] = 0
    for f in sorted(glob.glob(f"{WS}/{'scores_margin_' if a.margin is not None else 'scores_'}{a.critic_tag}/*.parquet")):
        pfx = _prefix_texts(f) if a.number_window > 0 else (lambda id_: None)
        for r in pq.read_table(f).to_pylist():
            st["activations"] += 1; st["bullets_in"] += len(r["claims"])
            if a.margin is not None:   # v2: contrastive margin over same-template activations of other documents, best-first; true-future types never become targets
                keep = sorted([(m_, c, t_) for m_, c, t_ in zip(r[a.margin_key], r["claims"], r["types"]) if m_ is not None and m_ > a.margin and t_ not in FUTURE_TYPES], key=lambda x: -x[0])
                if a.number_window > 0:
                    px = pfx(r["id"]); k0 = len(keep); keep = [x for x in keep if _number_ok(x[1], x[2], px, a.number_window)]; st["number_dropped"] += k0 - len(keep)
            else: keep = sorted([(p, c, t_) for p, c, t_ in zip(r["pmi"], r["claims"], r["types"]) if p is not None and p > a.lam], key=lambda x: -x[0])
            seen, sel = set(), []
            for p, c, t_ in keep:
                if c.lower() in seen: continue
                seen.add(c.lower()); sel.append((p, c, t_))
                if len(sel) >= a.cap: break
            if not sel: continue
            split = "val" if (r["is_val"] if r["source"] == "gold_split" else zlib.crc32(r["id"].encode()) % 20 == 0) else "train"
            rows[split].append({"prompt": prompt, "activation_vector": r["activation_vector"], "activation_layer": 42, "doc_id": r["doc_id"], "source": r["source"], "id": r["id"],
                                "response": "<explanation>\n" + "\n".join(f"• {c}" for _, c, _ in sel) + "\n</explanation>", "bullet_pmi": [p for p, _, _ in sel], "bullet_types": [t_ for _, _, t_ in sel]})
            st["kept"] += 1; st["bullets_kept"] += len(sel); s_ = st["by_source"].setdefault(r["source"], {"activations": 0, "kept": 0, "bullets": 0}); s_["activations"] += 1; s_["kept"] += 1; s_["bullets"] += len(sel)
    out = f"{WS}/sft_{a.critic_tag}" + (f"_margin{a.margin:g}" if a.margin is not None else ""); os.makedirs(out, exist_ok=True)
    for k, v in rows.items():
        if v: pq.write_table(pa.Table.from_pylist(v), f"{out}/{k}.parquet", compression="zstd")
    st.update(train=len(rows["train"]), val=len(rows["val"]), lam=a.lam, margin=a.margin, margin_key=a.margin_key, number_window=a.number_window, cap=a.cap, critic=a.critic_tag, bullets_per_kept=st["bullets_kept"] / max(st["kept"], 1))
    json.dump(st, open(f"{out}/stats.json", "w"), indent=1)
    ex = random.Random(0).sample(rows["train"], min(12, len(rows["train"])))
    json.dump([{k: v for k, v in e.items() if k not in ("activation_vector", "prompt")} for e in ex], open(f"{out}/examples.json", "w"), indent=1)
    print(json.dumps(st, indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score"); s.add_argument("--adapter", required=True); s.add_argument("--critic-tag", required=True); s.add_argument("--parts", required=True); s.add_argument("--D", type=int, default=1)
    b = sub.add_parser("build"); b.add_argument("--critic-tag", required=True); b.add_argument("--lam", type=float, default=20.0); b.add_argument("--cap", type=int, default=6)
    b.add_argument("--number-window", type=int, default=0, help="drop bullets quoting a document number that is not within the last N characters of the prefix (0 = off; v2: 12)")
    b.add_argument("--margin", type=float, default=None, help="v2: keep bullets with contrastive margin > this (reads scores_margin_<critic>)"); b.add_argument("--margin-key", default="margin_lse", choices=["margin_lse", "margin_mean"])
    m = sub.add_parser("score_margin"); m.add_argument("--adapter", required=True); m.add_argument("--critic-tag", required=True); m.add_argument("--parts", required=True)
    m.add_argument("--K", type=int, default=64); m.add_argument("--seed", type=int, default=0); m.add_argument("--rows-per-fwd", type=int, default=8192)
    ms = sub.add_parser("margin_stats"); ms.add_argument("--critic-tag", required=True)
    a = ap.parse_args(); {"score": cmd_score, "build": cmd_build, "score_margin": cmd_score_margin, "margin_stats": cmd_margin_stats}[a.cmd](a)
