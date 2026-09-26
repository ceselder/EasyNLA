"""Iterated best-of-N distillation for the bullet verbalizer, with Gemma verification + hedged rewrites scored by the frozen critic.

round r: sample N explanations per anchor from the current verbalizer (vLLM, mine_av_rollouts.py) -> VERIFY (Gemma sees the text up to the
position + the true continuation + the stored model state; every bullet SUPPORTED / UNSUPPORTED / CONTRADICTED, up to K hedged generalizations
down a specificity ladder; rewrites may add no new specifics: enforced here) -> SELECT (critic single-claim PMI of verified bullets and valid
hedges on the anchor's activation; a hallucinated bullet is replaced by its best hedge if that scores > lambda, else deleted; specifics seen in
< 2 of the N samples are hedged (lucky-guess guard); best-of-N by the RL reward sum v_i (1 - d_i) - lambda * n with the max3 duplicate
discount, over verified/hedged claims only) -> SFT targets. Control arm: the candidate with the most Gemma-supported claims, unsupported
bullets dropped, no hedging.

  pool    (CPU)    anchors -> {WS}/pool_<tag>.parquet (SFT schema + text / continuation / model state) + sidecar
  verify  (Gemma)  pool + samples -> {WS}/verify_<tag>/<part>.parquet (per anchor: unique bullets, labels, hedges, rule checks)
  select  (GPU)    verify -> {WS}/sft_<tag>_{bon,ctrl}.parquet + stats / examples
"""
import argparse, glob, json, os, random, re, sys, time, zlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
WS = "/vol_glp/claims/bon"


# ------------------------------------------------------------------------------------------------------------------------- pool
def cmd_pool(a):
    import pyarrow as pa, pyarrow.parquet as pq, shutil
    prompt = pq.read_table("/vol_q36/data/sft/av_sft_train.parquet", columns=["prompt"]).slice(0, 1).column(0).to_pylist()[0]
    files = [f"/vol_glp/claims/anchors/anchors_{n}.parquet" for n in a.shards.split(",")]
    rng = random.Random(a.seed); rows = []
    excl = set()
    for f in (a.exclude.split(",") if a.exclude else []):
        if os.path.exists(f): excl |= set(pq.read_table(f, columns=["anchor_id"]).column(0).to_pylist())
    cols = ["anchor_id", "doc_id", "is_val", "prefix_text", "cont_text", "top10_tokens", "top10_probs", "entropy", "jlens_tokens", "activation_vector"]
    for f in files:
        t = pq.read_table(f, columns=cols).to_pylist()
        rows += [r for r in t if (r["is_val"] == (a.split == "val")) and r["anchor_id"] not in excl]
    rng.shuffle(rows); rows = rows[: a.n]
    for r in rows: r["prompt"] = [{"content": m["content"], "role": m["role"]} for m in prompt]; r["activation_layer"] = 42
    sch = pa.schema([("anchor_id", pa.string()), ("doc_id", pa.string()), ("prompt", pa.list_(pa.struct([("content", pa.string()), ("role", pa.string())]))),
                     ("activation_vector", pa.list_(pa.float32(), 5120)), ("activation_layer", pa.int64()), ("prefix_text", pa.string()), ("cont_text", pa.string()),
                     ("top10_tokens", pa.list_(pa.string())), ("top10_probs", pa.list_(pa.float64())), ("entropy", pa.float64()), ("jlens_tokens", pa.list_(pa.string()))])
    os.makedirs(WS, exist_ok=True); out = f"{WS}/pool_{a.tag}.parquet"
    with pq.ParquetWriter(out, sch, compression="zstd") as w:
        for b0 in range(0, len(rows), 2048): w.write_table(pa.Table.from_pylist([{k: r[k] for k in sch.names} for r in rows[b0: b0 + 2048]], schema=sch))
    shutil.copy2("/vol_q36/data/sft/av_sft_train.parquet.nla_meta.yaml", out + ".nla_meta.yaml")
    print(f"[bon-pool] {a.tag}: {len(rows)} {a.split} anchors from {len(files)} shards -> {out}", flush=True)


# ------------------------------------------------------------------------------------------------------------------------- verify
VERIFY_SYS = """You check claims that a verbalizer wrote about a language model's internal state at the END of a text (the model is about to predict the next token).
You get: TEXT (everything the model has read; the model is at its very end), CONTINUATION (the true next words, which the model has NOT seen), MODEL STATE (the model's actual next-token distribution and related readouts at that position), and numbered CLAIMS.

For every claim decide:
- S (supported): true of the TEXT, or (for claims about what the model expects / predicts / is thinking about) consistent with MODEL STATE, or (for claims about what comes next) borne out by the CONTINUATION or clearly implied by the text;
- U (unsupported): not established by TEXT, MODEL STATE or CONTINUATION (e.g. invented details, guesses that the continuation does not bear out);
- C (contradicted): conflicts with the TEXT, MODEL STATE or CONTINUATION.
For every U or C claim, write up to 4 HEDGED versions that go DOWN a specificity ladder (partial -> category -> generic), each one TRUE and a GENERALIZATION of the claim: replace the wrong detail by what it is an instance of (e.g. "names the actresses Kirstie Alley and Polly Holliday" -> "names two actresses who play twins" -> "names two actresses" -> "mentions actors"). A hedge must NEVER add a name, number, quote or other specific detail that is not already in the claim. If nothing true remains, give no hedges. Claims about what comes next whose content the CONTINUATION does not bear out get NO hedges.
For every S claim that states a specific name, number or quote, also give 1-2 hedged versions (same rules) so the specific can be softened if needed.
Answer with JSON only: {"claims": [{"i": <number>, "label": "S"|"U"|"C", "hedges": ["...", ...]}, ...]} covering every claim number."""


def _state(r, k=6):
    tt = [f"{t!r} {p:.2f}" for t, p in zip((r["top10_tokens"] or [])[:k], (r["top10_probs"] or [])[:k])]
    return (f"next-token top-{k}: " + ", ".join(tt) + f"\nnext-token entropy: {r['entropy']:.2f} nats (low < 1.5 = easy to predict, high > 4 = hard)\n"
            f"words the model is thinking about (J-lens): " + ", ".join(repr(x.strip()) for x in (r["jlens_tokens"] or [])[:10]))


def verify_msgs(r, bullets, text_chars=3000):
    user = (f"TEXT (the model is at the very end):\n<<<{(r['prefix_text'] or '')[-text_chars:]}>>>\n\nCONTINUATION (not seen by the model):\n<<<{(r['cont_text'] or '')[:400]}>>>\n\n"
            f"MODEL STATE:\n{_state(r)}\n\nCLAIMS:\n" + "\n".join(f"{i}. {b}" for i, b in enumerate(bullets)))
    return [{"role": "system", "content": VERIFY_SYS}, {"role": "user", "content": user}]


_NUM = re.compile(r"\d[\d,.:/]*")
_QUOTE = re.compile(r"[\"“”'‘’`]([^\"“”'‘’`]{2,})[\"“”'‘’`]")
_CAP = re.compile(r"\b[A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+)*")
_STOP = {"The", "A", "An", "This", "That", "It", "Its", "Text", "Document", "Model", "Language", "Claims", "No", "Two", "Three", "One", "Some", "Several", "Many"}


def specifics(s):
    """numbers, quoted spans and capitalised name phrases (sentence-initial function words dropped)"""
    out = set(n.strip(".,:") for n in _NUM.findall(s)) | {q.strip().lower() for q in _QUOTE.findall(s)}
    for m in _CAP.finditer(s):
        ph = " ".join(w for w in m.group(0).split() if w not in _STOP)
        if ph: out.add(ph.lower())
    return {x for x in out if x}


def hedge_ok(orig, hedge):
    """a hedge may not introduce a specific (number / quote / name phrase) absent from the original claim (substring match, lowercased)"""
    o = orig.lower()
    return all(x in o for x in specifics(hedge))


def parse_verify(txt, n):
    if not txt: return None
    m = re.search(r"\{.*\}", txt, re.S)
    if not m: return None
    try: d = json.loads(m.group(0))
    except Exception: return None
    out = {}
    for c in d.get("claims", []):
        try: i = int(c.get("i"))
        except Exception: continue
        if 0 <= i < n and c.get("label") in ("S", "U", "C"): out[i] = {"label": c["label"], "hedges": [h for h in (c.get("hedges") or []) if isinstance(h, str) and h.strip()][:4]}
    return out


def cmd_verify(a):
    """(inside the Gemma container: server started by modal_claims_gemma) pool + mined samples -> per-anchor verification parquet"""
    import pyarrow as pa, pyarrow.parquet as pq
    import claims_gemma as cg
    from nla.flow.claims import split_claims
    P = {r["anchor_id"]: r for r in pq.read_table(f"{WS}/pool_{a.tag}.parquet", columns=["anchor_id", "doc_id", "prefix_text", "cont_text", "top10_tokens", "top10_probs", "entropy", "jlens_tokens"]).to_pylist()}
    ids = list(P); S = {}
    for f in sorted(glob.glob(f"{a.samples_dir}/*.parquet")):
        for r in pq.read_table(f).to_pylist(): S.setdefault(ids[r["row_idx"]], []).append((r["sample_idx"], r["explanation"]))
    todo = [aid for aid in ids if aid in S][a.shard::a.nshards]
    out_dir = f"{WS}/verify_{a.tag}"; os.makedirs(out_dir, exist_ok=True); dst = f"{out_dir}/part_{a.shard:03d}.parquet"
    if os.path.exists(dst): print(f"[bon-verify] {dst} exists", flush=True); return
    items, meta = [], []
    for aid in todo:
        samples = [split_claims(e) if e else [] for _, e in sorted(S[aid])]
        uniq = list(dict.fromkeys(b for s in samples for b in s))
        for c0 in range(0, len(uniq), a.chunk):
            ch = uniq[c0: c0 + a.chunk]; items.append((None, None, verify_msgs(P[aid], ch))); meta.append((aid, c0, ch))
    proc, t_start = cg.start_server([], 256, a.port, open("/tmp/server_bon_verify.log", "w"), max_model_len=8192)
    if t_start is None: raise SystemExit("gemma server did not start: " + open("/tmp/server_bon_verify.log").read()[-2000:])
    try:
        t0 = time.time(); texts, tok = cg.run_items(items, a.port, a.conc, max_tokens=a.max_tokens, temperature=0.0)
    finally:
        cg.stop_server(proc)
    res = {}
    for (aid, c0, ch), txt in zip(meta, texts):
        d = parse_verify(txt, len(ch)) or {}
        for i, b in enumerate(ch):
            v = d.get(i); hed = [h for h in (v["hedges"] if v else []) if hedge_ok(b, h)]
            res.setdefault(aid, []).append({"bullet": b, "label": v["label"] if v else None, "hedges": hed, "hedges_rejected": [h for h in (v["hedges"] if v else []) if not hedge_ok(b, h)]})
    recs = []
    for aid in todo:
        samples = [split_claims(e) if e else [] for _, e in sorted(S[aid])]
        recs.append({"anchor_id": aid, "doc_id": P[aid]["doc_id"], "samples": [json.dumps(s, ensure_ascii=False) for s in samples], "raw": [e or "" for _, e in sorted(S[aid])],
                     "verdicts": json.dumps(res.get(aid, []), ensure_ascii=False)})
    tmp = dst + ".tmp"; pq.write_table(pa.Table.from_pylist(recs), tmp, compression="zstd"); os.replace(tmp, dst)
    n_b = sum(len(v) for v in res.values()); n_parse = sum(1 for v in res.values() for x in v if x["label"])
    print(f"[bon-verify] {a.tag} part {a.shard}: {len(todo)} anchors, {n_b} unique bullets, labelled {n_parse / max(n_b, 1):.3f}, {len(items)} calls in {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------------------------------------------------------------- select
def cmd_select(a):
    import torch, pyarrow as pa, pyarrow.parquet as pq
    from nla.flow.scoring import FlowBundle
    from nla.flow.claims import format_claims
    from nla.flow.claim_redundancy import EmbSim, NLISim, LexSim, MaxSim, semdup_score
    from claims_controls import Scorer
    dev = "cuda:0"; lam = a.lam
    V = [r for f in sorted(glob.glob(f"{WS}/verify_{a.tag}/part_*.parquet")) for r in pq.read_table(f).to_pylist()]
    P = {r["anchor_id"]: r for r in pq.read_table(f"{WS}/pool_{a.tag}.parquet", columns=["anchor_id", "prompt", "activation_vector"]).to_pylist()}
    aa = torch.load(a.adapter, map_location="cpu")["args"]
    fb = FlowBundle(aa["prior"], a.adapter, aa["stats"], dev, base="Qwen/Qwen3.6-27B", enc_layer=aa.get("enc_layer", 42), ar_ckpt=aa.get("ar_ckpt", "/vol/ckpts/qwen36_27b/ar_sft_merged"))
    fmt = (lambda c: format_claims([c])) if aa.get("claim_subsets", 0) > 0 else (lambda c: c)
    sc = Scorer(fb, fmt, dev, a.D); sim = MaxSim(NLISim(device=dev), EmbSim(device=dev), LexSim(4))
    rows = {"bon": [], "ctrl": []}; st = {"anchors": 0, "bullets": 0, "S": 0, "U": 0, "C": 0, "unlabelled": 0, "hedged": 0, "deleted": 0, "guard_fired": 0, "hedges_rejected": 0,
                                          "by_kind": {}, "bon_claims": [], "bon_hedged_share": [], "ctrl_claims": [], "bon_reward": [], "bon_critic_value": []}
    ex = []
    kind = lambda b: ("next-token" if re.search(r"\b(next (word|token)|likely followed|expects|completion|complete)\b", b, re.I) else "quote" if _QUOTE.search(b)
                      else "number" if _NUM.search(b) else "name" if any(x for x in specifics(b)) else "general")
    for n_, r in enumerate(V):
        aid = r["anchor_id"]; ver = {x["bullet"]: x for x in json.loads(r["verdicts"])}; samples = [json.loads(s) for s in r["samples"]]; raws = [x.lower() for x in r["raw"]]
        if not ver or aid not in P: continue
        st["anchors"] += 1
        # candidate claims to score: verified bullets + valid hedges
        need = set()
        for b, x in ver.items():
            if x["label"] == "S": need.add(b)
            need |= set(x["hedges"])
        need = sorted(need); X = fb.norm.normalize(torch.tensor([P[aid]["activation_vector"]]).float().to(dev)).float()
        pm = dict(zip(need, sc.pmi_matrix(X, need, [17_000_003 + n_]).numpy()[0].tolist())) if need else {}
        for b, x in ver.items():
            st["bullets"] += 1; lab = x["label"] or "unlabelled"; st[lab if lab in ("S", "U", "C") else "unlabelled"] += 1; st["hedges_rejected"] += len(x["hedges_rejected"])
            k = st["by_kind"].setdefault(kind(b), {"n": 0, "S": 0, "U": 0, "C": 0, "none": 0}); k["n"] += 1; k[lab if lab in ("S", "U", "C") else "none"] += 1
        def best_hedge(b):
            hs = [(pm.get(h, -1e9), h) for h in ver[b]["hedges"]]; hs = [x for x in hs if x[0] > lam]
            return max(hs)[1] if hs else None
        def rewrite(cand, guard=True):
            out, nh, nd, ng = [], 0, 0, 0
            for b in cand:
                x = ver.get(b)
                if not x or not x["label"]: nd += 1; continue
                if x["label"] == "S":
                    sp = specifics(b)
                    if guard and sp and any(sum(s_ in rr for rr in raws) < 2 for s_ in sp):   # lucky-guess guard: a specific must appear in >= 2 of the N samples
                        ng += 1; h = best_hedge(b)
                        if h: out.append((h, True)); nh += 1
                        else: nd += 1
                    else: out.append((b, False))
                else:
                    h = best_hedge(b)
                    if h: out.append((h, True)); nh += 1
                    else: nd += 1
            seen, uo = set(), []
            for c, hdg in out:
                if c.lower() not in seen: seen.add(c.lower()); uo.append((c, hdg))
            return uo, nh, nd, ng
        best, info = None, None
        for s in samples:
            if not s: continue
            cl, nh, nd, ng = rewrite(s)
            if not cl: continue
            txt = [c for c, _ in cl]; val = semdup_score(txt, [pm.get(c, 0.0) for c in txt], sim, 0.5); rw = val - lam * len(txt)
            key = (rw, -len(txt))
            if best is None or key > best[0]: best = (key, cl, val); info = (nh, nd, ng, s)
        if best:
            (rw, _), cl, val = best; nh, nd, ng, s0 = info
            st["hedged"] += nh; st["deleted"] += nd; st["guard_fired"] += ng
            st["bon_claims"].append(len(cl)); st["bon_hedged_share"].append(sum(h for _, h in cl) / len(cl)); st["bon_reward"].append(rw); st["bon_critic_value"].append(val)
            rows["bon"].append({"prompt": P[aid]["prompt"], "activation_vector": P[aid]["activation_vector"], "activation_layer": 42, "doc_id": r["doc_id"], "anchor_id": aid,
                                "response": "<explanation>\n" + "\n".join(f"• {c}" for c, _ in cl) + "\n</explanation>"})
            if len(ex) < 12: ex.append({"anchor_id": aid, "before": s0, "after": [c for c, _ in cl], "hedged": [c for c, h in cl if h],
                                        "verdicts": {b: {"label": ver[b]["label"], "hedges": ver[b]["hedges"], "rejected": ver[b]["hedges_rejected"]} for b in s0 if b in ver}})
        # control: most Gemma-supported claims (dedup), unsupported dropped, no hedging
        bc = None
        for s in samples:
            sup = list(dict.fromkeys(b for b in s if ver.get(b, {}).get("label") == "S"))
            if sup:
                nsup = semdup_score(sup, [1.0] * len(sup), sim, 0.5)
                if bc is None or (nsup, -len(sup)) > bc[0]: bc = ((nsup, -len(sup)), sup)
        if bc:
            st["ctrl_claims"].append(len(bc[1]))
            rows["ctrl"].append({"prompt": P[aid]["prompt"], "activation_vector": P[aid]["activation_vector"], "activation_layer": 42, "doc_id": r["doc_id"], "anchor_id": aid,
                                 "response": "<explanation>\n" + "\n".join(f"• {c}" for c in bc[1]) + "\n</explanation>"})
        if n_ % 200 == 0: print(f"[bon-select] {a.tag}: {n_ + 1}/{len(V)} anchors", flush=True)
    sch = pa.schema([("prompt", pa.list_(pa.struct([("content", pa.string()), ("role", pa.string())]))), ("activation_vector", pa.list_(pa.float32(), 5120)), ("activation_layer", pa.int64()),
                     ("doc_id", pa.string()), ("anchor_id", pa.string()), ("response", pa.string())])
    import shutil
    for arm, rs in rows.items():   # SFT-ready dirs for modal_nla_exp.py --task sft --data-dir: av_sft_train.parquet (+ the v1b held-out test file) + sidecars
        random.Random(1).shuffle(rs); d_ = f"{WS}/sft_{a.tag}_{arm}"; os.makedirs(d_, exist_ok=True); out = f"{d_}/av_sft_train.parquet"
        with pq.ParquetWriter(out, sch, compression="zstd") as w:
            for b0 in range(0, len(rs), 2048): w.write_table(pa.Table.from_pylist(rs[b0: b0 + 2048], schema=sch))
        shutil.copy2("/vol/data/qwen36_27b_ws_v1b/av_sft_test.parquet", f"{d_}/av_sft_test.parquet")
        for f_ in ("av_sft_train", "av_sft_test"): shutil.copy2("/vol_q36/data/sft/av_sft_train.parquet.nla_meta.yaml", f"{d_}/{f_}.parquet.nla_meta.yaml")
    summ = {k: v for k, v in st.items() if not isinstance(v, list)}
    for k in ("bon_claims", "bon_hedged_share", "ctrl_claims", "bon_reward", "bon_critic_value"): summ[k + "_mean"] = float(np.mean(st[k])) if st[k] else None
    nb = max(st["bullets"], 1); summ.update(supported_frac=st["S"] / nb, unsupported_frac=st["U"] / nb, contradicted_frac=st["C"] / nb, unlabelled_frac=st["unlabelled"] / nb,
                                            n_bon=len(rows["bon"]), n_ctrl=len(rows["ctrl"]), lam=lam)
    json.dump({"summary": summ, "examples": ex}, open(f"{WS}/select_{a.tag}.json", "w"), indent=1)
    print(json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool"); p.add_argument("--tag", required=True); p.add_argument("--shards", required=True); p.add_argument("--n", type=int, default=2000)
    p.add_argument("--split", choices=["val", "train"], default="val"); p.add_argument("--seed", type=int, default=0); p.add_argument("--exclude", default="", help="comma list of earlier pool parquets")
    v = sub.add_parser("verify"); v.add_argument("--tag", required=True); v.add_argument("--samples-dir", required=True); v.add_argument("--shard", type=int, default=0); v.add_argument("--nshards", type=int, default=1)
    v.add_argument("--port", type=int, default=8000); v.add_argument("--conc", type=int, default=256); v.add_argument("--chunk", type=int, default=24); v.add_argument("--max-tokens", type=int, default=3000)
    s = sub.add_parser("select"); s.add_argument("--tag", required=True); s.add_argument("--adapter", required=True); s.add_argument("--lam", type=float, default=6.93); s.add_argument("--D", type=int, default=2)
    a = ap.parse_args(); {"pool": cmd_pool, "verify": cmd_verify, "select": cmd_select}[a.cmd](a)
