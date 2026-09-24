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
import argparse, glob, gzip, json, os, random, re, sys, time, zlib
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
        out = [None] * len(claims); per = max(1, rows_per_fwd // R)
        for k0 in range(0, len(claims), per):
            kk = list(range(k0, min(len(claims), k0 + per))); enc, mk, cv = fb.cond([fmt(claims[k]) for k in kk])
            xs = torch.cat([XT[owner[k]] for k in kk]); ts = tt.repeat(D * len(kk)); tg = torch.cat([TG[owner[k]] for k in kk])
            sel = torch.arange(len(kk), device=dev).repeat_interleave(R)
            with torch.autocast("cuda", dtype=torch.bfloat16): v = fb.model(xs, ts, enc[sel], mk[sel], cv[sel] if cv is not None else None).float()
            L = ((v - tg) ** 2).mean(-1).view(len(kk), R).mean(-1)
            for j, k in enumerate(kk): out[k] = float((d / 2) * (LU[owner[k]] - L[j]))
    return out


def cmd_score(a):
    """score candidate bullets of one part (a gold shard or a synthetic text shard) with a critic -> scores parquet"""
    import torch, pyarrow as pa, pyarrow.parquet as pq
    from nla.flow.scoring import FlowBundle
    from nla.flow.claims import format_claims
    dev = "cuda:0"; aa = torch.load(a.adapter, map_location="cpu")["args"]
    fb = FlowBundle(aa["prior"], a.adapter, aa["stats"], dev, base="Qwen/Qwen3.6-27B", enc_layer=aa.get("enc_layer", 42), ar_ckpt=aa.get("ar_ckpt", "/vol/ckpts/qwen36_27b/ar_sft_merged")); fb.model.eval()
    fmt = (lambda c: format_claims([c])) if aa.get("claim_subsets", 0) > 0 else (lambda c: c)
    out_dir = f"{WS}/scores_{a.critic_tag}"; os.makedirs(out_dir, exist_ok=True)
    for part in a.parts.split(","):
        dst = f"{out_dir}/{part.replace('/', '_')}.parquet"
        if os.path.exists(dst): continue
        t0 = time.time()
        if part.startswith("gold:"):   # gold:<shard_name>
            name = part[5:]; S = pq.read_table(f"{WS}/split/split_{name}.parquet").to_pylist()
            A = pq.read_table(f"/vol_q36/data/acts_qwen36_L42/{name}.parquet", columns=["activation_vector"]).column(0)
            recs = [dict(id=f"gold:{name}:{r['row']}", doc_id=r["doc_id"], is_val=r["is_val"], source="gold_split", claims=r["bullets"], types=["gold"] * len(r["bullets"])) for r in S]
            acts = [A[r["row"]].values.to_numpy(zero_copy_only=False) for r in S]
        else:                           # syn:<text shard name>: held-out anchors, internal + text claims (final) + Gemma multi claims
            name = part[4:]; F = pq.read_table(f"/vol_glp/claims/final/final_{name}.parquet", columns=["anchor_id", "doc_id", "is_val", "claims", "families", "types", "activation_vector"]).to_pylist()
            M = {r["anchor_id"]: r for r in (pq.read_table(f"{WS}/multi/multi_{name}.parquet").to_pylist() if os.path.exists(f"{WS}/multi/multi_{name}.parquet") else [])}
            recs, acts = [], []
            for r in F:
                if not r["is_val"]: continue
                cl = [c for c, g in zip(r["claims"], r["families"]) if g in ("internal", "text")]; ty = [f"{g}:{(t_ or '').split('/')[0]}" for c, g, t_ in zip(r["claims"], r["families"], r["types"]) if g in ("internal", "text")]
                m = M.get(r["anchor_id"])
                if m: cl += m["claims"]; ty += [f"semantic:{t_.split('/')[0]}" for t_ in m["types"]]
                if len(cl) < 2: continue
                recs.append(dict(id=r["anchor_id"], doc_id=r["doc_id"], is_val=True, source="synthetic_multi", claims=cl, types=ty)); acts.append(np.asarray(r["activation_vector"], dtype=np.float32))
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


def cmd_build(a):
    """critic-filtered SFT sets: bullets with PMI > lambda, best-first, <= cap per activation, in the verbalizer's SFT schema"""
    import pyarrow as pa, pyarrow.parquet as pq
    prompt = pq.read_table("/vol_q36/data/sft/av_sft_train.parquet", columns=["prompt"]).slice(0, 1).column(0).to_pylist()[0]
    rows = {"train": [], "val": []}; st = {"activations": 0, "kept": 0, "bullets_in": 0, "bullets_kept": 0, "by_source": {}}
    for f in sorted(glob.glob(f"{WS}/scores_{a.critic_tag}/*.parquet")):
        for r in pq.read_table(f).to_pylist():
            st["activations"] += 1; st["bullets_in"] += len(r["claims"])
            keep = sorted([(p, c, t_) for p, c, t_ in zip(r["pmi"], r["claims"], r["types"]) if p is not None and p > a.lam], key=lambda x: -x[0])
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
    out = f"{WS}/sft_{a.critic_tag}"; os.makedirs(out, exist_ok=True)
    for k, v in rows.items():
        if v: pq.write_table(pa.Table.from_pylist(v), f"{out}/{k}.parquet", compression="zstd")
    st.update(train=len(rows["train"]), val=len(rows["val"]), lam=a.lam, cap=a.cap, critic=a.critic_tag, bullets_per_kept=st["bullets_kept"] / max(st["kept"], 1))
    json.dump(st, open(f"{out}/stats.json", "w"), indent=1)
    ex = random.Random(0).sample(rows["train"], min(12, len(rows["train"])))
    json.dump([{k: v for k, v in e.items() if k not in ("activation_vector", "prompt")} for e in ex], open(f"{out}/examples.json", "w"), indent=1)
    print(json.dumps(st, indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score"); s.add_argument("--adapter", required=True); s.add_argument("--critic-tag", required=True); s.add_argument("--parts", required=True); s.add_argument("--D", type=int, default=1)
    b = sub.add_parser("build"); b.add_argument("--critic-tag", required=True); b.add_argument("--lam", type=float, default=20.0); b.add_argument("--cap", type=int, default=6)
    a = ap.parse_args(); {"score": cmd_score, "build": cmd_build}[a.cmd](a)
