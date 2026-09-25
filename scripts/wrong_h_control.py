"""Wrong-activation control for the flow critic's pairwise detail tests: does the critic prefer the true text over its perturbed twin BECAUSE it
reads the activation h, or because the twin text is recognisable on its own (a claim-only detector)?

Every (true, perturbed) pair is scored twice with IDENTICAL texts, noise draws and noise levels: once against its own activation h and once
against a wrong activation h' (a held-out row of ANOTHER document at the nearest token position, deterministic). Pairwise accuracy
P(loss(perturbed) > loss(true)) is reported for both; only own - wrong counts as reading the activation.
Tests (all on held-out av_sft_val rows):
  wrong_detail  the trainer's in-training eval exactly (first 1024 non-empty rows, make_negative with Random(2): one quote / number / name swapped;
                t in {0.3, 0.5, 0.7}, one shared noise draw per row and t)
  numbers       the controlled number-edit set (halluc_classify_numbers_sw_tokar.json: near / far / hedge / removed variants of each row's
                explanation; RL t grid 0.1..0.9, K=8 shared draws)
  ladder        held-out fact-sheet ladders (g2 pilot overlap rows; exact vs wrong-exact twin, hedge vs twin, omitted vs twin; RL grid, K=4)
Also: acceptance-threshold calibration from the wrong-activation distribution: reward r(z|h) = L_uncond(h) - L_cond(z|h) (PMI proxy) of the TRUE
text; lambda_95 = 95th percentile of r(z|h') over wrong activations; reported = the fraction of true texts / twins with r(.|h) > lambda_95.
usage: python scripts/wrong_h_control.py --critics tag/snap_N[,tag2/...] --out-dir /vol_glp/cond/scale_evals"""
import argparse, hashlib, json, os, random, sys, time
import numpy as np, torch, pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
VAL = "/vol_q36/data/sft/av_sft_val.parquet"
RL = [0.1, 0.3, 0.5, 0.7, 0.9]


def load_val():
    from nla.schema import extract_explanation
    t = pq.read_table(VAL, columns=["activation_vector", "response", "doc_id", "n_raw_tokens"])
    N = t.num_rows; acts = torch.tensor(np.asarray(t.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(N, -1))
    z = [(extract_explanation(r) or r or "").strip() for r in t.column("response").to_pylist()]
    return acts, z, t.column("doc_id").to_pylist(), [int(x) for x in t.column("n_raw_tokens").to_pylist()]


def wrong_rows(docs, npos):
    """raw row -> row of a DIFFERENT document at the nearest token position (ties broken by a hash, deterministic)"""
    order = sorted(range(len(docs)), key=lambda i: npos[i]); where = {r: k for k, r in enumerate(order)}; out = {}
    for r in range(len(docs)):
        k = where[r]; cands = []
        for j in range(max(0, k - 40), min(len(order), k + 41)):
            q = order[j]
            if docs[q] != docs[r]: cands.append((abs(npos[q] - npos[r]), hashlib.md5(f"{r}|{q}".encode()).hexdigest(), q))
        out[r] = min(cands)[2]
    return out


class Scorer:
    def __init__(self, fb): self.fb = fb

    @torch.no_grad()
    def losses(self, x0, texts, ts, eps):
        """x0 [1, d], texts [G], eps [K, d] -> cond losses [G, T] and uncond losses [T] (mean over the K draws), the critic's training metric"""
        fb = self.fb; enc, mk, cv = fb.cond(texts); G, K, d = len(texts), eps.shape[0], x0.shape[1]
        assert fb.last_shift is None, "resid-shift critics not supported here"
        out, outu = [], []
        for tt in ts:
            xt = (1 - tt) * x0 + tt * eps; tgt = eps - x0
            encB = enc.repeat_interleave(K, 0) if enc is not None else None; mkB = mk.repeat_interleave(K, 0) if mk is not None else None
            cvB = cv.repeat_interleave(K, 0) if cv is not None else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = fb.model(xt.repeat(G, 1), torch.full((G * K,), tt, device=x0.device), encB, mkB, cvB).float()
                vu = fb.model(xt, torch.full((K,), tt, device=x0.device)).float()
            out.append(fb.fm_err(v, tgt.repeat(G, 1)).view(G, K).mean(1)); outu.append(fb.fm_err(vu, tgt).mean())
        return torch.stack(out, 1).cpu().numpy(), torch.stack(outu).cpu().numpy()


def run_critic(critic, snap, V, W, out_dir):
    from nla.flow.scoring import FlowBundle
    from nla.flow.negatives import make_negative
    acts, zraw, docs, npos = V; dev = "cuda:0"
    ap = f"/vol_glp/cond/{critic}/adapter_latest.pt"; aa = torch.load(ap, map_location="cpu")["args"]; pco = os.path.join(os.path.dirname(ap), "prior_cotrained_latest.pt")
    fb = FlowBundle(aa["prior"], ap, aa["stats"], dev, base=snap, enc_layer=aa.get("enc_layer", 42), ar_ckpt=aa.get("ar_ckpt", "/vol/ckpts/qwen36_27b/ar_sft_merged"),
                    prior_override=pco if os.path.exists(pco) else None)
    fb.model.eval(); S = Scorer(fb); d = acts.shape[1]; res = {"critic": critic}; t0 = time.time()
    X = lambda r: fb.norm.normalize(acts[r][None].to(dev)).float()
    # ---- wrong_detail (the trainer's eval rows and negatives)
    rows = [i for i, z in enumerate(zraw) if z][:1024]; zs = [zraw[i] for i in rows]
    nrng = random.Random(2); negs = [make_negative(z, nrng, zs) for z in zs]
    wd = {"own": [], "wrong": [], "kind": [], "r_true_own": [], "r_true_wrong": [], "r_neg_own": []}
    for k, (r, z, (zn, kind)) in enumerate(zip(rows, zs, negs)):
        if zn is None: continue
        g = torch.Generator(device=dev).manual_seed(7000 + r); eps = torch.randn(1, d, device=dev, generator=g)
        for hname, rr in (("own", r), ("wrong", W[r])):
            L, Lu = S.losses(X(rr), [z, zn], [0.3, 0.5, 0.7], eps)
            wd[hname].append(float((L[1] > L[0]).mean()))
            if hname == "own": wd["r_true_own"].append(float((Lu - L[0]).mean())); wd["r_neg_own"].append(float((Lu - L[1]).mean()))
            else: wd["r_true_wrong"].append(float((Lu - L[0]).mean()))
        wd["kind"].append(kind)
    res["wrong_detail"] = wd; print(f"[ctrl {critic}] wrong_detail {len(wd['own'])} pairs ({time.time() - t0:.0f}s)", flush=True)
    # ---- numbers (controlled number-edit set)
    cj = json.load(open("/vol_glp/cond/halluc_classify_numbers_sw_tokar.json")); modes = [m for m in cj["modes"] if m != "orig"]; nu = {"modes": modes, "items": []}
    for it in cj["items"][:512]:
        r = it["row"]; texts = [it["variants"]["orig"]["text"]] + [it["variants"][m]["text"] for m in modes]
        g = torch.Generator(device=dev).manual_seed(5000 + r); eps = torch.randn(8, d, device=dev, generator=g); rec = {"row": r}
        for hname, rr in (("own", r), ("wrong", W[r])):
            L, Lu = S.losses(X(rr), texts, RL, eps); m_ = L.mean(1)
            rec[hname] = {m: bool(m_[1 + i] > m_[0]) for i, m in enumerate(modes)}
        nu["items"].append(rec)
    res["numbers"] = nu; print(f"[ctrl {critic}] numbers {len(nu['items'])} items ({time.time() - t0:.0f}s)", flush=True)
    # ---- ladder (held-out fact sheets; hedge_ladder_eval's texts)
    from hedge_ladder_eval import sentence
    P = [x for x in pq.read_table("/vol_glp/scale/g2pilot/g2_pilot_v3.parquet").to_pylist() if x["src"] == "opus_overlap" and x["facts"] and x["fact_ladders"]]
    la = {"items": []}
    for x in P:
        f = json.loads(x["facts"]); z0 = " ".join(s for s in [f"The text is about {f['topic']}." if f.get("topic") else "", f"Genre: {f['genre']}." if f.get("genre") else "",
                                                        f"At its end it is {f['doing']}." if f.get("doing") else ""] if s)
        facts = [y for y in json.loads(x["fact_ladders"]) if y.get("twin")][:3]
        if not facts or not z0: continue
        r = x["row"]; g = torch.Generator(device=dev).manual_seed(r); eps = torch.randn(4, d, device=dev, generator=g)
        for y in facts:
            rungs = {k: (z0 + " " + sentence(y, k)) if sentence(y, k) else (z0 if k == "omit" else None) for k in ("exact", "category", "omit", "twin")}
            names = [k for k, v in rungs.items() if v]; rec = {"row": r, "type": y["type"]}
            for hname, rr in (("own", r), ("wrong", W[r])):
                L, _ = S.losses(X(rr), [rungs[k] for k in names], RL, eps); rw = dict(zip(names, (-L.mean(1)).tolist())); rec[hname] = rw
            la["items"].append(rec)
    res["ladder"] = la; print(f"[ctrl {critic}] ladder {len(la['items'])} items ({time.time() - t0:.0f}s)", flush=True)
    # ---- summary
    s = {}
    for h in ("own", "wrong"): s[f"wrong_detail_{h}"] = float(np.mean(wd[h]))
    for kind in sorted(set(wd["kind"])):
        idx = [i for i, k in enumerate(wd["kind"]) if k == kind]
        for h in ("own", "wrong"): s[f"wrong_detail_{kind}_{h}"] = float(np.mean([wd[h][i] for i in idx]))
    lam = float(np.percentile(wd["r_true_wrong"], 95))
    s["accept_lambda95_from_wrong_h"] = lam; s["accept_true_own"] = float(np.mean(np.array(wd["r_true_own"]) > lam)); s["accept_twin_own"] = float(np.mean(np.array(wd["r_neg_own"]) > lam))
    s["true_text_own_beats_wrong_h"] = float(np.mean(np.array(wd["r_true_own"]) > np.array(wd["r_true_wrong"])))
    for m in modes:
        for h in ("own", "wrong"): s[f"numbers_{m}_{h}"] = float(np.mean([it[h][m] for it in nu["items"]]))
    for hi, lo in (("exact", "twin"), ("category", "twin"), ("omit", "twin")):
        for h in ("own", "wrong"):
            w = [it[h][hi] > it[h][lo] for it in la["items"] if hi in it[h] and lo in it[h]]; s[f"ladder_{hi}>{lo}_{h}"] = float(np.mean(w)) if w else None
    for k in list(s):
        if k.endswith("_own") and k[:-4] + "_wrong" in s and s[k] is not None and s[k[:-4] + "_wrong"] is not None and not k.startswith("accept"):
            s[k[:-4] + "_gap"] = s[k] - s[k[:-4] + "_wrong"]
    res["summary"] = s
    fn = f"{out_dir}/wrongh_{critic.replace('/', '__')}.json"; json.dump(res, open(fn, "w")); print(f"[ctrl {critic}] SUMMARY {json.dumps({k: round(v, 3) for k, v in s.items() if v is not None})}", flush=True)
    del fb; torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(); p.add_argument("--critics", required=True); p.add_argument("--out-dir", default="/vol_glp/cond/scale_evals"); a = p.parse_args()
    from huggingface_hub import snapshot_download
    snap = snapshot_download("Qwen/Qwen3.6-27B", token=os.environ.get("HF_TOKEN"), local_dir="/root/base_snap", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.py", "*.model", "*.tiktoken"])
    V = load_val(); W = wrong_rows(V[2], V[3]); os.makedirs(a.out_dir, exist_ok=True)
    print(f"[ctrl] {len(W)} val rows; wrong-h partner: other document, median |position gap| {np.median([abs(V[3][r] - V[3][q]) for r, q in W.items()]):.0f} tokens", flush=True)
    for c in a.critics.split(","): run_critic(c.strip(), snap, V, W, a.out_dir)


if __name__ == "__main__":
    main()
