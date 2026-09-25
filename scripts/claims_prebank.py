"""Double-difference twin control with a SAME-DOCUMENT bank (peer review item): when wrong-activation accuracy sits below 0.5, a plain own - wrong gap
misleads, so compare the true-vs-twin margin on the row's own activation with the same margin on activations of the SAME document where the
detail is not (yet) readable.

Per benchmark false pair (true claim, minimal twin): the detail = the longest word of the true claim absent from the twin; kept only if it occurs in
the row's prefix text. Positions (Qwen3.6-27B tokens of the prefix, L42 residual = output of decoder block 42, as in claims_extract):
  own    the last prefix token (checked against the stored activation: cosine reported)
  pre    the token just BEFORE the detail's first mention
  back k the tokens k = 1 / 4 / 16 before the last one (the detail usually still in view)
margin(h) = PMI(h; true) - PMI(h; twin) (FM proxy, D noise draws x 5 t, the same noise for both claims); double difference
DD_pre = margin(own) - margin(pre), DD_back_k = margin(own) - margin(back k); report P(margin(own) > 0), P(DD > 0), means, by claim type.
  python scripts/claims_prebank.py --adapter <path> --tag <tag>   -> /vol_glp/cond/compnla/prebank_<tag>.json"""
import argparse, json, os, re, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from claims_controls import Scorer, OUT
BACK = (1, 4, 16)
_W = re.compile(r"[\w][\w'’.-]*[\w]|[\w]")


def detail_word(true_c, false_c):
    fw = {w.lower() for w in _W.findall(false_c)}
    cand = [w for w in _W.findall(true_c) if w.lower() not in fw and len(w) >= 3]
    return max(cand, key=len) if cand else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--adapter", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--D", type=int, default=4)
    ap.add_argument("--base", default="Qwen/Qwen3.6-27B"); ap.add_argument("--layer", type=int, default=42)
    a = ap.parse_args(); dev = "cuda:0"; t0 = time.time()
    import pyarrow.parquet as pq
    from transformers import AutoModelForCausalLM, AutoTokenizer
    C = json.load(open(f"{OUT}/claims.json"))["items"]
    t = pq.read_table("/vol_q36/data/sft/av_sft_val_clean1.parquet", columns=["activation_vector", "detokenized_text_truncated", "n_raw_tokens"])
    txt = t.column("detokenized_text_truncated").to_pylist(); nraw = t.column("n_raw_tokens").to_pylist(); N = t.num_rows
    acts = torch.tensor(np.asarray(t.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(N, -1))
    tok = AutoTokenizer.from_pretrained(a.base)
    items, need = [], {}
    for it in C:
        r = int(it["row"]); text = txt[r]; enc = tok(text, add_special_tokens=False, return_offsets_mapping=True); ids = enc["input_ids"]; off = enc["offset_mapping"]; L = len(ids)
        for p in it["false_pairs"]:
            tc = it["true_claims"][p["true_index"]]; w = detail_word(tc["claim"], p["false_claim"])
            if not w: continue
            m = text.lower().find(w.lower())
            if m < 0: continue
            k = next((i for i, (s, e) in enumerate(off) if e > m), None)
            if k is None or k < 1: continue
            pos = {"own": L - 1, "pre": k - 1, **{f"back{b}": L - 1 - b for b in BACK if L - 1 - b >= 0}}
            items.append({"row": r, "type": tc["type"], "true": tc["claim"], "twin": p["false_claim"], "detail": w, "mention_tok": k, "len": L, "n_raw": nraw[r], "pos": pos})
            need.setdefault(r, (ids, set()))[1].update(pos.values())
    print(f"[prebank {a.tag}] {len(items)} pairs with a locatable detail over {len(need)} rows ({time.time() - t0:.0f}s)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    inner = model.model; owner = inner.language_model if hasattr(inner, "language_model") else inner; cap = {}
    hnd = owner.layers[a.layer].register_forward_hook(lambda _m, _i, out: cap.__setitem__("h", out[0] if isinstance(out, tuple) else out))
    H = {}
    with torch.no_grad():
        for r, (ids, ps) in need.items():
            inner(input_ids=torch.tensor([ids], device=dev), use_cache=False); h = cap.pop("h")[0]
            for p in ps: H[(r, p)] = h[p].float().cpu()
    hnd.remove(); del model, inner, owner; import gc; gc.collect(); torch.cuda.empty_cache()
    cosv = [float(torch.nn.functional.cosine_similarity(H[(r, len(ids) - 1)], acts[r], dim=0)) for r, (ids, _) in need.items()]
    print(f"[prebank {a.tag}] re-extracted own activation vs stored: cosine median {np.median(cosv):.4f} min {np.min(cosv):.4f} ({time.time() - t0:.0f}s)", flush=True)
    from nla.flow.scoring import FlowBundle
    from nla.flow.claims import format_claims
    aa = torch.load(a.adapter, map_location="cpu")["args"]; pov = os.path.join(os.path.dirname(a.adapter), "prior_cotrained_latest.pt")
    fb = FlowBundle(aa["prior"], a.adapter, aa["stats"], dev, base=a.base, enc_layer=aa.get("enc_layer", 42), ar_ckpt=aa.get("ar_ckpt", "/vol/ckpts/qwen36_27b/ar_sft_merged"),
                    prior_override=(pov if os.path.exists(pov) else None))
    fmt = (lambda c: format_claims([c])) if aa.get("claim_subsets", 0) > 0 else (lambda c: c)
    sc = Scorer(fb, fmt, dev, a.D)
    for q, it in enumerate(items):
        names = list(it["pos"]); X = fb.norm.normalize(torch.stack([H[(it["row"], it["pos"][k])] for k in names]).to(dev)).float()
        M = sc.pmi_matrix(X, [it["true"], it["twin"]], [9_000_003 + q] * len(names)).numpy()   # the same noise for every position of the pair
        it["margin"] = {k: float(M[j, 0] - M[j, 1]) for j, k in enumerate(names)}; it["pmi_true"] = {k: float(M[j, 0]) for j, k in enumerate(names)}
    def summ(P):
        o = {"n": len(P), "own_margin_pos": float(np.mean([p["margin"]["own"] > 0 for p in P])), "own_margin_mean": float(np.mean([p["margin"]["own"] for p in P]))}
        for k in ["pre"] + [f"back{b}" for b in BACK]:
            v = [p["margin"]["own"] - p["margin"][k] for p in P if k in p["margin"]]
            if v: o[f"dd_{k}_pos"] = float(np.mean([x > 0 for x in v])); o[f"dd_{k}_mean"] = float(np.mean(v)); o[f"{k}_margin_pos"] = float(np.mean([p["margin"][k] > 0 for p in P if k in p["margin"]]))
        return o
    res = {"adapter": a.adapter, "tag": a.tag, "D": a.D, "cos_own_vs_stored_median": float(np.median(cosv)), "all": summ(items), "by_type": {}}
    for ty in sorted({p["type"] for p in items}): res["by_type"][ty] = summ([p for p in items if p["type"] == ty])
    os.makedirs(OUT, exist_ok=True); json.dump({"summary": res, "pairs": items}, open(f"{OUT}/prebank_{a.tag}.json", "w"), indent=1)
    s = res["all"]; print(f"[prebank {a.tag}] {s['n']} pairs: own margin > 0 {s['own_margin_pos']:.3f}; pre-mention margin > 0 {s.get('pre_margin_pos', float('nan')):.3f}; "
                          f"DD(own - pre) > 0 {s.get('dd_pre_pos', float('nan')):.3f} (mean {s.get('dd_pre_mean', float('nan')):+.1f}); DD back1/4/16 > 0 "
                          + " / ".join(f"{s.get(f'dd_back{b}_pos', float('nan')):.3f}" for b in BACK), flush=True)
    print(json.dumps(res["by_type"], indent=1), flush=True)


if __name__ == "__main__":
    main()
