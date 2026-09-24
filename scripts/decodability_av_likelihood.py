"""Decodability test 1 — is a wrong detail (quote / number / name) checkable from the layer-42 activation by the model best trained to read it?

For every (true explanation z, edited explanation z') pair we compute, under the WARM-START verbalizer (Qwen3.6-27B + SFT LoRA
iter_0007813, activation injected as a soft token exactly as in SFT/RL),
    Δ(h)      = log p_AV(z | h) − log p_AV(z' | h)                    (sum over response tokens; the shared prefix cancels)
    Δ(h_null) = the same under a MISMATCHED activation (K activations from other documents, and the mean activation)
    E         = Δ(h) − mean_k Δ(h_null_k)                             (the h-specific evidence for the true detail; the text prior cancels)
and report P(E > 0), the mean E (nats) and CIs per edit set / kind, next to the critics' numbers on the SAME items.

Edit sets (all held-out from the AV SFT: av_sft_val docs are hash-split from the SFT train docs):
  wrong_detail   av_sft_val rows 0..1023, nla.flow.negatives.make_negative with random.Random(2) (= train_cond / clip_eval / scale evals; 1,023 items)
  numbers        the controlled number test (halluc_classify: 512 av_sft_val rows whose gold explanation states a number that occurs in the
                 source; variants near / far / hedge / removed of the FIRST grounded number) — the exact texts of the critics' run
  twins          flow_noise/twins.json: 40 clean1 rows, Sonnet twins of the SOURCE text (entity / number swaps) — the AV scores raw document
                 text here, which is off-distribution for a verbalizer (caveat)
  deletions      flow_noise/deletions.json: 104 clean1 explanations with their false claims removed vs their true claims removed
  ladder         g2 pilot fact ladders on av_sft_val positions (hedge_ladder_eval items): base z0 + one fact at rungs exact / partial /
                 category / omit / twin, by fact type (person, organisation, place, other_entity, number, date, quote)
Outputs one JSON with every per-text, per-condition log-probability (so any pairwise comparison can be recomputed offline) and a summary.
usage (Modal, 1 B200):  python scripts/decodability_av_likelihood.py --out /vol_glp/decodability/av_likelihood.json
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, time
import numpy as np, torch, pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from nla.schema import extract_explanation, wrap_explanation
from nla.flow.negatives import make_negative

KIND = {"person": "a person", "organisation": "an organisation", "place": "a place", "other_entity": "a named item", "number": "a number", "date": "a date", "quote": "a phrase"}
LADDER_PAIRS = [("exact", "twin"), ("category", "twin"), ("omit", "twin"), ("partial", "twin"), ("exact", "omit"), ("exact", "partial"), ("partial", "category"), ("category", "omit")]


def ladder_sentence(x, rung):
    """= scripts/hedge_ladder_eval.sentence (copied: scripts/ is not a package)."""
    if rung == "omit": return None
    if rung == "twin": val = x.get("twin")
    else: val = x["value"] if rung == "exact" else (x["ladder"] or {}).get(rung)
    if not val: return None
    if x["type"] == "quote": return f"It contains “{val}”." if rung in ("exact", "twin") else f"It contains {val}."
    desc = f" ({x['desc']})" if x.get("desc") and rung in ("exact", "twin") else ""
    return f"It mentions {KIND[x['type']]}: {val}{desc}."


def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def build_groups(a):
    """-> groups: list of {set, kind, act: ("val"|"clean1", row), doc, texts: {name: text}, true: name}; plus the activation tables."""
    V = pq.read_table(a.val, columns=["prompt", "activation_vector", "response", "doc_id"])
    VA = torch.tensor(np.asarray(V.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(V.num_rows, -1))
    VZ = [(extract_explanation(r) or r or "").strip() for r in V.column("response").to_pylist()]; VD = V.column("doc_id").to_pylist()
    prompt_msgs = V.column("prompt").slice(0, 1).to_pylist()[0]
    C = pq.read_table(a.clean1, columns=["activation_vector", "response", "doc_id", "detokenized_text_truncated"])
    CA = torch.tensor(np.asarray(C.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(C.num_rows, -1))
    CZ = [(extract_explanation(r) or r or "").strip() for r in C.column("response").to_pylist()]; CD = C.column("doc_id").to_pylist(); CT = C.column("detokenized_text_truncated").to_pylist()
    groups = []; sets = set(a.sets.split(","))
    if "wrong_detail" in sets:   # identical construction to clip_eval.py / train_cond.py evaluate()
        nrng = random.Random(2); negs = [make_negative(z, nrng, VZ[:1024]) for z in VZ[:1024]]
        for i, (zn, kind) in enumerate(negs):
            if zn: groups.append({"set": "wrong_detail", "kind": kind, "act": ("val", i), "doc": VD[i], "texts": {"true": VZ[i], "alt": zn}, "true": "true"})
    if "numbers" in sets:
        cj = json.load(open(a.classify_json)); modes = [m for m in cj["modes"] if m != "orig"]
        for it in cj["items"][: 512]:
            r = it["row"]; texts = {"true": it["variants"]["orig"]["text"]}; texts.update({m: it["variants"][m]["text"] for m in modes})
            groups.append({"set": "numbers", "kind": "number", "act": ("val", r), "doc": VD[r], "texts": texts, "true": "true", "number": it["number"]})
    if "twins" in sets and os.path.exists(a.twins_json):
        tw = json.load(open(a.twins_json)); n_bad = 0
        for it in tw["items"]:
            r = it["row"]; texts = {}; recon = None
            for j, t in enumerate(it["twins"]):
                base = t["text"].replace(t["replacement"], t["span"], 1)
                if recon is None: recon = base
                elif base != recon: n_bad += 1; continue
                texts[f"twin{j}:{t['type']}"] = t["text"]
            if recon is None or not texts: continue
            texts["true"] = recon
            groups.append({"set": "twins", "kind": "source_twin", "act": ("clean1", r), "doc": CD[r], "texts": texts, "true": "true", "matches_clean1_text": recon.strip() == (CT[r] or "").strip()})
        if n_bad: print(f"[twins] {n_bad} twins did not reconstruct the same source text -> skipped", flush=True)
    if "deletions" in sets and os.path.exists(a.deletions_json):
        for it in json.load(open(a.deletions_json))["items"]:
            r = int(it["row"]); z = it["z"]
            if not (it.get("remove_false") and it.get("remove_true")): continue
            groups.append({"set": "deletions", "kind": it.get("av", ""), "act": ("clean1", r), "doc": CD[r], "texts": {"true": z, "remove_false": it["remove_false"], "remove_true": it["remove_true"]}, "true": "true",
                           "n_removed": int(it.get("n_removed", 0)), "n_false": int(it.get("n_false", 0))})
    if "ladder" in sets and os.path.exists(a.ladder_parquet):
        P = pq.read_table(a.ladder_parquet).to_pylist(); P = [r for r in P if r.get("src") == "opus_overlap" and r.get("facts") and r.get("fact_ladders")]
        for r in P:
            f = json.loads(r["facts"]); z0 = " ".join(s for s in [f"The text is about {f['topic']}." if f.get("topic") else "", f"Genre: {f['genre']}." if f.get("genre") else "",
                                                             f"At its end it is {f['doing']}." if f.get("doing") else ""] if s)
            facts = [x for x in json.loads(r["fact_ladders"]) if x.get("twin")][: a.max_facts]
            if not facts or not z0: continue
            for fi, x in enumerate(facts):
                texts = {}
                for k in ("exact", "partial", "category", "omit", "twin"):
                    s = ladder_sentence(x, k); t = (z0 + " " + s) if s else (z0 if k == "omit" else None)
                    if t: texts[k] = t
                if "exact" not in texts or "twin" not in texts: continue
                groups.append({"set": "ladder", "kind": x["type"], "act": ("val", r["row"]), "doc": VD[r["row"]], "texts": texts, "true": "exact", "value": x.get("value"), "twin": x.get("twin"), "fact_idx": fi})
    if a.limit: groups = groups[: a.limit]
    return groups, {"val": VA, "clean1": CA, "mean": VA.mean(0, keepdim=True)}, VD, prompt_msgs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True); p.add_argument("--adapter", default="/vol_q36/ckpts/qwen36_av/iter_0007813"); p.add_argument("--base", default="Qwen/Qwen3.6-27B")
    p.add_argument("--val", default="/vol_q36/data/sft/av_sft_val.parquet"); p.add_argument("--clean1", default="/vol_q36/data/sft/av_sft_val_clean1.parquet")
    p.add_argument("--classify-json", default="/vol_glp/cond/halluc_classify_numbers_sw_tokar.json")
    p.add_argument("--twins-json", default="/vol_glp/cond/flow_noise/twins.json"); p.add_argument("--deletions-json", default="/vol_glp/cond/flow_noise/deletions.json")
    p.add_argument("--ladder-parquet", default="/vol_glp/scale/g2pilot/g2_pilot_v3.parquet"); p.add_argument("--max-facts", type=int, default=3)
    p.add_argument("--sets", default="wrong_detail,numbers,twins,deletions,ladder"); p.add_argument("--k-null", type=int, default=4); p.add_argument("--bs", type=int, default=16)
    p.add_argument("--max-resp-tokens", type=int, default=400); p.add_argument("--limit", type=int, default=0); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(); dev = "cuda:0"; t0 = time.time()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    groups, ACT, VD, prompt_msgs = build_groups(a)
    from collections import Counter
    print(f"[av-lik] {len(groups)} groups: {Counter((g['set'], g['kind']) for g in groups).most_common()}", flush=True)

    # ---- null activations per group: K activations from OTHER documents of av_sft_val (seeded) + the mean activation
    rng = random.Random(a.seed); n_val = ACT["val"].shape[0]
    for gi, g in enumerate(groups):
        nulls = []
        while len(nulls) < a.k_null:
            j = rng.randrange(n_val)
            if VD[j] != g["doc"]: nulls.append(j)
        g["null_rows"] = nulls
    conds = ["h"] + [f"null{k}" for k in range(a.k_null)] + ["mean"]

    # ---- model: raw base + the SFT verbalizer LoRA, activation injected at the marker (ADD-norm-matched, output of decoder layer 1)
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from nla.config import load_nla_config
    from nla.utils.hooks import register_karvonen_hook
    from nla.utils.prompts import build_prompt_text
    snap = snapshot_download(a.base, token=os.environ.get("HF_TOKEN"), local_dir="/root/base_snap", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.py", "*.model", "*.tiktoken"])
    tok = AutoTokenizer.from_pretrained(snap)
    lm = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval(); lm.requires_grad_(False)
    peft = PeftModel.from_pretrained(lm, a.adapter, adapter_name="av"); peft.eval(); peft.set_adapter("av")
    cfg = load_nla_config(a.adapter, tok); vref = [None]
    register_karvonen_hook(peft, vref, cfg.injection_token_id, cfg.injection_left_neighbor_id, cfg.injection_right_neighbor_id, layer_idx=1)
    prompt_ids = tok.encode(build_prompt_text(prompt_msgs, cfg.injection_char, tok), add_special_tokens=False); P = len(prompt_ids)
    assert prompt_ids.count(cfg.injection_token_id) == 1, "prompt must contain exactly one injection marker"
    eos = tok.eos_token or ""; pad_id = tok.eos_token_id
    print(f"[av-lik] model ready ({time.time() - t0:.0f}s); prompt {P} tokens; eos {eos!r}", flush=True)

    # ---- flat job list: (group, text name, condition) -> response ids; scored in length-sorted micro-batches
    resp_cache = {}
    def resp_ids(text):
        if text not in resp_cache: resp_cache[text] = tok.encode(wrap_explanation(text) + eos, add_special_tokens=False)[: a.max_resp_tokens]
        return resp_cache[text]
    def act_of(g, cond):
        if cond == "h": src, r = g["act"]; return ACT[src][r]
        if cond == "mean": return ACT["mean"][0]
        return ACT["val"][g["null_rows"][int(cond[4:])]]
    jobs = [(gi, name, cond) for gi, g in enumerate(groups) for name in g["texts"] for cond in conds]
    jobs.sort(key=lambda j: -len(resp_ids(groups[j[0]]["texts"][j[1]])))
    print(f"[av-lik] {len(jobs)} sequences to score (bs {a.bs})", flush=True)
    lp = {}   # (gi, name, cond) -> summed log p of the response tokens
    done = 0; t1 = time.time()
    for s in range(0, len(jobs), a.bs):
        ch = jobs[s: s + a.bs]; ids_l = [prompt_ids + resp_ids(groups[gi]["texts"][nm]) for gi, nm, _ in ch]
        T = max(len(x) for x in ids_l); B = len(ch)
        bx = torch.full((B, T), pad_id, dtype=torch.long); am = torch.zeros((B, T), dtype=torch.long)
        for r, x in enumerate(ids_l): bx[r, : len(x)] = torch.tensor(x); am[r, : len(x)] = 1
        bx = bx.to(dev); am = am.to(dev); vref[0] = torch.stack([act_of(groups[gi], cond) for gi, _, cond in ch]).to(dev)
        try:
            with torch.no_grad(): logits = peft(input_ids=bx, attention_mask=am, use_cache=False).logits
        finally: vref[0] = None
        for r, (gi, nm, cond) in enumerate(ch):
            L = len(ids_l[r]); tgt = bx[r, P: L]; lg = logits[r, P - 1: L - 1].float(); lsm = torch.log_softmax(lg, -1)
            lp[(gi, nm, cond)] = float(lsm.gather(-1, tgt[:, None]).sum())
        del logits; done += B
        if (s // a.bs) % 50 == 0: print(f"[av-lik] {done}/{len(jobs)} ({(time.time() - t1) / max(done, 1) * 1000:.0f} ms/seq, T={T})", flush=True)
    print(f"[av-lik] scoring done in {(time.time() - t1) / 60:.1f} min", flush=True)

    # ---- per-group records + summary
    recs = []
    for gi, g in enumerate(groups):
        rec = {k: v for k, v in g.items() if k not in ("texts",)}; rec["act"] = list(g["act"]); rec["n_tokens"] = {nm: len(resp_ids(t)) for nm, t in g["texts"].items()}
        rec["logp"] = {nm: {cond: lp[(gi, nm, cond)] for cond in conds} for nm in g["texts"]}
        recs.append(rec)
    def pair_E(rec, a_, b_):
        """h-specific evidence that text a_ fits h better than text b_: Δ(h) − mean over random-doc nulls of Δ(null); also the mean-activation control."""
        L = rec["logp"]; d_h = L[a_]["h"] - L[b_]["h"]; d_null = [L[a_][f"null{k}"] - L[b_][f"null{k}"] for k in range(a.k_null)]; d_mean = L[a_]["mean"] - L[b_]["mean"]
        return {"d_h": d_h, "d_null_mean": float(np.mean(d_null)), "d_null_sd": float(np.std(d_null)), "d_meanact": d_mean, "E": d_h - float(np.mean(d_null)), "E_meanact": d_h - d_mean}
    def summarize(pairs):
        """pairs: list of pair_E dicts -> accuracies + effect sizes with CIs (Wilson for accuracies, bootstrap for mean E)."""
        n = len(pairs)
        if n == 0: return {"n": 0}
        E = np.array([q["E"] for q in pairs]); Em = np.array([q["E_meanact"] for q in pairs]); dh = np.array([q["d_h"] for q in pairs]); dn = np.array([q["d_null_mean"] for q in pairs])
        rs = np.random.default_rng(0); boots = np.array([E[rs.integers(0, n, n)].mean() for _ in range(2000)])
        out = {"n": n, "acc_E": float((E > 0).mean()), "acc_E_ci": wilson(int((E > 0).sum()), n), "acc_E_meanact": float((Em > 0).mean()), "acc_E_meanact_ci": wilson(int((Em > 0).sum()), n),
               "acc_raw_h": float((dh > 0).mean()), "acc_raw_h_ci": wilson(int((dh > 0).sum()), n), "acc_text_prior": float((dn > 0).mean()), "acc_text_prior_ci": wilson(int((dn > 0).sum()), n),
               "mean_E_nats": float(E.mean()), "mean_E_ci": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))), "sd_E": float(E.std()), "effect_size_d": float(E.mean() / (E.std() + 1e-9)),
               "mean_d_h": float(dh.mean()), "mean_d_null": float(dn.mean()), "median_E": float(np.median(E))}
        return out
    summary = {}
    by = {}
    for rec in recs:
        if rec["set"] == "wrong_detail": by.setdefault(("wrong_detail", rec["kind"]), []).append(pair_E(rec, "true", "alt")); by.setdefault(("wrong_detail", "all"), []).append(pair_E(rec, "true", "alt"))
        elif rec["set"] == "numbers":
            for m in ("near", "far", "hedge", "removed"):
                if m in rec["logp"]: by.setdefault(("numbers", m), []).append(pair_E(rec, "true", m))
            if "hedge" in rec["logp"] and "near" in rec["logp"]: by.setdefault(("numbers", "hedge_vs_near"), []).append(pair_E(rec, "hedge", "near"))
            if "removed" in rec["logp"] and "near" in rec["logp"]: by.setdefault(("numbers", "removed_vs_near"), []).append(pair_E(rec, "removed", "near"))
        elif rec["set"] == "twins":
            for nm in rec["logp"]:
                if nm != "true": by.setdefault(("twins", nm.split(":")[1]), []).append(pair_E(rec, "true", nm)); by.setdefault(("twins", "all"), []).append(pair_E(rec, "true", nm))
        elif rec["set"] == "deletions":
            by.setdefault(("deletions", "true_vs_remove_false"), []).append(pair_E(rec, "true", "remove_false")); by.setdefault(("deletions", "true_vs_remove_true"), []).append(pair_E(rec, "true", "remove_true"))
            by.setdefault(("deletions", "remove_false_vs_remove_true"), []).append(pair_E(rec, "remove_false", "remove_true"))
        elif rec["set"] == "ladder":
            for a_, b_ in LADDER_PAIRS:
                if a_ in rec["logp"] and b_ in rec["logp"]:
                    q = pair_E(rec, a_, b_); by.setdefault(("ladder", f"{a_}>{b_}", rec["kind"]), []).append(q); by.setdefault(("ladder", f"{a_}>{b_}", "all"), []).append(q)
    for key, pairs in by.items(): summary["/".join(key)] = summarize(pairs)
    for k, v in sorted(summary.items()):
        if v.get("n"): print(f"[av-lik] {k:48s} n={v['n']:5d}  P(E>0)={v['acc_E']:.3f} [{v['acc_E_ci'][0]:.3f},{v['acc_E_ci'][1]:.3f}]  raw={v['acc_raw_h']:.3f}  prior={v['acc_text_prior']:.3f}  meanE={v['mean_E_nats']:+.2f} nats  d={v['effect_size_d']:+.2f}", flush=True)
    json.dump({"adapter": a.adapter, "base": a.base, "k_null": a.k_null, "conds": conds, "prompt_tokens": P, "n_groups": len(groups), "summary": summary, "records": recs,
               "note": "logp = summed log p_AV(response tokens | prompt, activation); response = <explanation>\\n z \\n</explanation> + eos; E = Δ(h) − mean_k Δ(null_k)."},
              open(a.out, "w"))
    print(f"[av-lik] wrote {a.out} ({time.time() - t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
