"""Hedge-ladder ordering of a flow critic on HELD-OUT positions: does the reward prefer a true detail over a hedge over nothing over a wrong detail?

Items: the g2 pilot's Opus-overlap positions (av_sft_val rows, never trained on) with their validated fact sheets and programmatic ladders
(nla/datagen/g2_spec.py): base z0 = topic + genre + what the text is doing; one fact appended at each rung
  exact    "It mentions a person: Mark Hill (hair stylist to the stars)."
  partial  "It mentions a person: someone surnamed Hill."
  category "It mentions a person: a person (hair stylist to the stars)."
  omit     z0 alone
  twin     the same sentence with a WRONG exact value (same fact subtype, another document; same description)
Reward = -(flow-matching loss) on the RL t grid (0.1..0.9) with K noise draws SHARED across the rungs of an item (the trainer's scheme).
Reports pairwise P(reward_a > reward_b) for exact>twin (wrong-detail detection on a single fact), category>twin (a hedge containing the truth
beats a confident wrong value), omit>twin (saying nothing beats a wrong value), exact>omit, exact>partial, partial>category, by fact type.
usage: python scripts/hedge_ladder_eval.py --critic <tag or tag/snap_N> --out <json>"""
import argparse, json, os, sys, time
import numpy as np, torch, pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
TS = [0.1, 0.3, 0.5, 0.7, 0.9]
KIND = {"person": "a person", "organisation": "an organisation", "place": "a place", "other_entity": "a named item", "number": "a number", "date": "a date", "quote": "a phrase"}
PAIRS = [("exact", "twin"), ("category", "twin"), ("omit", "twin"), ("exact", "omit"), ("exact", "partial"), ("partial", "category"), ("category", "omit")]


def sentence(x, rung):
    if rung == "omit": return None
    if rung == "twin": val = x.get("twin")
    else: val = x["value"] if rung == "exact" else (x["ladder"] or {}).get(rung)
    if not val: return None
    if x["type"] == "quote": return f"It contains “{val}”." if rung in ("exact", "twin") else f"It contains {val}."
    desc = f" ({x['desc']})" if x.get("desc") and rung in ("exact", "twin") else ""
    return f"It mentions {KIND[x['type']]}: {val}{desc}."


def main():
    p = argparse.ArgumentParser(); p.add_argument("--critic", required=True); p.add_argument("--out", required=True); p.add_argument("--K", type=int, default=4)
    p.add_argument("--items", default="/vol_glp/scale/g2pilot/g2_pilot_v3.parquet"); p.add_argument("--max-facts", type=int, default=3)
    a = p.parse_args()
    from nla.flow.scoring import FlowBundle
    P = pq.read_table(a.items).to_pylist(); P = [r for r in P if r["src"] == "opus_overlap" and r["facts"] and r["fact_ladders"]]
    V = pq.read_table("/vol_q36/data/sft/av_sft_val.parquet", columns=["activation_vector"])
    ap = f"/vol_glp/cond/{a.critic}/adapter_latest.pt"; aa = torch.load(ap, map_location="cpu")["args"]; pco = os.path.join(os.path.dirname(ap), "prior_cotrained_latest.pt")
    fb = FlowBundle(aa["prior"], ap, aa["stats"], "cuda", base="Qwen/Qwen3.6-27B", enc_layer=aa.get("enc_layer", 42), ar_ckpt=aa.get("ar_ckpt", "/vol/ckpts/qwen36_27b/ar_sft_merged"),
                    prior_override=pco if os.path.exists(pco) else None)
    fb.model.eval(); tt = torch.tensor(TS, device="cuda"); T = len(TS); res = []; t0 = time.time()
    for n, r in enumerate(P):
        f = json.loads(r["facts"]); z0 = " ".join(s for s in [f"The text is about {f['topic']}." if f.get("topic") else "", f"Genre: {f['genre']}." if f.get("genre") else "",
                                                        f"At its end it is {f['doing']}." if f.get("doing") else ""] if s)
        facts = [x for x in json.loads(r["fact_ladders"]) if x.get("twin")][: a.max_facts]
        if not facts or not z0: continue
        h = torch.tensor(np.asarray(V.column(0)[r["row"]].as_py(), dtype=np.float32))[None].cuda(); x0 = fb.norm.normalize(h).float(); d = x0.shape[1]
        g = torch.Generator(device="cuda").manual_seed(r["row"]); eps = torch.randn(a.K, d, device="cuda", generator=g)
        for x in facts:
            rungs = {k: (z0 + " " + sentence(x, k)) if sentence(x, k) else (z0 if k == "omit" else None) for k in ("exact", "partial", "category", "omit", "twin")}
            names = [k for k, v in rungs.items() if v]; texts = [rungs[k] for k in names]; G = len(texts)
            with torch.no_grad():
                enc, mk, cv = fb.cond(texts)
                xt = ((1 - tt)[None, :, None] * x0[None] + tt[None, :, None] * eps[:, None, :]).reshape(a.K * T, d); tv = tt.repeat(a.K)
                tgt = (eps[:, None, :] - x0[None]).expand(a.K, T, d).reshape(a.K * T, d); R_ = a.K * T
                xB, tB, tgtB = xt.repeat(G, 1), tv.repeat(G), tgt.repeat(G, 1)
                encB = enc.repeat_interleave(R_, 0) if enc is not None else None; mkB = mk.repeat_interleave(R_, 0) if mk is not None else None
                cvB = cv.repeat_interleave(R_, 0) if cv is not None else None
                with torch.autocast("cuda", dtype=torch.bfloat16): v = fb.model(xB, tB, encB, mkB, cvB).float()
                L = fb.fm_err(v, tgtB).view(G, R_).mean(1).cpu().numpy()
            res.append({"row": r["row"], "type": x["type"], "rewards": {k: float(-l) for k, l in zip(names, L)}})
        if n % 50 == 0: print(f"[ladder {a.critic}] {n}/{len(P)} positions, {len(res)} items, {time.time() - t0:.0f}s", flush=True)
    summ = {}
    for grp in ["all"] + sorted(set(it["type"] for it in res)):
        its = [it for it in res if grp == "all" or it["type"] == grp]; summ[grp] = {"n": len(its)}
        for hi, lo in PAIRS:
            w = [it["rewards"][hi] > it["rewards"][lo] for it in its if hi in it["rewards"] and lo in it["rewards"]]
            if w: summ[grp][f"P({hi}>{lo})"] = float(np.mean(w)); summ[grp][f"n({hi}>{lo})"] = len(w)
    json.dump({"critic": a.critic, "ts": TS, "K": a.K, "summary": summ, "items": res}, open(a.out, "w"), indent=1)
    print("[ladder] SUMMARY", json.dumps(summ["all"]), flush=True)


if __name__ == "__main__":
    main()
