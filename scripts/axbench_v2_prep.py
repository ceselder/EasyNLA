"""AxBench v2 prep (Qwen3.6-27B, layer 42): per (concept, instruction) of the SAME seeded plan as scripts/axbench_steer.py, the layer-42
activation h at the last prompt position of the chat-formatted instruction and the warm verbalizer's greedy explanation z of it (the base text
that Claude rewrites into a concept-carrying explanation z_c, scripts/axbench_v2_rewrite.py); plus the DiffMean baseline (AxBench's strongest
simple method): per concept, the mean layer-42 activation over tokens of 8 short concept passages written by Qwen3.6-27B itself minus the mean
over 8 passages of 8 OTHER concepts (one each, seeded).
  -> /vol_glp/axbench/v2_prep.json (plan, z, passages) + /vol_glp/axbench/v2_prep.pt (h [C, I, d], diffmean [C, d])
usage: python scripts/axbench_v2_prep.py [--n-concepts 40 --n-instr 5]"""
import argparse, json, os, random, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import intervene_playground_app as pg
import rhyme_plan_steer as R
OUT = "/vol_glp/axbench"
PASSAGE_PROMPT = "Write a short passage of three or four sentences that is clearly about the following concept: {c}\nOutput only the passage."


def plan_of(a):
    concepts = json.load(open(a.concepts)); alp = [x["instruction"] for x in json.load(open(a.alpaca))]
    sel = random.Random(a.seed).sample(concepts, a.n_concepts)
    return [(c, [alp[i] for i in random.Random(a.seed * 1000 + c["concept_id"]).sample(range(len(alp)), a.n_instr)]) for c in sel]


def chat(tok, user):
    try: return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--concepts", default=f"{OUT}/concepts_9b_l20_positive.json"); p.add_argument("--alpaca", default=f"{OUT}/alpaca_eval.json")
    p.add_argument("--n-concepts", type=int, default=40); p.add_argument("--n-instr", type=int, default=5); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-pass", type=int, default=8); p.add_argument("--pass-tokens", type=int, default=96); p.add_argument("--tag", default="v2_prep")
    a = p.parse_args(); t0 = time.time(); os.makedirs(OUT, exist_ok=True)
    pg._load(); tok, lm, st = pg.S["tok"], pg.S["lm"], pg.S["st"]
    plan = plan_of(a); C, I = len(plan), a.n_instr
    # ---- h at the last prompt position of every chat-formatted instruction
    H = None
    for k, (c, instrs) in enumerate(plan):
        for j, ins in enumerate(instrs):
            _, _, h0, _ = pg.capture_h(chat(tok, ins))
            if H is None: H = torch.zeros(C, I, h0.shape[-1])
            H[k, j] = h0.cpu()
    print(f"[prep] {C * I} last-position activations ({time.time() - t0:.0f}s)", flush=True)
    # ---- warm verbalizer explanations (greedy)
    Z = pg._verbalize(H.view(C * I, -1), R.AV_WARM, temperature=0.0, max_new=200)
    print(f"[prep] {len(Z)} explanations ({time.time() - t0:.0f}s); e.g. {Z[0][:300]!r}", flush=True)
    # ---- DiffMean: concept passages from the model itself (adapters off, no steering hook)
    passages = {}
    with pg.S["gpu"], pg._base():
        st.update(cap=None, vec=None, decode_fn=None, prefill_fn=None)
        tok.padding_side = "left"
        for k, (c, _) in enumerate(plan):
            enc = tok([chat(tok, PASSAGE_PROMPT.format(c=c["output_concept"]))], return_tensors="pt").to(pg.DEV0)
            with torch.no_grad():
                g = lm.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=a.pass_tokens, num_return_sequences=a.n_pass,
                                pad_token_id=tok.pad_token_id or tok.eos_token_id)
            passages[c["concept_id"]] = [x.strip() for x in tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
    print(f"[prep] passages done ({time.time() - t0:.0f}s); e.g. {passages[plan[0][0]['concept_id']][0][:200]!r}", flush=True)
    pmean = {}
    for cid, ps in passages.items():
        pmean[cid] = torch.stack([pg.capture_all(x)[1].float().mean(0).cpu() for x in ps if x])            # [n_pass, d] per-passage token means
    cids = [c["concept_id"] for c, _ in plan]; DM = torch.zeros(C, H.shape[-1])
    for k, cid in enumerate(cids):
        others = [x for x in cids if x != cid]; rng = random.Random(1000 + cid); neg = rng.sample(others, min(a.n_pass, len(others)))
        negv = torch.stack([pmean[o][rng.randrange(pmean[o].shape[0])] for o in neg])
        DM[k] = pmean[cid].mean(0) - negv.mean(0)
    torch.save({"h": H, "diffmean": DM, "concept_ids": cids}, f"{OUT}/{a.tag}.pt")
    json.dump({"args": vars(a), "plan": [dict(concept_id=c["concept_id"], concept=c["output_concept"], genre=c.get("concept_genre"), instructions=instrs, z=Z[k * I:(k + 1) * I])
                                         for k, (c, instrs) in enumerate(plan)], "passages": {str(k_): v for k_, v in passages.items()}, "passage_prompt": PASSAGE_PROMPT},
              open(f"{OUT}/{a.tag}.json", "w"), indent=1)
    print(f"[prep] done {time.time() - t0:.0f}s -> {OUT}/{a.tag}.json, .pt | |diffmean|/|h| median {float((DM.norm(dim=-1) / H.norm(dim=-1).mean(1)).median()):.3f}", flush=True)


if __name__ == "__main__":
    main()
