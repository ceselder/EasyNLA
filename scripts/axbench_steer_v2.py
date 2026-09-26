"""AxBench v2 steering generations (Qwen3.6-27B, layer 42): the same seeded plan and protocol as scripts/axbench_steer.py (40 Concept500
concepts x 5 Alpaca-Eval instructions, 96 new tokens, T = 1), with explanation-REWRITE edits. Every activation edit is applied at the last
prompt position and at every generated position, and is DIRECTION-ONLY: the edited vector is rescaled to ||h||.

Text pairs per (concept, instruction) (scripts/axbench_v2_prep.py + scripts/axbench_v2_rewrite.py):
  rw    (z, z_c)       z = warm verbalizer's explanation of the last-prompt-position activation, z_c = Claude Opus 5's careful rewrite of z
                       into the explanation the verbalizer would write if the model were also thinking about the concept   [the method]
  add   (z, z_add)     z + two bolted-on claim sentences ("The text is about C. The next words will mention C.")          [templated claims]
  tmpl  (N, T)         the fixed neutral / concept templates of the 7d run                                                 [comparability]
Families (method name, factor = strength beta):
  none, prompt (the concept appended to the instruction), random_b
  diffmean       h + b||h|| unit(DiffMean_c)            (prep: concept passages minus other-concept passages, AxBench's strongest simple method)
  ar_<pair>      h + b||h|| unit(AR(z1) - AR(z0))        (MSE reconstructor direction, fixed per instruction)
  <critic>_<pair>_t<t>   delta at the CURRENT activation: h + b||h|| unit(x0hat(h,t,z1) - x0hat(h,t,z0)), x0hat(x,t,c) = x - t v(x,t,c)
usage: python scripts/axbench_steer_v2.py --family refs --tag A      |   --family flow --critic g1ann=/vol_glp/cond/.../adapter_latest.pt --pairs rw,add,tmpl --tag B
   -> /vol_glp/axbench/v2_<tag>.json  (records for scripts/axbench_judge_batch.py)"""
import argparse, json, os, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import intervene_playground_app as pg
import rhyme_plan_steer as R
from axbench_v2_prep import chat
from nla.flow.claims import split_claims, format_claims
OUT = "/vol_glp/axbench"
COND_TMPL = "The text is about {c}. The model expects the continuation to explicitly discuss {c}."
CLAIMS_C = ["The text is about {c}.", "The next words will mention {c}.", "The answer will bring up {c}."]
CLAIMS_NEU = ["The text is about the topic of the instruction.", "The next words will continue answering the instruction.", "The answer will stay on the instruction."]
NEUTRAL = "The text is a generic answer to an instruction; the model expects the continuation to keep answering it."
PROMPT_TMPL = "{instruction}\n\nIn your response, incorporate the following concept: {c}."
unit = lambda d: d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
renorm = lambda h, hn: hn / hn.norm(dim=-1, keepdim=True).clamp_min(1e-8) * h.norm(dim=-1, keepdim=True)


def rows_fn(f):
    """pg's hook passes [B, d] (decode) or [B, k, d] (prefill positions); edits are per row (one row per instruction)."""
    def g(h):
        sh = h.shape; return f(h.reshape(sh[0], -1, sh[-1])[:, -1]).reshape(sh[0], 1, sh[-1]).expand(sh) if len(sh) == 3 else f(h)
    return g


def load_flow(path):
    from nla.flow.scoring import FlowBundle
    aa = torch.load(path, map_location="cpu")["args"]; pco = os.path.join(os.path.dirname(path), "prior_cotrained_latest.pt")
    fb = FlowBundle(aa["prior"], path, aa["stats"], pg.DEV1, base=pg.S["snap"], enc_layer=aa.get("enc_layer", 42), ar_ckpt=aa.get("ar_ckpt", "/vol/ckpts/qwen36_27b/ar_sft_merged"),
                    prior_override=pco if os.path.exists(pco) else None)
    assert not fb.aa.get("resid_shift"); fb.model.eval(); return fb


def delta_fn(fb, c0, c1, t, beta):
    """h -> renorm(h + beta |h| unit(x0hat(h,t,c1) - x0hat(h,t,c0))), conditions batched per row."""
    @torch.no_grad()
    def f(h):
        hd = h.to(pg.DEV1).float(); x = fb.norm.normalize(hd).float(); tt = torch.full((x.shape[0],), float(t), device=x.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v1 = fb.model(x, tt, *c1).float(); v0 = fb.model(x, tt, *c0).float()
        d = fb.norm.denormalize(x - t * v1).float() - fb.norm.denormalize(x - t * v0).float()
        return renorm(hd, hd + beta * hd.norm(dim=-1, keepdim=True) * unit(d)).to(h.device)
    return f


def dir_fn(d, beta):
    def f(h):
        dd = d.to(h.device, h.dtype); dd = dd if dd.dim() == 2 else dd[None].expand_as(h)
        return renorm(h, h + beta * h.norm(dim=-1, keepdim=True) * unit(dd))
    return f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True, choices=["refs", "flow"]); p.add_argument("--critic", default="", help="name=adapter_path (family flow)")
    p.add_argument("--pairs", default="rw,add,tmpl"); p.add_argument("--ts", default="0.1,0.3"); p.add_argument("--betas", default="0.25,0.5,1")
    p.add_argument("--prep", default=f"{OUT}/v2_prep"); p.add_argument("--rewrites", default=f"{OUT}/v2_rewrites.json"); p.add_argument("--tag", required=True)
    p.add_argument("--cmean", action="store_true", help="family flow: also the h-independent conditional-mean direction per pair"); p.add_argument("--skip-refs", action="store_true", help="family refs: only the AR directions"); p.add_argument("--skip-base", action="store_true", help="refs family: skip none/prompt (strength fill-in runs)"); p.add_argument("--start", type=int, default=0); p.add_argument("--end", type=int, default=10 ** 9); p.add_argument("--max-new-tokens", type=int, default=96)
    a = p.parse_args(); betas = [float(x) for x in a.betas.split(",")]; ts = [float(x) for x in a.ts.split(",")]; pairs = [x for x in a.pairs.split(",") if x]
    if torch.cuda.device_count() == 1: pg.DEV1 = pg.DEV0                                          # one B200 holds the LM (54 GB) + a flow critic (~80 GB)
    t00 = time.time(); pg._load(); tok, lm, st = pg.S["tok"], pg.S["lm"], pg.S["st"]; tok.padding_side = "left"
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    prep = json.load(open(a.prep + ".json")); P = torch.load(a.prep + ".pt", map_location="cpu"); rw = json.load(open(a.rewrites))["items"]
    plan = prep["plan"]; outp = f"{OUT}/v2_{a.tag}.json"
    out = json.load(open(outp)) if os.path.exists(outp) else {"args": vars(a), "cond_template": COND_TMPL, "neutral": NEUTRAL, "prompt_template": PROMPT_TMPL, "records": []}
    have = {r["concept_id"] for r in out["records"]}
    def texts(k, pair):
        pl = plan[k]; n = len(pl["instructions"]); c = pl["concept"]
        if pair == "tmpl": return [NEUTRAL] * n, [COND_TMPL.format(c=c)] * n
        if pair in ("c1", "c3"):                                               # bullet claims, exactly the '• c' training format of the claim critics
            m = 1 if pair == "c1" else 3
            return [format_claims(CLAIMS_NEU[:m])] * n, [format_claims([x.format(c=c) for x in CLAIMS_C[:m]])] * n
        z0 = [rw[f"{k}:{j}"]["z"] for j in range(n)]; z1 = [rw[f"{k}:{j}"]["z_c" if pair == "rw" else "z_add"] for j in range(n)]; return z0, z1
    fb = crit = None
    claim_critic = False
    if a.family == "flow":
        cname, _, cpath = a.critic.partition("="); fb = load_flow(cpath); claim_critic = int(fb.aa.get("claim_subsets") or 0) > 0
        print(f"[axb2] flow critic {cname}: cond_mode {fb.cond_mode}, claim critic {claim_critic} ({time.time() - t00:.0f}s)", flush=True)
    fmt = lambda z: z if (not claim_critic or z.lstrip().startswith("•")) else format_claims(split_claims(z) or [z])   # scripts/steer_delta.py FlowDen.fmt
    @torch.no_grad()
    def cmean_dir(z0, z1, K=8, seed=7):
        """h-independent conditional-mean difference E[h|z1] - E[h|z0]: one-step x0 prediction from pure noise at t = 1, K shared draws, per text pair."""
        out = []
        for j, (x0t, x1t) in enumerate(zip(z0, z1)):
            E = torch.randn(K, fb.norm.normalize(torch.zeros(1, P["h"].shape[-1], device=pg.DEV1)).shape[-1], device=pg.DEV1, generator=torch.Generator(device=pg.DEV1).manual_seed(seed + j)); mu = []
            for tx in (x0t, x1t):
                enc, mk, cv = fb.cond([tx]); rep = lambda x: None if x is None else x.expand(K, *x.shape[1:])
                with torch.autocast("cuda", dtype=torch.bfloat16): v = fb.model(E, torch.ones(K, device=pg.DEV1), rep(enc), rep(mk), rep(cv)).float()
                mu.append(fb.norm.denormalize((E - v).mean(0, keepdim=True))[0].float())
            out.append((mu[1] - mu[0]).cpu())
        return torch.stack(out)
    if a.family == "refs": crit = pg._critic(R.AR_NAME)
    def generate(prompts, fn):
        enc = tok(prompts, return_tensors="pt", padding=True).to(pg.DEV0); T = enc["input_ids"].shape[1]
        st.update(cap=None, vec=None, prefill_fn=rows_fn(fn) if fn else None, prefill_idx=torch.tensor([T - 1], device=pg.DEV0), decode_fn=rows_fn(fn) if fn else None)
        try:
            with torch.no_grad():
                g = lm.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=a.max_new_tokens, pad_token_id=tok.pad_token_id)
        finally: st.update(prefill_fn=None, prefill_idx=None, decode_fn=None)
        cont = g[:, T:]; txt = tok.batch_decode(cont, skip_special_tokens=True)
        with torch.no_grad():
            lg = lm(input_ids=g, attention_mask=torch.cat([enc["attention_mask"], (cont != tok.pad_token_id).long()], 1)).logits[:, T - 1:-1].float()
            nll = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), cont.reshape(-1), reduction="none").view(cont.shape)
            m = (cont != tok.pad_token_id).float(); nll = ((nll * m).sum(1) / m.sum(1).clamp_min(1)).tolist()
        return txt, nll
    with pg.S["gpu"], pg._base():
        for k, pl in enumerate(plan):
            if k < a.start or k >= a.end or pl["concept_id"] in have: continue
            t0 = time.time(); c = pl["concept"]; instrs = pl["instructions"]; prompts = [chat(tok, i) for i in instrs]; S = []
            if a.family == "refs":
                if not a.skip_base: S += [("none", 0.0, prompts, None), ("prompt", 0.0, [chat(tok, PROMPT_TMPL.format(instruction=i, c=c)) for i in instrs], None)]
                if not a.skip_refs:
                    rnd = torch.randn(P["h"].shape[-1], generator=torch.Generator().manual_seed(777 + pl["concept_id"]))
                    S += [("random", b, prompts, dir_fn(rnd, b)) for b in betas]
                    S += [("diffmean", b, prompts, dir_fn(P["diffmean"][k], b)) for b in betas]
                for pair in pairs:
                    z0, z1 = texts(k, pair); d = torch.stack([pg.ar_pred(crit, y) - pg.ar_pred(crit, x) for x, y in zip(z0, z1)])
                    S += [(f"ar_{pair}", b, prompts, dir_fn(d, b)) for b in betas]
            else:
                for pair in pairs:
                    z0, z1 = texts(k, pair); z0, z1 = [fmt(x) for x in z0], [fmt(x) for x in z1]
                    with torch.no_grad(): c0, c1 = fb.cond(z0), fb.cond(z1)
                    for t in ts: S += [(f"{cname}_{pair}_t{t:g}", b, prompts, delta_fn(fb, c0, c1, t, b)) for b in betas]
                    if a.cmean: dcm = cmean_dir(z0, z1); S += [(f"{cname}_cmean_{pair}", b, prompts, dir_fn(dcm, b)) for b in betas]
            for name, factor, pr, fn in S:
                txt, nll = generate(pr, fn)
                for j, (ins, t_, n_) in enumerate(zip(instrs, txt, nll)):
                    out["records"].append(dict(concept_id=pl["concept_id"], concept=c, genre=pl.get("genre"), instr_id=j, instruction=ins, method=name, factor=factor, generation=t_, nll_unpatched=n_))
            json.dump(out, open(outp, "w"))
            ex = {nm: out["records"][-len(S) * len(instrs) + i * len(instrs)]["generation"][:70] for i, (nm, *_ ) in enumerate(S) if i % max(1, len(S) // 4) == 0}
            print(f"[axb2] {k + 1}/{len(plan)} concept {pl['concept_id']} '{c[:40]}' {len(S)} settings {time.time() - t0:.0f}s (total {(time.time() - t00) / 60:.1f} min) | {ex}", flush=True)
    print(f"[done] {len(out['records'])} records -> {outp} ({(time.time() - t00) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
