"""Make the model think about a different animal: a concept-swap steering demo where the concept IS readable in J-space (Qwen3.6-27B, layer 42).

For ~24 short prompts whose last token (the ANCHOR) has the model clearly thinking about one animal (18 name it, 6 only imply it):
 (1) J-space monitor at the anchor: J-lens rank of the source animal, the target, and a panel of 24 animals, at layers 24-60; the anchor's
     layer-42 activation verbalized by two verbalizers (does the explanation name the animal?);
 (2) steering at the anchor ("anchor" scope) and at the anchor plus every generated position ("anchor_on" scope) toward a target animal of a
     different category: J-lens coordinate swap / J-lens direction (layer 42, and the swap at layer 54 as a reference), MSE-reconstructor
     direction with z' = the verbalizer's explanation with the animal swapped (templated sentence appended if the explanation does not name
     it), flow inversion at tau in {0.1,0.3,0.5,0.7,0.9} (source condition z or the unconditional prior; both flow critics), SDEdit at the
     same tau, the inversion displacement as a direction, random controls;
 (3) measured: J-lens rank of target vs source on the edited anchor activation, verbalizer read-back of the edited activation (greedy, warm
     start), whether the 40-token continuation mentions the target / source animal (string match; Sonnet judge offline), base-model NLL of
     the continuation, KL at the first generated token, edit norm.
Reuses scripts/rhyme_plan_steer.py (J-lens loader, generation with steering hooks, flow ODE). Outputs /vol_glp/cond/animal/{jspace,steer}.json."""
import argparse, json, os, re, sys, time
import numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import rhyme_plan_steer as R
pg = R.pg; DEV0, DEV1 = R.DEV0, R.DEV1; OUT = "/vol_glp/cond/animal"
PROMPTS = [  # (text ending at the anchor, source animal, target animal of a different kind, implied?) -- every animal name is ONE token for Qwen3.6
    ("My dog loves chasing the ball in the park. Every morning, my", "dog", "shark", False),
    ("The zookeeper fed the elephant a huge pile of", "elephant", "butterfly", False),
    ("At the aquarium, the dolphin leapt out of the water and the", "dolphin", "eagle", False),
    ("The cat curled up on the windowsill and", "cat", "frog", False),
    ("We watched the horse gallop across the field, its", "horse", "spider", False),
    ("The farmer milked the cow before sunrise, and the", "cow", "dolphin", False),
    ("In the savanna, the lion stalked its prey through the tall", "lion", "owl", False),
    ("The rabbit hopped into the garden and nibbled on the", "rabbit", "shark", False),
    ("Deep in the ocean, the whale sang a long, low", "whale", "monkey", False),
    ("The snake slithered across the warm rock and", "snake", "elephant", False),
    ("At the pond, a frog jumped onto a lily pad and", "frog", "bear", False),
    ("The owl perched on the branch and turned its head to", "owl", "pig", False),
    ("The little mouse crept out of its hole and sniffed the", "mouse", "wolf", False),
    ("The shark circled the boat, its fin cutting through the", "shark", "sheep", False),
    ("A bee buzzed from flower to flower, collecting", "bee", "tiger", False),
    ("The duck waddled across the muddy bank toward the", "duck", "camel", False),
    ("The monkey swung from vine to vine, chattering at the", "monkey", "turtle", False),
    ("The fox crept into the henhouse at midnight and grabbed a", "fox", "whale", False),
    ("It barked at the mailman, wagged its tail, and then the", "dog", "eagle", True),
    ("With its long trunk, it sprayed water over its back, and the", "elephant", "mouse", True),
    ("It purred loudly and rubbed against my legs, so I gave the", "cat", "snake", True),
    ("It spun a web in the corner of the room and waited for a", "spider", "dog", True),
    ("It slithered through the tall grass, flicking its forked tongue, and the", "snake", "rabbit", True),
    ("The hive was busy all summer; thousands of them made honey, and the", "bee", "horse", True),
]
POOL = ["penguin", "shark", "horse", "dolphin", "owl", "bear", "pig", "wolf", "sheep", "tiger", "turtle", "eagle", "camel", "fish", "spider", "dog", "cat", "whale", "snake", "frog", "lion", "monkey", "duck", "fox"]
PANEL = ["dog", "cat", "horse", "cow", "pig", "sheep", "bear", "wolf", "lion", "tiger", "shark", "whale", "dolphin", "eagle", "owl", "snake", "frog", "fish", "bird", "monkey", "mouse", "rabbit", "bee", "spider"]
PLURAL = {"mouse": ["mice"], "wolf": ["wolves"], "sheep": ["sheep"], "fish": ["fish", "fishes"], "octopus": ["octopuses", "octopi"], "goose": ["geese"]}
CRIT = R.CRIT; TAUS = [0.1, 0.3, 0.5, 0.7, 0.9]
RB_CONDS = ["none", "rand_b1", "jswap_a8", "jadd_b0.5", "jadd_b1", "jadd_b2", "ar_a1", "ar_a2", "invdir_trunk_a1", "inv_trunk_z_t0.5", "inv_trunk_z_t0.9", "inv_trunk_u_t0.5", "inv_trunk_u_t0.9",
            "sde_trunk_t0.5", "sde_trunk_t0.9", "inv_sw_z_t0.9", "sde_sw_t0.9"]


def ment(text, w):
    forms = {w, w + "s", w + "es"} | set(PLURAL.get(w, [])); return any(re.search(rf"\b{re.escape(f)}\b", text or "", re.I) for f in forms)
def cont_of(tok, s, T):
    t = tok.decode(s[T:].tolist(), skip_special_tokens=True); return t.split("<think>")[0].strip()


@torch.no_grad()
def nll_cont(ids, seqs, T):
    """mean NLL per token of the generated continuation (until EOS / '<think>') under the UNSTEERED base model."""
    tok, lm, st = pg.S["tok"], pg.S["lm"], pg.S["st"]; pad = tok.pad_token_id or tok.eos_token_id; eos = tok.eos_token_id
    L = max(int(s_.shape[0]) for s_ in seqs); X = torch.full((len(seqs), L), pad, dtype=torch.long)
    for i, s_ in enumerate(seqs): X[i, : s_.shape[0]] = s_
    cuts = []
    for i in range(len(seqs)):
        g_ = X[i, T:].tolist(); cut = len(g_)
        for j, t_ in enumerate(g_):
            if t_ == eos or "<think>" in tok.decode([t_]): cut = j; break
        cuts.append(cut)
    out = [None] * len(seqs)
    for c0 in range(0, len(seqs), 32):
        xb = X[c0:c0 + 32].to(DEV0)
        with pg.S["gpu"], pg._base():
            st.update(cap=None, vec=None, decode_fn=None, prefill_fn=None); lg = lm(input_ids=xb).logits[:, T - 1: L - 1].float()
        ce = F.cross_entropy(lg.transpose(1, 2), xb[:, T:], reduction="none")
        for i in range(xb.shape[0]):
            c = cuts[c0 + i]; out[c0 + i] = float(ce[i, :c].mean()) if c > 0 else None
    return out


def run_batch(ids, names, fnmap, layer, scope, K, greedy, n_new):
    """one batch, K rows per condition; 'anchor' = edit the anchor only; 'anchor_on' = the anchor and every generated position."""
    T = ids.shape[1]; sl = {nm: slice(i * K, (i + 1) * K) for i, nm in enumerate(names)}
    def comp(hb):
        o = hb.clone()
        for nm, s_ in sl.items(): o[s_] = fnmap[nm](hb[s_].float()).to(hb.dtype)
        return o
    idx = torch.tensor([T - 1], device=DEV0); dec = comp if scope == "anchor_on" else None
    if layer == 42: g = R.gen(ids, len(names) * K, greedy, n_new=n_new, prefill_fn=comp, prefill_idx=idx, decode_fn=dec, scores=greedy)
    else:
        R.ST54.update(prefill_fn=comp, idx=idx, decode_fn=dec)
        try: g = R.gen(ids, len(names) * K, greedy, n_new=n_new, scores=greedy)
        finally: R.ST54.update(prefill_fn=None, idx=None, decode_fn=None)
    return g, sl


def stage_a(a, lens):
    """J-space monitor at the anchor (source / target / panel ranks, layers 24-60) + verbalizer explanations of the anchor's layer-42 activation."""
    tok, lm, st = pg.S["tok"], pg.S["lm"], pg.S["st"]; items = []; t0 = time.time()
    for n, (text, src, tgt, implied) in enumerate(PROMPTS[: a.n]):
        if not R.single(tok, tgt): tgt = next(w for w in POOL if w != src and R.single(tok, w))
        ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEV0); T = ids.shape[1]
        with pg.S["gpu"], pg._base():
            st.update(cap=None, vec=None, decode_fn=None, prefill_fn=None)
            with torch.no_grad(): hs = lm(input_ids=ids, output_hidden_states=True).hidden_states
        ts_, tt_ = R.tid(tok, src), R.tid(tok, tgt); panel_ids = {w: R.tid(tok, w) for w in PANEL}; mon = {}
        for l in R.LAYERS_MON:
            lg = lens.logits(hs[l + 1][0, -1:], l)[0]
            mon[str(l)] = dict(src_rank=int(R.Lens.rank(lg, ts_)), tgt_rank=int(R.Lens.rank(lg, tt_)), panel={w: int(R.Lens.rank(lg, i)) for w, i in panel_ids.items()},
                               top10=[tok.decode([int(x)]) for x in lg.topk(10).indices])
        items.append(dict(n=n, text=text, src=src, tgt=tgt, implied=implied, src_single=R.single(tok, src), T=T, monitor=mon, _h=hs[43][0, -1].float().cpu(), _h54=hs[55][0, -1].float().cpu(), _ids=ids.cpu()))
        del hs; print(f"[A] {n}: '{src}'->'{tgt}' implied={implied} | J-lens rank of '{src}' at the anchor: " + " ".join(f"L{l} {mon[str(l)]['src_rank']}" for l in R.LAYERS_MON) + f" | top5@42 {mon['42']['top10'][:5]}", flush=True)
    H = torch.stack([it["_h"] for it in items])
    for key, av in (("warm", R.AV_WARM), ("nokl100", R.AV_NOKL)):
        vg = pg._verbalize(H, av, temperature=0.0, max_new=200); vs = pg._verbalize(H.repeat_interleave(4, 0), av, temperature=1.0, max_new=200)
        for i, it in enumerate(items):
            ex = [vg[i]] + vs[4 * i: 4 * i + 4]; it.setdefault("verbalizer", {})[key] = dict(greedy=ex[0], samples=ex[1:], names_src=[ment(e, it["src"]) for e in ex])
    for it in items:
        src, tgt, z0 = it["src"], it["tgt"], it["verbalizer"]["warm"]["greedy"]
        if ment(z0, src): z = z0; templ = False
        else: z = z0.rstrip() + f" The text is about a {src}; the continuation will keep talking about the {src}."; templ = True
        ze = z
        for f in sorted({src + "es", src + "s", src} | set(PLURAL.get(src, [])), key=len, reverse=True): ze = re.sub(rf"\b{re.escape(f)}\b", tgt + ("s" if f != src else ""), ze, flags=re.I)
        it.update(z=z, ze=ze, z_templated=templ)
        print(f"[A] {it['n']} verbalizer names '{src}': warm {it['verbalizer']['warm']['names_src']} nokl {it['verbalizer']['nokl100']['names_src']} | z' = {ze[:160]!r}", flush=True)
    json.dump([{k: v for k, v in it.items() if not k.startswith("_")} for it in items], open(f"{OUT}/jspace{a.tag}.json", "w")); print(f"[A] done {time.time() - t0:.0f}s", flush=True)
    return items


def flow_edits(a, items):
    for cname, path in CRIT.items():
        t1 = time.time(); fb = R.load_flow(path); fb.model.eval()
        for it in items:
            hh = it["_h"].to(DEV1); x0 = fb.norm.normalize(hh[None]).float(); co, ce = fb.cond([it["z"]]), fb.cond([it["ze"]]); E = it.setdefault("flow", {})
            for tau in TAUS:
                ns = max(3, int(a.ode_steps * tau))
                xt = R.ode1(fb, x0, co, 0.0, tau, ns); E[f"inv_{cname}_z_t{tau:g}"] = fb.norm.denormalize(R.ode1(fb, xt, ce, tau, 0.0, ns))[0].cpu()
                xt = R.ode1(fb, x0, None, 0.0, tau, ns); E[f"inv_{cname}_u_t{tau:g}"] = fb.norm.denormalize(R.ode1(fb, xt, ce, tau, 0.0, ns))[0].cpu()
                g_ = torch.Generator(device=DEV1).manual_seed(7 + it["n"]); e_ = torch.randn(x0.shape, device=DEV1, generator=g_)
                E[f"sde_{cname}_t{tau:g}"] = fb.norm.denormalize(R.ode1(fb, (1 - tau) * x0 + tau * e_, ce, tau, 0.0, ns))[0].cpu()
        del fb; torch.cuda.empty_cache(); print(f"[B] flow edits with {cname}: {len(items)} prompts in {time.time() - t1:.0f}s", flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--n", type=int, default=len(PROMPTS)); p.add_argument("--k", type=int, default=3); p.add_argument("--n-new", type=int, default=40)
    p.add_argument("--ode-steps", type=int, default=16); p.add_argument("--tag", default=""); p.add_argument("--resume", action="store_true")
    a = p.parse_args(); os.makedirs(OUT, exist_ok=True); t0 = time.time()
    pg._load(); lens = R.Lens(); tok, lm, st = pg.S["tok"], pg.S["lm"], pg.S["st"]; critic = pg._critic(R.AR_NAME); R.install_hook54()
    cache = f"{OUT}/cache{a.tag}.pt"
    if a.resume and os.path.exists(cache):
        items = torch.load(cache, map_location="cpu", weights_only=False); print(f"[resume] {len(items)} prompts with flow edits from {cache}", flush=True)
    else:
        items = stage_a(a, lens); flow_edits(a, items); torch.save(items, cache)
    # ---------------- (2)+(3) steering and measurement
    rows, rbq = [], []
    for it in items:
        h = it["_h"].to(DEV0); h54 = it["_h54"].to(DEV0); ids = it["_ids"].to(DEV0); T = ids.shape[1]; src, tgt = it["src"], it["tgt"]
        ts_, tt_ = R.tid(tok, src), R.tid(tok, tgt); V = torch.stack([lens.vec(ts_), lens.vec(tt_)], 1); V54 = torch.stack([lens.vec(ts_, 54), lens.vec(tt_, 54)], 1)
        d_ar = pg.ar_pred(critic, it["ze"]) - pg.ar_pred(critic, it["z"]); d_jl = V[:, 1] - V[:, 0]
        g_ = torch.Generator(device=DEV0).manual_seed(99 + it["n"]); rnd = torch.randn(h.shape, device=DEV0, generator=g_)
        one = {"none": h}
        for b in (0.5, 1.0): one[f"rand_b{b:g}"] = R.resc(h, rnd, b)
        for al in (1, 2, 4, 8): one[f"jswap_a{al:g}"] = R.swap_fn(V, al)(h[None])[0]
        for b in (0.25, 0.5, 1, 2): one[f"jadd_b{b:g}"] = R.resc(h, d_jl, b)
        for al in (0.5, 1, 2): one[f"ar_a{al:g}"] = R.resc(h, d_ar, al)
        for k_, v_ in it["flow"].items(): one[k_] = v_.to(DEV0)
        for cname in CRIT:
            dinv = it["flow"][f"inv_{cname}_z_t0.5"].to(DEV0) - h
            for al in (0.5, 1, 2): one[f"invdir_{cname}_a{al:g}"] = R.resc(h, dinv, al)
        on = {"none_on": (lambda hb: hb)}
        for al in (1, 2, 4): on[f"jswap_on_a{al:g}"] = R.swap_fn(V, al)
        for b in (0.1, 0.25, 0.5): on[f"jadd_on_b{b:g}"] = (lambda hb, b=b: R.resc(hb, d_jl.expand_as(hb), b))
        for al in (0.25, 0.5, 1): on[f"ar_on_a{al:g}"] = (lambda hb, al=al: R.resc(hb, d_ar.expand_as(hb), al))
        dinv_t = it["flow"]["inv_trunk_z_t0.5"].to(DEV0) - h
        for al in (0.25, 0.5, 1): on[f"invdir_trunk_on_a{al:g}"] = (lambda hb, al=al: R.resc(hb, dinv_t.expand_as(hb), al))
        on["rand_on_b0.25"] = (lambda hb: R.resc(hb, rnd.expand_as(hb), 0.25))
        f54 = {f"jswap54_a{al:g}": R.swap_fn(V54, al) for al in (2, 4, 8)}; f54on = {f"jswap54_on_a{al:g}": R.swap_fn(V54, al) for al in (1, 2, 4)}
        # J-lens monitor + edit size on the edited anchor activation
        n1, nA = list(one), list(on); E1 = torch.stack([one[k] for k in n1]).float(); EA = torch.stack([on[k](h[None].clone())[0] for k in nA]).float()
        lgE = lens.logits(torch.cat([E1, EA]), 42); E5 = torch.stack([f(h54[None])[0] for f in list(f54.values()) + list(f54on.values())]); lg5 = lens.logits(E5, 54)
        out = {}
        for i, nm in enumerate(n1 + nA):
            Ei = torch.cat([E1, EA])[i]
            out[nm] = dict(scope="anchor" if nm in one else "anchor_on", layer=42, tgt_rank=int(R.Lens.rank(lgE[i], tt_)), src_rank=int(R.Lens.rank(lgE[i], ts_)), edit_rel=float((Ei - h).norm() / h.norm()))
        for i, nm in enumerate(list(f54) + list(f54on)):
            out[nm] = dict(scope="anchor_on" if "_on" in nm else "anchor", layer=54, tgt_rank=int(R.Lens.rank(lg5[i], tt_)), src_rank=int(R.Lens.rank(lg5[i], ts_)), edit_rel=float((E5[i] - h54).norm() / h54.norm()))
        groups = [(n1, {nm: (lambda hb, v=one[nm]: v.to(hb.dtype).expand_as(hb)) for nm in n1}, 42, "anchor"), (nA, on, 42, "anchor_on"), (list(f54), f54, 54, "anchor"), (list(f54on), f54on, 54, "anchor_on")]
        for names_g, fmap, layer, scope in groups:
            for greedy, KK in ((True, 1), (False, a.k)):
                gg, sl = run_batch(ids, names_g, fmap, layer, scope, KK, greedy, a.n_new); seqs = gg.sequences
                if greedy: first_lg = gg.scores[0].float()
                for nm, s_ in sl.items():
                    cs = [cont_of(tok, seqs[i], T) for i in range(s_.start, s_.stop)]; out[nm]["greedy" if greedy else "samples"] = cs
                    if greedy: out[nm]["_seq_g"] = seqs[s_.start].cpu(); out[nm]["_lg1"] = first_lg[s_.start]
                    else: out[nm]["_seq_s"] = [seqs[i].cpu() for i in range(s_.start, s_.stop)]
        ref = out["none"]["_lg1"]
        for nm, r in out.items():
            r["kl1"] = float(F.kl_div(torch.log_softmax(ref, -1), torch.log_softmax(r.pop("_lg1"), -1), log_target=True, reduction="sum"))
            cs = r["greedy"] + r["samples"]; r["mentions_tgt"] = [ment(c, tgt) for c in cs]; r["mentions_src"] = [ment(c, src) for c in cs]
            r["nll"] = nll_cont(ids, [r.pop("_seq_g")] + r.pop("_seq_s"), T)
        for c in RB_CONDS:
            if c in one: rbq.append((len(rows), c, one[c].float().cpu()))
        rows.append(dict(n=it["n"], text=it["text"], src=src, tgt=tgt, implied=it["implied"], z=it["z"], ze=it["ze"], z_templated=it["z_templated"], conds=out))
        pick = ["none", "jadd_on_b0.25", "jswap54_on_a2", "ar_on_a0.5", "inv_trunk_z_t0.9", "sde_trunk_t0.9"]
        print(f"[C] {it['n']} {src}->{tgt} ({time.time() - t0:.0f}s): " + " | ".join(f"{c} tgt {sum(out[c]['mentions_tgt'])}/{len(out[c]['mentions_tgt'])}" for c in pick), flush=True)
        print(f"    greedy: none {out['none']['greedy'][:90]!r} | jadd_on_b0.25 {out['jadd_on_b0.25']['greedy'][:90]!r} | ar_on_a0.5 {out['ar_on_a0.5']['greedy'][:90]!r}", flush=True)
        if len(rows) % 4 == 0: json.dump(rows, open(f"{OUT}/steer{a.tag}.partial.json", "w"))
    # verbalizer read-back of the edited anchor activation (warm-start verbalizer, greedy), all prompts in one batch
    ex = pg._verbalize(torch.stack([v for _, _, v in rbq]), R.AV_WARM, temperature=0.0, max_new=200, bs=64)
    for (j, c, _), e in zip(rbq, ex):
        r = rows[j]["conds"][c]; r["readback"] = e; r["readback_names_tgt"] = ment(e, rows[j]["tgt"]); r["readback_names_src"] = ment(e, rows[j]["src"])
    print("[C] read-back names target: " + " ".join(f"{c} {sum(rows[j]['conds'][c]['readback_names_tgt'] for j in range(len(rows)))}/{len(rows)}" for c in RB_CONDS), flush=True)
    json.dump(rows, open(f"{OUT}/steer{a.tag}.json", "w")); print(f"[done] {len(rows)} prompts in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
