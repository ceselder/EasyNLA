"""Rhyme planning in J-space, and steering the plan at different noise levels (Qwen3.6-27B, layer 42).

Stage A (J-space monitor). For couplet prompts "A rhyming couplet:\\n<line 1>\\n" the base model completes line 2 greedily; prompts whose
line-2 end word rhymes with line 1's (CMU dict) and is a single token are kept, and that end word is the model's PLANNED rhyme. On the full
text, the J-lens (camilablank/workspace-lenses, qwen3.6-27b/j-lens, readout softmax(W_U norm(J_l h_l)), layer l = output of decoder block l =
HF hidden_states[l+1] = nla layer_index l) is read at every position from line 1's last word to line 2's last word, at several layers: rank /
probability of the planned word's token and the best rank of any single-token CMU rhyme of line 1's end word. The layer-42 activation at the
last prompt token (the line break) is verbalized by two verbalizers (greedy + 4 samples).

Stage B (edits at the line break). Two alternative plans per prompt: the best-ranked OTHER rhyme in J-space (same rhyme sound) and a common word
with a DIFFERENT rhyme sound. Edits of the layer-42 activation at the line break ("one position" scope) and at every position of the prompt and
of the generated line ("every position" scope):
  J-lens coordinate swap (the paper's §2.5 patching: c = V^+ h for V = [v_s, v_t], v = J_42^T W_U[tok]; h + V(alpha*swap(c) - c)), J-lens
  additive direction h + b||h|| unit(v_t - v_s), MSE-reconstructor direction h + a||h|| unit(AR(z') - AR(z)), flow inversion at noise level tau
  (ODE h -> x_tau under z or under the unconditional prior, then tau -> 0 under z'), SDEdit at tau (noise to tau, denoise under z'), the
  inversion displacement as an alpha-rescaled direction, and random-direction controls. z = the warm-start verbalizer's greedy explanation of
  the line-break activation (+ one sentence naming the planned rhyme if it does not already), z' = z with the planned word -> target.
Stage C (outcome). Line 2 under every edit (greedy + 4 samples at T=1): hits the target word / rhymes with the target / still rhymes with line 1
/ keeps the original plan; J-lens rank of target and planned word on the EDITED line-break activation; KL at the first generated token; edit
size ||h'-h||/||h||; base-model NLL of the generated line (comprehensibility; the Sonnet fluency judge runs offline in the report plotter).
Outputs /vol_glp/cond/rhyme/{jspace,steer}.json."""
import argparse, json, math, os, re, sys, time
import numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import intervene_playground_app as pg
import pronouncing

OUT = "/vol_glp/cond/rhyme"; DEV0, DEV1 = pg.DEV0, pg.DEV1
LAYERS_MON = [24, 30, 36, 42, 48, 54, 60]
LINE1 = ["He saw a carrot and had to grab it,", "The soldier marched into the night,", "The cat sat down upon the mat,", "She walked along the sandy shore,",
         "The old man sat beside the fire,", "I wandered lonely through the town,", "The morning sun began to rise,", "A little bird sang in the tree,",
         "He opened up the dusty book,", "The children laughed and ran to play,", "The wind was howling through the trees,", "She wore a dress of deepest blue,",
         "The king sat on his golden throne,", "We climbed up to the mountain peak,", "The baker woke before the dawn,", "A lonely wolf howled at the moon,",
         "The garden bloomed in early spring,", "He tried to fix the broken clock,", "The rain came pouring from the sky,", "She planted roses by the gate,",
         "The farmer walked behind his plow,", "The river wound around the hill,", "My dog ran off to chase a ball,", "The candle flickered on the shelf,",
         "The teacher wrote upon the board,", "A stranger knocked upon the door,", "The snow fell softly on the ground,", "He climbed aboard the rusty train,",
         "The painter mixed a brand new shade,", "The fisherman cast out his line,", "She baked a cake for her best friend,", "The ship was sailing out to sea,"]
DIFF_WORDS = ["moon", "light", "sea", "heart", "day", "tree", "sky", "fire", "road", "dream", "gold", "rain", "song", "home", "stone", "bread"]
CRIT = {"sw": "/vol_glp/cond/sw_tokar/adapter_latest.pt", "trunk": "/vol_glp/cond/trunk_dn64/adapter_latest.pt"}
AV_WARM = "warm start (SFT verbalizer, iter_0007813)"; AV_NOKL = "fast 128×8, whole-trunk flow critic, NO KL, step 100 (best judged checkpoint)"
AR_NAME = "SFT reconstructor (ar_sft_merged, pre-RL)"


def last_word(s):
    w = re.findall(r"[A-Za-z']+", s or ""); return w[-1].strip("'") if w else ""
def rparts(w): return {pronouncing.rhyming_part(p) for p in pronouncing.phones_for_word(w.lower())}
def rhymes(a, b):
    if not a or not b or a.lower() == b.lower(): return False
    ra, rb = rparts(a), rparts(b); return bool(ra and rb and (ra & rb))


class Lens:
    def __init__(self):
        from huggingface_hub import hf_hub_download
        p = hf_hub_download("camilablank/workspace-lenses", "qwen3.6-27b/j-lens/lens.pt", token=os.environ.get("HF_TOKEN"), local_dir="/vol_glp/jlens")
        L = torch.load(p, map_location="cpu", weights_only=False); J = L["J"]; key = lambda l: l if l in J else str(l)
        self.J = {l: J[key(l)].to(DEV0, torch.bfloat16) for l in LAYERS_MON}; del L, J
        lm = pg.S["lm"]; inner = lm.model; owner = inner.language_model if hasattr(inner, "language_model") else inner
        self.norm = owner.norm; self.WU = lm.lm_head.weight                                                     # [V, d] bf16
        print(f"[lens] J-lens layers {LAYERS_MON} loaded; W_U {tuple(self.WU.shape)}", flush=True)
    @torch.no_grad()
    def logits(self, H, l):                                                                                   # H [N, d] -> [N, V] fp32
        out = []
        for i in range(0, H.shape[0], 64):
            x = (H[i:i + 64].to(DEV0, torch.bfloat16) @ self.J[l].T); out.append((self.norm(x) @ self.WU.T).float())
        return torch.cat(out)
    def vec(self, tid, l=42): return (self.WU[tid].float() @ self.J[l].float())                                 # J-lens vector v_t = row t of W_U J_l, in layer-l space
    @staticmethod
    def rank(lg, tid): return (lg > lg[..., tid:tid + 1]).sum(-1)


def single(tok, w): return len(tok(" " + w, add_special_tokens=False)["input_ids"]) == 1
def tid(tok, w): return tok(" " + w, add_special_tokens=False)["input_ids"][0]


def gen(ids, K, greedy, n_new=24, prefill_fn=None, prefill_idx=None, decode_fn=None, scores=False):
    tok, lm, st = pg.S["tok"], pg.S["lm"], pg.S["st"]; T = ids.shape[1]
    with pg.S["gpu"], pg._base():
        st.update(cap=None, vec=None, decode_fn=decode_fn, prefill_fn=prefill_fn, prefill_idx=prefill_idx if prefill_idx is not None else torch.tensor([T - 1], device=DEV0))
        try:
            with torch.no_grad():
                g = lm.generate(input_ids=ids.expand(K, -1), attention_mask=torch.ones(K, T, device=DEV0, dtype=torch.long), do_sample=not greedy, temperature=1.0, top_p=1.0, top_k=0,
                                max_new_tokens=n_new, pad_token_id=tok.pad_token_id or tok.eos_token_id, output_scores=scores, return_dict_in_generate=True)
        finally: st.update(prefill_fn=None, decode_fn=None)
    return g


def line2_of(tok, seq_ids, T):
    txt = tok.decode(seq_ids[T:].tolist(), skip_special_tokens=True); return txt.split("\n")[0].strip() if txt.strip() else ""


# ------------------------------------------------------------------ stage A
def stage_a(a, lens):
    tok, lm, st = pg.S["tok"], pg.S["lm"], pg.S["st"]; items = []
    for n, l1 in enumerate(LINE1[: a.n_prompts]):
        prompt = f"A rhyming couplet:\n{l1}\n"; ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEV0); T = ids.shape[1]
        g = gen(ids, 1, True, n_new=24).sequences[0]; l2 = line2_of(tok, g, T); w1, w2 = last_word(l1), last_word(l2)
        rec = dict(n=n, line1=l1, line2=l2, w1=w1, w2=w2, rhymes=rhymes(w2, w1), single=bool(w2) and single(tok, w2))
        rec["usable"] = rec["rhymes"] and rec["single"]
        if not rec["usable"]: items.append(rec); print(f"[A] skip {n}: {l1} / {l2}", flush=True); continue
        # full sequence up to and including the token that emits w2
        gen_ids = g[T:].tolist(); j_emit = None; acc = ""
        for j, t_ in enumerate(gen_ids):
            acc += tok.decode([t_])
            if re.search(rf"\b{re.escape(w2)}\b", acc): j_emit = j; break
        full = g[: T + (j_emit if j_emit is not None else len(gen_ids)) + 1][None]
        p_start = T - 1
        for k in range(T - 1, 0, -1):
            if w1.lower() in tok.decode(ids[0, k:T].tolist()).lower(): p_start = k; break
        p_nl, p_emit = T - 1, T + (j_emit or 0) - 1                                                             # line break = last prompt token; p_emit predicts w2
        with pg.S["gpu"], pg._base():
            st.update(cap=None, vec=None, decode_fn=None, prefill_fn=None)
            with torch.no_grad(): hs = lm(input_ids=full, output_hidden_states=True).hidden_states
        pos = list(range(p_start, full.shape[1])); tP = tid(tok, w2)
        fam = [w for w in pronouncing.rhymes(w1.lower()) if w.isalpha() and len(w) >= 3 and single(tok, w)]
        fam_ids = sorted({tid(tok, w) for w in fam} | {tP}); famt = torch.tensor(fam_ids, device=DEV0)
        conc_ids = sorted({tid(tok, w) for w in ("rhyme", "rhyming", "poem", "poetry", "poetic", "verse")}); conct = torch.tensor(conc_ids, device=DEV0)
        mon = {}
        for l in LAYERS_MON:
            lg = lens.logits(hs[l + 1][0, pos], l); r = Lens.rank(lg, tP); pr = torch.softmax(lg, -1)[:, tP]
            fr = (lg > lg[:, famt].max(1, keepdim=True).values).sum(-1)                                          # rank of the best-ranked rhyme-family token
            top = [[tok.decode([int(x)]) for x in lg[i].topk(10).indices] for i in range(len(pos))]
            cr = (lg > lg[:, conct].max(1, keepdim=True).values).sum(-1)                                           # best rank of the rhyme/poem concept tokens
            mon[l] = dict(rank=r.tolist(), prob=pr.tolist(), family_best_rank=fr.tolist(), concept_best_rank=cr.tolist(), top10=top)
        H42 = hs[43][0].float(); h_nl = H42[p_nl].clone(); h54_nl = hs[55][0, p_nl].float().clone(); h60_emit = hs[61][0, p_emit].float().clone(); del hs
        lg42 = lens.logits(h_nl[None], 42)[0]
        # same-rhyme alternative = the model's own runner-up rhyme when it is about to emit (layer-60 J-lens at the emit position), no homophones
        ph2 = set(pronouncing.phones_for_word(w2.lower())); ph1 = set(pronouncing.phones_for_word(w1.lower()))
        lg60 = lens.logits(h60_emit[None], 60)[0]
        cand = sorted(((int(Lens.rank(lg60, tid(tok, w))), w) for w in fam if w.lower() not in (w2.lower(), w1.lower()) and tid(tok, w) != tP
                       and not (set(pronouncing.phones_for_word(w)) & (ph2 | ph1))))
        alt_same = cand[0][1] if cand else None
        rot = DIFF_WORDS[n % len(DIFF_WORDS):] + DIFF_WORDS[: n % len(DIFF_WORDS)]
        alt_diff = next((w for w in rot if not rhymes(w, w1) and w.lower() != w2.lower() and single(tok, w)), None)
        # verbalize the line-break activation
        verb = {}
        for key, av in (("warm", AV_WARM), ("nokl100", AV_NOKL)):
            vg = pg._verbalize(h_nl[None], av, temperature=0.0, max_new=200)[0]; vs = pg._verbalize(h_nl[None].repeat(4, 1), av, temperature=1.0, max_new=200)
            verb[key] = dict(greedy=vg, samples=vs)
        rec.update(prompt=prompt, T=T, p_start=p_start, p_nl=p_nl, p_emit=p_emit, positions=pos, tokens=[tok.decode([int(x)]) for x in full[0, pos]], planned_tid=tP,
                   monitor={str(l): v for l, v in mon.items()}, family=fam[:200], alt_same=alt_same, alt_same_rank_emit60=(cand[0][0] if cand else None), alt_diff=alt_diff,
                   planned_rank_nl42=int(Lens.rank(lg42, tP)), top10_nl42=[tok.decode([int(x)]) for x in lg42.topk(10).indices], verbalizer=verb)
        rec["_h_nl"] = h_nl.cpu(); rec["_h54_nl"] = h54_nl.cpu(); rec["_ids"] = ids.cpu(); items.append(rec)
        print(f"[A] {n}: '{w1}' -> planned '{w2}' | J42 rank at line break {rec['planned_rank_nl42']} | alt same '{alt_same}' diff '{alt_diff}' | warm verb mentions: {w2.lower() in vg.lower()}", flush=True)
    return items


# ------------------------------------------------------------------ flow helpers
def load_flow(path):
    from nla.flow.scoring import FlowBundle
    aa = torch.load(path, map_location="cpu")["args"]; pco = os.path.join(os.path.dirname(path), "prior_cotrained_latest.pt")
    return FlowBundle(aa["prior"], path, aa["stats"], DEV1, base=pg.S["snap"], enc_layer=aa.get("enc_layer", 42), ar_ckpt=aa.get("ar_ckpt", "/vol/ckpts/qwen36_27b/ar_sft_merged"),
                      prior_override=pco if os.path.exists(pco) else None)

def ode1(fb, x, cond, t0, t1, steps):
    """Heun probability-flow ODE for a single activation (B = 1): cond = fb.cond([z]) (token states or trunk ids) or None = unconditional prior."""
    enc, mk, cv = cond if cond is not None else (None, None, None); ts = torch.linspace(t0, t1, steps + 1, device=x.device)
    def v(x_, t_):
        tt = torch.full((1,), float(t_), device=x.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return (fb.model(x_, tt, enc, mk, cv) if cond is not None else fb.model(x_, tt)).float()
    for i in range(steps):
        h = ts[i + 1] - ts[i]; v0 = v(x, ts[i]); xp = x + h * v0; v1 = v(xp, ts[i + 1]); x = x + h * 0.5 * (v0 + v1)
    return x


# ------------------------------------------------------------------ stage B + C
def resc(h, d, a_): return h + a_ * h.norm(dim=-1, keepdim=True) * d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6)

def swap_fn(V, alpha):
    """lens-coordinate swap of the paper (§2.5): c = V^+ h, h + V(alpha * swap(c) - c); V = [v_s, v_t] columns."""
    Vp = torch.linalg.pinv(V)                                                                                  # [2, d]
    def f(hb):
        c = hb @ Vp.T; cs = alpha * torch.stack([c[..., 1], c[..., 0]], -1); return hb + (cs - c) @ V.T
    return f


ST54 = {"prefill_fn": None, "idx": None, "decode_fn": None}
def install_hook54():
    from nla.utils.arch_adapters import resolve_decoder_layers
    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] > 1:
            if ST54["prefill_fn"] is not None: ix = ST54["idx"]; h[:, ix] = ST54["prefill_fn"](h[:, ix].float()).to(h.dtype)
        elif ST54["decode_fn"] is not None: h[:, 0] = ST54["decode_fn"](h[:, 0].float()).to(h.dtype)
        return out
    resolve_decoder_layers(pg.S["lm"])[54].register_forward_hook(hook); print("[B] layer-54 steering hook installed", flush=True)


def run_batch(ids, names, fnmap, layer, scope, K, greedy):
    """generate line 2 for every condition in `names` in one batch (K rows each); fnmap[name] edits h[..., d] at `layer`; scope 'one' = the
    line-break position only, 'all' = every prompt position and every generated position."""
    T = ids.shape[1]; sl = {nm: slice(i * K, (i + 1) * K) for i, nm in enumerate(names)}
    def comp(hb):
        o = hb.clone()
        for nm, s_ in sl.items(): o[s_] = fnmap[nm](hb[s_].float()).to(hb.dtype)
        return o
    idx = torch.tensor([T - 1], device=DEV0) if scope == "one" else torch.arange(T, device=DEV0); dec = comp if scope == "all" else None
    if layer == 42: g = gen(ids, len(names) * K, greedy, prefill_fn=comp, prefill_idx=idx, decode_fn=dec, scores=greedy)
    else:
        ST54.update(prefill_fn=comp, idx=idx, decode_fn=dec)
        try: g = gen(ids, len(names) * K, greedy, scores=greedy)
        finally: ST54.update(prefill_fn=None, idx=None, decode_fn=None)
    return g, sl


def stage_bc(a, lens, items):
    install_hook54()
    tok = pg.S["tok"]; critic = pg._critic(AR_NAME); usable = [it for it in items if it.get("usable") and it.get("alt_same") and it.get("alt_diff")]
    TAUS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9] if not a.smoke else [0.3, 0.5]; TAU_U = [0.3, 0.5, 0.7, 0.9] if not a.smoke else [0.5]
    jobs = []                                                                                                 # (item, kind, target, z, z')
    for it in usable:
        z0 = it["verbalizer"]["warm"]["greedy"]; w2 = it["w2"]
        z = z0 if re.search(rf"\b{re.escape(w2)}\b", z0, re.I) else z0.rstrip() + f' The second line of the couplet will end with the rhyming word "{w2}".'
        for kind in ("same_rhyme", "diff_rhyme"):
            tgt = it["alt_same"] if kind == "same_rhyme" else it["alt_diff"]; ze = re.sub(rf"\b{re.escape(w2)}\b", tgt, z, flags=re.I)
            jobs.append(dict(it=it, kind=kind, target=tgt, z=z, ze=ze))
    # --- flow-based edited vectors (both critics, one at a time on cuda:1)
    for cname, path in CRIT.items():
        t0 = time.time(); fb = load_flow(path); fb.model.eval()
        for jb in jobs:
            h = jb["it"]["_h_nl"].to(DEV1); x0 = fb.norm.normalize(h[None]).float(); co, ce = fb.cond([jb["z"]]), fb.cond([jb["ze"]]); S = a.ode_steps
            E = jb.setdefault("flow", {})
            for tau in TAUS:
                xt = ode1(fb, x0, co, 0.0, tau, max(3, int(S * tau))); E[f"inv_{cname}_z_t{tau:g}"] = fb.norm.denormalize(ode1(fb, xt, ce, tau, 0.0, max(3, int(S * tau))))[0].cpu()
                if tau == 0.5: E[f"inv_{cname}_zz_t0.5"] = fb.norm.denormalize(ode1(fb, xt, co, tau, 0.0, max(3, int(S * tau))))[0].cpu()   # round-trip control
            for tau in TAU_U:
                xt = ode1(fb, x0, None, 0.0, tau, max(3, int(S * tau))); E[f"inv_{cname}_u_t{tau:g}"] = fb.norm.denormalize(ode1(fb, xt, ce, tau, 0.0, max(3, int(S * tau))))[0].cpu()
                g_ = torch.Generator(device=DEV1).manual_seed(1234 + jb["it"]["n"]); e_ = torch.randn(x0.shape, device=DEV1, generator=g_)
                E[f"sde_{cname}_t{tau:g}"] = fb.norm.denormalize(ode1(fb, (1 - tau) * x0 + tau * e_, ce, tau, 0.0, max(3, int(S * tau))))[0].cpu()
        del fb; torch.cuda.empty_cache(); print(f"[B] flow edits with {cname}: {len(jobs)} jobs in {time.time() - t0:.0f}s", flush=True)
    # --- conditions per job
    rows = []
    for jb in jobs:
        it = jb["it"]; h = it["_h_nl"].to(DEV0); ids = it["_ids"].to(DEV0); T = ids.shape[1]; w2, tgt = it["w2"], jb["target"]
        ts_, tt_ = tid(tok, w2), tid(tok, tgt); V = torch.stack([lens.vec(ts_), lens.vec(tt_)], 1)             # [d, 2]
        d_ar = pg.ar_pred(critic, jb["ze"]) - pg.ar_pred(critic, jb["z"]); g_ = torch.Generator(device=DEV0).manual_seed(99 + it["n"]); rnd = torch.randn(h.shape, device=DEV0, generator=g_)
        d_jl = V[:, 1] - V[:, 0]
        one = {"none": h}                                                                                     # scope: line break only -> precomputed vectors
        for b in (0.25, 0.5, 1.0): one[f"rand_b{b:g}"] = resc(h, rnd, b)
        for al in (1, 2, 4, 8): one[f"jswap_a{al:g}"] = swap_fn(V, al)(h[None])[0]
        for b in (0.25, 0.5, 1, 2): one[f"jadd_b{b:g}"] = resc(h, d_jl, b)
        for al in (0.5, 1, 2): one[f"ar_a{al:g}"] = resc(h, d_ar, al)
        for k_, v_ in jb["flow"].items(): one[k_] = v_.to(DEV0)
        for cname in CRIT:
            dinv = jb["flow"][f"inv_{cname}_z_t0.5"].to(DEV0) - h
            for al in (0.5, 1, 2): one[f"invdir_{cname}_a{al:g}"] = resc(h, dinv, al)
        allf = {"none_all": (lambda hb: hb)}                                                                  # scope: every prompt + generated position
        for al in (1, 2, 4): allf[f"jswap_all_a{al:g}"] = swap_fn(V, al)
        for b in (0.1, 0.25, 0.5): allf[f"jadd_all_b{b:g}"] = (lambda hb, b=b: resc(hb, d_jl.expand_as(hb), b))
        for al in (0.25, 0.5, 1): allf[f"ar_all_a{al:g}"] = (lambda hb, al=al: resc(hb, d_ar.expand_as(hb), al))
        dinv_t = jb["flow"]["inv_trunk_z_t0.5"].to(DEV0) - h
        for al in (0.25, 0.5, 1): allf[f"invdir_trunk_all_a{al:g}"] = (lambda hb, al=al: resc(hb, dinv_t.expand_as(hb), al))
        allf["rand_all_b0.25"] = (lambda hb: resc(hb, rnd.expand_as(hb), 0.25))
        # J-lens monitor on the edited line-break activation
        names1 = list(one); E1 = torch.stack([one[k] for k in names1]).float()
        namesA = list(allf); EA = torch.stack([allf[k](h[None].clone())[0] for k in namesA]).float()
        lgE = lens.logits(torch.cat([E1, EA]), 42); rt = Lens.rank(lgE, tt_).tolist(); rp = Lens.rank(lgE, ts_).tolist()
        edit_rel = ((torch.cat([E1, EA]) - h[None]).norm(dim=-1) / h.norm()).tolist()
        mon = {nm: dict(target_rank=rt[i], planned_rank=rp[i], edit_rel=edit_rel[i]) for i, nm in enumerate(names1 + namesA)}
        out = {nm: dict(scope="one", **mon[nm]) for nm in names1} | {nm: dict(scope="all", **mon[nm]) for nm in namesA}
        K = a.k
        V54 = torch.stack([lens.vec(ts_, 54), lens.vec(tt_, 54)], 1); h54 = it["_h54_nl"].to(DEV0)
        f54 = {f"jswap54_a{al:g}": swap_fn(V54, al) for al in (1, 2, 4, 8)}; f54a = {f"jswap54_all_a{al:g}": swap_fn(V54, al) for al in (1, 2, 4)}
        E54 = torch.stack([f(h54[None])[0] for f in list(f54.values()) + list(f54a.values())]); lg54 = lens.logits(E54, 54)
        for i, nm in enumerate(list(f54) + list(f54a)):
            out[nm] = dict(scope="all" if "_all" in nm else "one", layer=54, target_rank=int(Lens.rank(lg54[i], tt_)), planned_rank=int(Lens.rank(lg54[i], ts_)), edit_rel=float((E54[i] - h54).norm() / h54.norm()))
        groups = [(names1, {nm: (lambda hb, v=one[nm]: v.to(hb.dtype).expand_as(hb)) for nm in names1}, 42, "one"), (namesA, allf, 42, "all"), (list(f54), f54, 54, "one"), (list(f54a), f54a, 54, "all")]
        for names_g, fmap, layer, scope in groups:
            for greedy, KK in ((True, 1), (False, K)):
                gg, sl = run_batch(ids, names_g, fmap, layer, scope, KK, greedy); seqs = gg.sequences
                if greedy: first_lg = gg.scores[0].float()
                for nm, s_ in sl.items():
                    ls = [line2_of(tok, seqs[i], T) for i in range(s_.start, s_.stop)]
                    out[nm]["greedy" if greedy else "samples"] = ls
                    if greedy: out[nm]["_seq_g"] = seqs[s_.start].cpu(); out[nm]["_lg1"] = first_lg[s_.start]
                    else: out[nm]["_seq_s"] = [seqs[i].cpu() for i in range(s_.start, s_.stop)]
        # outcome metrics + KL + base-model NLL of the generated line
        base_lg = out["none"]["_lg1"]; base_lgA = out["none_all"]["_lg1"]
        for nm, r in out.items():
            ref = base_lg if r["scope"] == "one" else base_lgA
            r["kl1"] = float(F.kl_div(torch.log_softmax(ref, -1), torch.log_softmax(r.pop("_lg1"), -1), log_target=True, reduction="sum"))
            lines = [r["greedy"][0]] + r["samples"]; ws = [last_word(x) for x in lines]
            r["end_words"] = ws; r["hit_target"] = [w.lower() == tgt.lower() for w in ws]; r["rhymes_target"] = [w.lower() == tgt.lower() or rhymes(w, tgt) for w in ws]
            r["rhymes_line1"] = [rhymes(w, it["w1"]) for w in ws]; r["kept_plan"] = [w.lower() == w2.lower() for w in ws]
            r["nll"] = nll_lines(ids, [r.pop("_seq_g")] + r.pop("_seq_s"), T)
        rows.append(dict(n=it["n"], kind=jb["kind"], target=tgt, planned=w2, w1=it["w1"], line1=it["line1"], z=jb["z"], ze=jb["ze"], conds=out))
        print(f"[C] {it['n']} {jb['kind']} '{w2}'->'{tgt}': none {out['none']['end_words'][:3]} | jswap54_all_a2 {out['jswap54_all_a2']['end_words'][:3]} | jswap_a4 {out['jswap_a4']['end_words'][:3]} | jswap_all_a2 {out['jswap_all_a2']['end_words'][:3]} "
              f"| inv_trunk_z_t0.5 {out['inv_trunk_z_t0.5']['end_words'][:3]} | ar_all_a0.5 {out['ar_all_a0.5']['end_words'][:3]}", flush=True)
    return rows


@torch.no_grad()
def nll_lines(ids, seqs, T):
    """mean NLL per token of each generated line (up to and incl. the first newline) under the UNSTEERED base model, given the prompt (batched)."""
    tok, lm, st = pg.S["tok"], pg.S["lm"], pg.S["st"]; pad = tok.pad_token_id or tok.eos_token_id; eos = tok.eos_token_id
    L = max(int(s_.shape[0]) for s_ in seqs); X = torch.full((len(seqs), L), pad, dtype=torch.long)
    for i, s_ in enumerate(seqs): X[i, : s_.shape[0]] = s_
    cuts = []
    for i in range(len(seqs)):
        g_ = X[i, T:].tolist(); cut = len(g_)
        for j, t_ in enumerate(g_):
            if t_ == eos: cut = j; break
            if "\n" in tok.decode([t_]): cut = j + 1; break
        cuts.append(cut)
    out = [None] * len(seqs)
    for c0 in range(0, len(seqs), 32):
        xb = X[c0:c0 + 32].to(DEV0)
        with pg.S["gpu"], pg._base():
            st.update(cap=None, vec=None, decode_fn=None, prefill_fn=None); lg = lm(input_ids=xb).logits[:, T - 1: L - 1].float()
        ce = F.cross_entropy(lg.transpose(1, 2), xb[:, T:], reduction="none")                                  # [b, L-T]
        for i in range(xb.shape[0]):
            c = cuts[c0 + i]; out[c0 + i] = float(ce[i, :c].mean()) if c > 0 else None
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("--n-prompts", type=int, default=len(LINE1)); p.add_argument("--k", type=int, default=4); p.add_argument("--ode-steps", type=int, default=16)
    p.add_argument("--smoke", action="store_true"); p.add_argument("--tag", default="")
    a = p.parse_args(); os.makedirs(OUT, exist_ok=True); t0 = time.time()
    pg._load(); lens = Lens()
    items = stage_a(a, lens)
    json.dump([{k: v for k, v in it.items() if not k.startswith("_")} for it in items], open(f"{OUT}/jspace{a.tag}.json", "w"))
    print(f"[A] {sum(1 for it in items if it.get('usable'))} usable of {len(items)} prompts; {time.time() - t0:.0f}s", flush=True)
    rows = stage_bc(a, lens, items)
    json.dump(rows, open(f"{OUT}/steer{a.tag}.json", "w")); print(f"[done] {len(rows)} (prompt, target) jobs in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
