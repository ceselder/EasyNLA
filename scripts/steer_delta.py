"""Delta-denoiser steering: plug the CLEAN activation into the conditional flow, change the explanation, and let the flow move it (no added noise).

Same next-token concept-swap eval as scripts/unclip_steer_v2.py (reuses its cache: 48 prompts, h = layer-42 anchor activation, zA = the warm
verbalizer's own explanation of h, zAe = zA with the concept word swapped; zB/zBe = the short concept-centred pair) and the same measurement,
so every number is comparable with the v2 frontier. Unlike the v2 conditional-mean direction (computed from pure noise, independent of h), every
edit here is evaluated AT h:

  dlt_t{t}_b{b}     h + b|h| unit(D_t),  D_t = x0hat(h, t, z') - x0hat(h, t, z),  x0hat(x, t, c) = x - t v(x, t, c)   (one forward pass per condition;
                    for this flow D_t = t^2/(1-t) grad_h [log p_t(z'|h) - log p_t(z|h)], i.e. ascent on how much better z' explains h than z does)
  dltavg_b{b}       the same with unit(D_t) averaged over t in {0.05, 0.1, 0.2, 0.3, 0.5, 0.7}
  dltit_b{b}        8 small steps of size b|h|/8 along unit(D_0.2), re-evaluating D at the moved point, kept on the |h| sphere
  run_t{tau}        literal version: treat the clean h as the state at tau and integrate the probability-flow ODE tau -> 0 under z' (no noise)
  rundiff_t{tau}_b{b}  h + b|h| unit(run under z' - run under z)   (cancels the shared "h is too clean for this t" response)
Models: flow critics (sw_tokar, trunk_dn64; token-conditioned adapters on the 13.7B prior) and the unCLIP decoder p(h|e) with two condition
pairs: uctext = (g(z), g(z')) text embeddings, ucediff = (e, renorm(e + 4 (g(z') - g(z)))) with e = f(h).
Every edited vector is rescaled to |h| before it replaces the anchor (direction only), exactly as in v2.
usage: python scripts/steer_delta.py --tag T [--models sw_tokar,trunk_dn64,uctext,ucediff] [--smoke]   -> /vol_glp/unclip/steer/<tag>.json"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import rhyme_plan_steer as R
import animal_swap_steer as A
import unclip_steer_v2 as V2
from nla.unclip.steer_models import Encoder, Decoder
pg = R.pg; DEV0, DEV1 = R.DEV0, R.DEV1; OUT = "/vol_glp/unclip/steer"
T_AVG = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7)
_FAM0 = V2.family_of                                                      # v2 families for the reference rows (V2.family_of is swapped for family() later)
unit = lambda d: d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8)


class FlowDen:
    """token-conditioned flow critic (FlowBundle): batched velocity under per-prompt text conditions."""
    CH = 16
    def __init__(self, fb):
        assert not fb.aa.get("resid_shift"), "resid_shift adapters not supported"
        self.fb, self.norm = fb, fb.norm; self.claims = int(fb.aa.get("claim_subsets") or 0) > 0
        assert not (self.claims and fb.aa.get("set_encode")), "set-encoded claim critics not supported"
        if self.claims: print("[B] claim-set critic: conditions formatted as bullet claims exactly as in training (split_claims -> format_claims)", flush=True)
    def fmt(self, z):
        """claim-set critics (train_cond --claim-subsets > 0) were trained on '• c' lines, one per claim; other critics read the text raw"""
        if not self.claims: return z
        from nla.flow.claims import split_claims, format_claims
        return format_claims(split_claims(z) or [z])
    def conds(self, texts):
        texts = [self.fmt(z) for z in texts]
        return [self.fb.cond(list(texts[i:i + self.CH])) for i in range(0, len(texts), self.CH)]
    @torch.no_grad()
    def vel(self, x, t, cs):
        out = []
        for j, (enc, mk, cv) in enumerate(cs):
            xb = x[j * self.CH:(j + 1) * self.CH]; tt = torch.full((xb.shape[0],), float(t), device=x.device)
            with torch.autocast("cuda", dtype=torch.bfloat16): out.append(self.fb.model(xb, tt, enc, mk, cv).float())
        return torch.cat(out)


class DecDen:
    """unCLIP decoder p(h | e): conditions are embedding vectors [N, d_e]."""
    def __init__(self, dec): self.dec, self.norm = dec, dec.norm
    @torch.no_grad()
    def vel(self, x, t, c): return self.dec.v(x, t, c)


@torch.no_grad()
def x0hat_raw(m, x, t, c): return m.norm.denormalize(x - t * m.vel(x, t, c)).float()

@torch.no_grad()
def delta_raw(m, H, t, c0, c1):
    x = m.norm.normalize(H).float(); return x0hat_raw(m, x, t, c1) - x0hat_raw(m, x, t, c0)

@torch.no_grad()
def run_raw(m, H, tau, c, steps):
    x = m.norm.normalize(H).float(); ts = torch.linspace(tau, 0.0, steps + 1, device=x.device)
    for i in range(steps):
        dt = ts[i + 1] - ts[i]; v0 = m.vel(x, ts[i], c); xp = x + dt * v0; v1 = m.vel(xp, ts[i + 1], c); x = x + dt * 0.5 * (v0 + v1)
    return m.norm.denormalize(x).float()


def model_edits(m, H, c0, c1, a, pre):
    """-> raw edited activations {name: [N, d]} and edit directions {name: [N, d]} (for the J-lens-cosine diagnostic)."""
    V, D = {}, {}; hn = H.norm(dim=-1, keepdim=True)
    for t in a.ts:
        d = delta_raw(m, H, t, c0, c1); D[f"{pre}dlt_t{t:g}"] = d
        for b in a.betas: V[f"{pre}dlt_t{t:g}_b{b:g}"] = R.resc(H, d, b)
    if not a.no_avg:
        davg = torch.stack([unit(delta_raw(m, H, t, c0, c1)) for t in T_AVG]).mean(0); D[f"{pre}dltavg"] = davg
        for b in a.betas: V[f"{pre}dltavg_b{b:g}"] = R.resc(H, davg, b)
    if not a.delta_only:
        for b in a.it_betas:
            hk = H.clone()
            for _ in range(a.it_steps):
                hk = hk + (b / a.it_steps) * hn * unit(delta_raw(m, hk, a.it_t, c0, c1)); hk = hk / hk.norm(dim=-1, keepdim=True) * hn
            V[f"{pre}dltit_b{b:g}"] = hk
        for tau in a.taus:
            ns = max(4, int(round(16 * tau))); r1 = run_raw(m, H, tau, c1, ns); r0 = run_raw(m, H, tau, c0, ns)
            if not a.no_run_literal: V[f"{pre}run_t{tau:g}"] = r1
            D[f"{pre}rundiff_t{tau:g}"] = r1 - r0
            for b in a.run_betas: V[f"{pre}rundiff_t{tau:g}_b{b:g}"] = R.resc(H, r1 - r0, b)
    return V, D


CLAIM = 'The next word will be "{w}".'
CLAIM_N = "The model expects the next word to be '{w}'."          # the single-claim critics' own next_token template (scripts/claims_extract.py)
DM_PROMPTS = ("Here is a short story about a {w}. The {w}", "Some facts about the {w}: the {w}", "I saw a {w} yesterday. The {w}")


def mention_index(ids_list, tid):
    """last position before the final one that holds the source concept token (where a copy prompt's filler lives), or None"""
    for j in range(len(ids_list) - 2, -1, -1):
        if ids_list[j] == tid: return j
    return None


def run_batch_sites(ids, names, specs, pos, K, greedy, n_new):
    """K rows per condition; specs[nm] = {index into pos: replacement layer-42 vector}; unedited positions keep their activation"""
    sl = {nm: slice(i * K, (i + 1) * K) for i, nm in enumerate(names)}
    def comp(hb):
        o = hb.clone()
        for nm, s_ in sl.items():
            for pi, v in specs[nm].items(): o[s_, pi] = v.to(o.dtype)
        return o
    g = R.gen(ids, len(names) * K, greedy, n_new=n_new, prefill_fn=comp, prefill_idx=torch.tensor(pos, device=DEV0), scores=greedy)
    return g, sl


@torch.no_grad()
def build_diffmean(a, items):
    """label-shared difference-of-means directions from passages Qwen writes itself: dmn = mean layer-42 activation at positions whose NEXT
    token is the concept (target minus source), dmp = mean over all passage positions (AxBench-style DiffMean)."""
    tok, lm, st = pg.S["tok"], pg.S["lm"], pg.S["st"]; eos = tok.eos_token_id; words = sorted({it["src"] for it in items} | {it["tgt"] for it in items})
    mu_n, mu_p, cnt = {}, {}, {}
    if a.dm_cache and os.path.exists(a.dm_cache):
        c_ = torch.load(a.dm_cache, map_location="cpu"); mu_n, mu_p, cnt = c_["mu_n"], c_["mu_p"], c_["cnt"]; print(f"[A] DiffMean means from {a.dm_cache} ({c_.get('samples')} samples x {len(DM_PROMPTS)} prompts)", flush=True)
    for w in [w for w in words if w not in mu_n]:
        torch.manual_seed(1000 + words.index(w))
        tid = R.tid(tok, w); hn_, hp_ = [], []
        for pr in DM_PROMPTS:
            ids = tok(pr.format(w=w), return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEV0)
            seqs = R.gen(ids, a.dm_samples, False, n_new=a.dm_tokens).sequences
            with pg.S["gpu"], pg._base():
                st.update(cap=None, vec=None, decode_fn=None, prefill_fn=None); o = lm(input_ids=seqs, output_hidden_states=True)
            H = o.hidden_states[43].float(); del o
            valid = torch.ones_like(seqs, dtype=torch.bool)
            for r in range(seqs.shape[0]):
                e = (seqs[r, ids.shape[1]:] == eos).nonzero()
                if len(e): valid[r, ids.shape[1] + int(e[0]):] = False
            nxt = (seqs[:, 1:] == tid) & valid[:, 1:]; hn_.append(H[:, :-1][nxt]); valid[:, :3] = False; hp_.append(H[valid])
        hn_ = torch.cat(hn_); cnt[w] = int(hn_.shape[0]); mu_n[w] = hn_.mean(0).cpu(); mu_p[w] = torch.cat(hp_).mean(0).cpu()
    if a.dm_cache and not os.path.exists(a.dm_cache): torch.save(dict(mu_n=mu_n, mu_p=mu_p, cnt=cnt, samples=a.dm_samples, tokens=a.dm_tokens, prompts=DM_PROMPTS), a.dm_cache)
    for it in items: it["_dm_next"] = mu_n[it["tgt"]] - mu_n[it["src"]]; it["_dm_pass"] = mu_p[it["tgt"]] - mu_p[it["src"]]
    print(f"[A] DiffMean over {len(words)} concepts; next-token positions per concept min {min(cnt.values())} median {int(np.median(list(cnt.values())))}", flush=True)
MULTI = 'The next word will be "{w}". The text is about a {w}. The continuation will keep talking about the {w}.'


def donor_text(text, src, tgt):
    """the prompt with the source concept replaced by the target (whole words, plural + capitalisation kept, a/an fixed); None if absent"""
    import re
    forms = sorted({src + "es", src + "s", src} | set(A.PLURAL.get(src, [])), key=len, reverse=True); out = text
    for f in forms:
        rep = lambda m, f=f: (tgt + ("s" if f != src else ""))[0].upper() + (tgt + ("s" if f != src else ""))[1:] if m.group(0)[0].isupper() else tgt + ("s" if f != src else "")
        out = re.sub(rf"\b{re.escape(f)}\b", rep, out, flags=re.I)
    if out == text: return None
    out = re.sub(rf"\b([Aa])n? ({re.escape(tgt)})", lambda m: m.group(1) + ("n" if tgt[0].lower() in "aeiou" else "") + " " + m.group(2), out)
    return out


def build_types(a, items, types):
    """condition pairs (z{T}, z{T}e) per text type: A/B from the cache; C add one claim; M claims only; R/P Opus rewrites; E the verbalizer's
    own explanation of the donor context (prompt with the concept swapped), whose layer-42 activation is kept as the donor reference."""
    if any(t in types for t in "RP"):
        rw = json.load(open(a.rewrites))["rewrites"]; bad = [it["n"] for it in items if rw[str(it["n"])]["zA"] != it["zA"]]
        assert not bad, f"rewrites were made from different explanations for prompts {bad}"
    for it in items:
        if "C" in types: it["zC"], it["zCe"] = it["zA"], it["zA"].rstrip() + "\n" + CLAIM.format(w=it["tgt"])
        if "M" in types: it["zM"], it["zMe"] = MULTI.format(w=it["src"]), MULTI.format(w=it["tgt"])
        if "S" in types: it["zS"], it["zSe"] = CLAIM_N.format(w=it["src"]), CLAIM_N.format(w=it["tgt"])
        if "R" in types: it["zR"], it["zRe"] = it["zA"], rw[str(it["n"])]["R"]["text"]
        if "P" in types: it["zP"], it["zPe"] = it["zA"], rw[str(it["n"])]["P"]["text"]
    if "E" not in types and not a.donor: return
    tok, lm, st = pg.S["tok"], pg.S["lm"], pg.S["st"]; hs, idx = [], []
    for i, it in enumerate(items):
        dt = donor_text(it["text"], it["src"], it["tgt"]); it["donor_text"] = dt
        if dt is None: it["no_donor"] = True; it["zE"], it["zEe"] = it["zA"], it["zA"]; continue
        ids = tok(dt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEV0)
        with pg.S["gpu"], pg._base():
            st.update(cap=None, vec=None, decode_fn=None, prefill_fn=None)
            with torch.no_grad(): o = lm(input_ids=ids, output_hidden_states=True)
        hd_all = o.hidden_states[43][0].float(); hd = hd_all[-1]; pr = torch.softmax(o.logits[0, -1].float(), -1); del o
        it["_hd"] = hd.cpu(); it["donor_p_tgt"] = float(pr[R.tid(tok, it["tgt"])]); it["donor_top1_tgt"] = bool(int(pr.argmax()) == R.tid(tok, it["tgt"])); hs.append(hd); idx.append(i)
        m_ = mention_index(it["_ids"][0].tolist(), R.tid(tok, it["src"]))
        if m_ is not None and ids.shape[1] == it["_ids"].shape[1]: it["_hd_m"] = hd_all[m_].cpu()
    zs = pg._verbalize(torch.stack(hs), R.AV_WARM, temperature=0.0, max_new=200)
    for i, z in zip(idx, zs): items[i]["zE"], items[i]["zEe"] = items[i]["zA"], z
    print(f"[A] donor contexts for {len(idx)}/{len(items)} prompts; median donor p(target) {float(np.median([items[i]['donor_p_tgt'] for i in idx])):.2f}; e.g. {items[idx[0]]['donor_text']!r} -> {items[idx[0]]['zEe'][:160]!r}", flush=True)


def family(nm):
    """'A|trunk_dn64|dlt_t0.3_b1' -> 'trunk_dn64_dlt_t0.3_A';  references keep the v2 families."""
    if "@" in nm and "|" not in nm: return nm.rsplit("_b", 1)[0] if "_b" in nm else nm
    if nm.startswith(("dmn_", "dmp_")): return nm.split("_")[0]
    if nm.startswith("donor"): return "donor_act"
    if "|" not in nm: return _FAM0(nm)
    T_, mod, meth = nm.split("|"); core = meth.rsplit("_b", 1)[0] if "_b" in meth and not meth.startswith("run_") else meth
    return f"{mod}_{core}_{T_}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="delta_v1"); p.add_argument("--models", default="sw_tokar,trunk_dn64,uctext,ucediff")
    p.add_argument("--decoder", default="/vol_glp/unclip/decoder/dec_main/snap_000262M"); p.add_argument("--encoder-json", default="/vol_glp/unclip/encoder.json")
    p.add_argument("--cache", default=f"{OUT}/cache_next.pt"); p.add_argument("--out", default=OUT); p.add_argument("--n", type=int, default=48)
    p.add_argument("--k", type=int, default=3); p.add_argument("--n-new", type=int, default=40); p.add_argument("--types", default="A,B")
    p.add_argument("--ts", default="0.1,0.3,0.6"); p.add_argument("--betas", default="0.5,1,2"); p.add_argument("--taus", default="0.2,0.5"); p.add_argument("--run-betas", default="1,2")
    p.add_argument("--it-betas", default="1,2"); p.add_argument("--it-steps", type=int, default=8); p.add_argument("--it-t", type=float, default=0.2)
    p.add_argument("--ediff-alpha", type=float, default=4.0); p.add_argument("--gen-chunk", type=int, default=48); p.add_argument("--smoke", action="store_true")
    p.add_argument("--delta-only", action="store_true", help="skip the iterative and ODE-run edits")
    p.add_argument("--extra-critics", default="", help="comma list name=adapter_path of more token-conditioned flow critics (usable in --models)")
    p.add_argument("--rewrites", default=f"{OUT}/steer_rewrites.json", help="Opus rewrites (scripts/steer_rewrites.py) for text types R / P")
    p.add_argument("--no-avg", action="store_true"); p.add_argument("--no-run-literal", action="store_true")
    p.add_argument("--diffmean", action="store_true"); p.add_argument("--dm-samples", type=int, default=16); p.add_argument("--dm-tokens", type=int, default=64)
    p.add_argument("--dm-cache", default="", help="torch file with per-concept DiffMean means: loaded if present, else computed (seeded) and saved")
    p.add_argument("--ref-betas", default="0.25,0.5,1,2", help="strengths for the J-lens / DiffMean / donor reference directions")
    p.add_argument("--sites", action="store_true", help="also apply direction edits at the source-mention position (@m) and at mention + anchor (@mb)")
    p.add_argument("--site-dirs", default="", help="comma list of edit-direction keys (e.g. S|c1p3b|dlt_t0.1,M|ar|dir) also tested at the mention sites")
    p.add_argument("--site-betas", default="1,2"); p.add_argument("--donor", action="store_true", help="donor activations (patch references) without the E text type")
    a = p.parse_args(); fl = lambda s_: [float(x) for x in s_.split(",") if x]
    a.ts, a.betas, a.taus, a.run_betas, a.it_betas = fl(a.ts), fl(a.betas), fl(a.taus), fl(a.run_betas), fl(a.it_betas); types = [x for x in a.types.split(",") if x]
    mods = [x for x in a.models.split(",") if x]
    for c in [c for c in a.extra_critics.split(",") if c]: nm_, _, pth = c.partition("="); V2.CRIT[nm_] = pth
    if a.smoke: a.n, a.k, a.n_new, a.it_steps = 3, 1, 8, 2
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    pg._load(); lens = R.Lens(); tok = pg.S["tok"]
    items = torch.load(a.cache, map_location="cpu", weights_only=False)[: a.n]; N = len(items); print(f"[A] {N} prompts from {a.cache}", flush=True)
    build_types(a, items, types)
    if a.diffmean: build_diffmean(a, items)
    H1 = torch.stack([it["_h"] for it in items]).to(DEV1).float()
    V, D, mu0 = {}, {}, None
    # ---- flow critics
    for cname in [m_ for m_ in mods if m_ in V2.CRIT]:
        t1 = time.time(); fb = R.load_flow(V2.CRIT[cname]); fb.model.eval(); m = FlowDen(fb)
        print(f"[B] {cname}: cond_mode {fb.cond_mode} norm {type(fb.norm).__name__} loaded {time.time() - t1:.0f}s", flush=True)
        for T_ in types:
            c0 = m.conds([it[f"z{T_}"] for it in items]); c1 = m.conds([it[f"z{T_}e"] for it in items])
            v_, d_ = model_edits(m, H1, c0, c1, a, f"{T_}|{cname}|"); V.update(v_); D.update(d_)
        del fb, m; torch.cuda.empty_cache(); print(f"[B] {cname} edits in {time.time() - t1:.0f}s", flush=True)
    # ---- MSE reconstructor direction AR(z') - AR(z) (independent of h, like its v2 rows)
    if "ar" in mods:
        t1 = time.time(); critic = pg._critic(R.AR_NAME)
        for T_ in types:
            dl = torch.stack([(pg.ar_pred(critic, it[f"z{T_}e"]) - pg.ar_pred(critic, it[f"z{T_}"])).float().cpu() for it in items]).to(DEV1); D[f"{T_}|ar|dir"] = dl
            for b in a.betas: V[f"{T_}|ar|dir_b{b:g}"] = R.resc(H1, dl, b)
        del critic; torch.cuda.empty_cache(); print(f"[B] AR directions in {time.time() - t1:.0f}s", flush=True)
    # ---- unCLIP decoder
    uc = [m_ for m_ in mods if m_ in ("uctext", "ucediff")]
    if uc:
        t1 = time.time(); enc = Encoder(a.encoder_json, pg.S["snap"], DEV1); dec = Decoder(a.decoder, DEV1, base=pg.S["snap"]); m = DecDen(dec); mu0 = dec.mu.to(DEV0)
        E = enc.f(H1)
        for T_ in types:
            Gz, Gze = enc.g([it[f"z{T_}"] for it in items]), enc.g([it[f"z{T_}e"] for it in items])
            if "uctext" in uc: v_, d_ = model_edits(m, H1, Gz, Gze, a, f"{T_}|uctext|"); V.update(v_); D.update(d_)
            if "ucediff" in uc: v_, d_ = model_edits(m, H1, E, enc.renorm(E + a.ediff_alpha * (Gze - Gz), E), a, f"{T_}|ucediff|"); V.update(v_); D.update(d_)
        del enc, dec, m; torch.cuda.empty_cache(); print(f"[B] unCLIP edits in {time.time() - t1:.0f}s", flush=True)
    print(f"[B] {len(V)} edited conditions ({time.time() - t0:.0f}s)", flush=True)
    # ---- measurement (same metrics as unclip_steer_v2.py)
    rows = []; betas_ref = tuple(float(x) for x in a.ref_betas.split(","))
    for i, it in enumerate(items):
        h = it["_h"].to(DEV0).float(); ids = it["_ids"].to(DEV0); T = ids.shape[1]; src, tgt = it["src"], it["tgt"]; ts_, tt_ = R.tid(tok, src), R.tid(tok, tgt); hn = h.norm()
        Vl = torch.stack([lens.vec(ts_), lens.vec(tt_)], 1); d_jl = Vl[:, 1] - Vl[:, 0]
        g_ = torch.Generator(device=DEV0).manual_seed(99 + it["n"]); rnd = torch.randn(h.shape, device=DEV0, generator=g_)
        raw = {"none": h}
        for b in betas_ref: raw[f"jadd_b{b:g}"] = R.resc(h, d_jl, b)
        for b in (1.0, 2.0): raw[f"rand_b{b:g}"] = R.resc(h, rnd, b)
        if it.get("_hd") is not None:
            hd = it["_hd"].to(DEV0).float(); raw["donor_full"] = hd
            for b in betas_ref: raw[f"donor_b{b:g}"] = R.resc(h, hd - h, b)
        if "_dm_next" in it:
            for b in betas_ref: raw[f"dmn_b{b:g}"] = R.resc(h, it["_dm_next"].to(DEV0).float(), b); raw[f"dmp_b{b:g}"] = R.resc(h, it["_dm_pass"].to(DEV0).float(), b)
        for k_, v_ in V.items():
            if k_.startswith("E|") and it.get("no_donor"): continue
            raw[k_] = v_[i].to(DEV0).float()
        one = {k_: (v_ / v_.norm().clamp_min(1e-6) * hn if k_ != "none" else v_) for k_, v_ in raw.items()}
        m = mention_index(ids[0].tolist(), ts_) if a.sites else None
        pos = [T - 1] if m is None else [m, T - 1]; ai = len(pos) - 1
        specs = {nm: ({} if nm == "none" else {ai: v}) for nm, v in one.items()}
        if m is not None:
            with pg.S["gpu"], pg._base():
                pg.S["st"].update(cap=None, vec=None, decode_fn=None, prefill_fn=None)
                with torch.no_grad(): o_ = pg.S["lm"](input_ids=ids, output_hidden_states=True)
            hm = o_.hidden_states[43][0, m].float(); del o_; hmn = hm.norm(); sdirs = {"jadd": d_jl}
            if "_dm_next" in it: sdirs["dmn"] = it["_dm_next"].to(DEV0).float(); sdirs["dmp"] = it["_dm_pass"].to(DEV0).float()
            for k_ in [k for k in a.site_dirs.split(",") if k and k in D]: sdirs[k_] = D[k_][i].to(DEV0).float()
            for sk, dd in sdirs.items():
                for b in [float(x) for x in a.site_betas.split(",")]:
                    vm = R.resc(hm, dd, b); vm = vm / vm.norm() * hmn; va = R.resc(h, dd, b); va = va / va.norm() * hn
                    specs[f"{sk}@m_b{b:g}"] = {0: vm}; specs[f"{sk}@mb_b{b:g}"] = {0: vm, ai: va}; one[f"{sk}@m_b{b:g}"] = h; one[f"{sk}@mb_b{b:g}"] = va
            if it.get("_hd_m") is not None:
                hdm = it["_hd_m"].to(DEV0).float(); specs["donor@m"] = {0: hdm}; specs["donor@mb"] = {0: hdm, ai: it["_hd"].to(DEV0).float()}; one["donor@m"] = h; one["donor@mb"] = it["_hd"].to(DEV0).float()
        names = list(one); Eall = torch.stack([one[k_] for k_ in names]).float(); lg = lens.logits(Eall, 42); hc = h - (mu0 if mu0 is not None else 0)
        out = {}
        for j, nm in enumerate(names):
            out[nm] = dict(jl_tgt_rank=int(R.Lens.rank(lg[j], tt_)), jl_src_rank=int(R.Lens.rank(lg[j], ts_)), edit_rel=float((Eall[j] - h).norm() / hn), edit_rel_raw=float((raw.get(nm, Eall[j]) - h).norm() / hn), site="mb" if "@mb" in nm else "m" if "@m" in nm else "anchor",
                           cos_raw=float(F.cosine_similarity(Eall[j], h, dim=0)), cos_c=float(F.cosine_similarity(Eall[j] - (mu0 if mu0 is not None else 0), hc, dim=0)))
        seqs_by = {}; lg1 = {}
        for c0_ in range(0, len(names), a.gen_chunk):
            nms = names[c0_:c0_ + a.gen_chunk]
            for greedy, KK in ((True, 1), (False, a.k)):
                gg, sl = run_batch_sites(ids, nms, specs, pos, KK, greedy, a.n_new); seqs = gg.sequences
                for nm, s_ in sl.items():
                    out[nm].setdefault("conts", []).extend(A.cont_of(tok, seqs[j], T) for j in range(s_.start, s_.stop)); seqs_by.setdefault(nm, []).extend(seqs[j].cpu() for j in range(s_.start, s_.stop))
                    if greedy: lg1[nm] = gg.scores[0][s_.start].float()
        ref = lg1["none"]
        for nm, r in out.items():
            r.update(V2.next_metrics(lg1[nm], ts_, tt_, ref)); r["tgt"] = [A.ment(c, tgt) for c in r["conts"]]; r["src"] = [A.ment(c, src) for c in r["conts"]]
        allseqs = [s_ for nm in names for s_ in seqs_by[nm]]; nll = A.nll_cont(ids, allseqs, T); q = 0
        for nm in names: out[nm]["nll"] = nll[q:q + len(seqs_by[nm])]; q += len(seqs_by[nm])
        dcos = {k_: float(F.cosine_similarity(d_[i].to(DEV0).float(), d_jl, dim=0)) for k_, d_ in D.items() if not (k_.startswith("E|") and it.get("no_donor"))}   # direction vs the J-lens direction
        rows.append(dict(n=it["n"], kind=it["kind"], text=it["text"], src=src, tgt=tgt, zA=it["zA"], zAe=it["zAe"], zB=it["zB"], zBe=it["zBe"], base_p_src=it["base_p_src"], base_p_tgt=it["base_p_tgt"],
                         dir_cos_jlens=dcos, donor_text=it.get("donor_text"), donor_p_tgt=it.get("donor_p_tgt"), donor_top1_tgt=it.get("donor_top1_tgt"), mention_pos=m, zE=it.get("zEe"), zR=it.get("zRe"), zP=it.get("zPe"), conds=out))
        pick = [c for c in ["jadd_b2", "dmn_b2", "dmp_b2", "jadd@m_b2", "dmn@m_b2", "donor@m", "S|c1p3b|dlt_t0.1_b2", "S|c1ctr|dlt_t0.1_b2", "M|ar|dir_b2"] if c in out]
        print(f"[C] {it['n']} {src}->{tgt} ({time.time() - t0:.0f}s): " + " | ".join(f"{c} top1 {int(out[c]['top1_tgt'])} p_t {out[c]['p_tgt']:.2f} kl {out[c]['kl1']:.2f}" for c in pick), flush=True)
        if (i + 1) % 8 == 0: json.dump(dict(partial=True, rows=rows), open(f"{a.out}/{a.tag}.partial.json", "w"))
    # ---- summaries
    V2.family_of = family                  # summarize() looks family_of up in its module
    keys = list(dict.fromkeys(k for r in rows for k in r["conds"])); full = V2.summarize(rows, keys)
    fams = {}
    for nm in keys: fams.setdefault(family(nm), []).append(nm)
    budgets = [0.5, 1.0, 2.0, 4.0, 8.0]; matched = {}
    groups = dict(fams)
    for nm in keys:                                                            # coarse groups: every delta method of one model and text type
        if "|" in nm: T_, mod, _ = nm.split("|"); groups.setdefault(f"ALL_{mod}_{T_}", []).append(nm)
    for fam, ks in groups.items():
        matched[fam] = {}
        for B in budgets:
            ok = [k for k in ks if full[k]["kl1_median"] <= B]
            if ok:
                best = max(ok, key=lambda k: full[k]["flip_rate"]); matched[fam][str(B)] = dict(cond=best, flip_rate=full[best]["flip_rate"], tgt_mention=full[best]["tgt_mention"], p_tgt_mean=full[best]["p_tgt_mean"], kl1_median=full[best]["kl1_median"])
    dcm = {k_: float(np.mean([r["dir_cos_jlens"][k_] for r in rows if k_ in r["dir_cos_jlens"]])) for k_ in dict.fromkeys(k for r in rows for k in r["dir_cos_jlens"])}
    res = dict(tag=a.tag, models=mods, decoder=a.decoder, n=N, k=a.k, ts=a.ts, betas=a.betas, taus=a.taus, it=dict(steps=a.it_steps, t=a.it_t, betas=a.it_betas), ediff_alpha=a.ediff_alpha,
               t_avg=T_AVG, families=fams, summary=full, matched_kl=matched, dir_cos_jlens_mean=dcm, rows=rows)
    json.dump(res, open(f"{a.out}/{a.tag}.json", "w"))
    for k_, v_ in dcm.items(): print(f"[dir] cos(direction, J-lens direction) {k_:28s} {v_:+.3f}", flush=True)
    for nm in keys:
        s = full[nm]; print(f"[summary] {nm:34s} flip {s['flip_rate']:.2f} p_tgt {s['p_tgt_mean']:.3f} kl1 {s['kl1_median']:.2f} nll {s['nll_median'] if s['nll_median'] is None else round(s['nll_median'], 2)} | mention tgt {s['tgt_mention']:.2f} src {s['src_mention']:.2f}", flush=True)
    for fam, m_ in matched.items():
        if fam.startswith("ALL_") or fam in ("jlens_add", "random"): print(f"[matched] {fam:22s} " + " ".join(f"KL<={B}: flip {m_[str(B)]['flip_rate']:.2f} ({m_[str(B)]['cond']})" for B in budgets if str(B) in m_), flush=True)
    print(f"[done] {N} prompts in {time.time() - t0:.0f}s -> {a.out}/{a.tag}.json", flush=True)


if __name__ == "__main__":
    main()
