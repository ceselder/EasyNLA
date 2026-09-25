"""unCLIP steering v2: a NEXT-TOKEN concept-swap eval where the J-lens control has headroom, with strength sweeps for every method and a
success-vs-KL frontier (Qwen3.6-27B, layer-42 anchor activation h = the last prompt token).

Prompt set: ~48 prompts (animals + objects) whose IMMEDIATE next token is the source concept (single Qwen3.6 token, e.g. "Every morning,
my" -> " dog"), built in-process and filtered by the base model's rank of the source at the next position (--max-src-rank). Every edit
replaces h at the anchor (direction only: every edited vector is rescaled to ||h||) and is measured at the NEXT position: target-token
probability and rank, FLIP (p(target) > p(source)), source-probability drop, KL to the unedited next-token distribution (KL1), plus the
40-token continuation (target / source mention, greedy + k samples) and its NLL under the unedited model (fluency), J-lens rank of the target
on the edited h, centred cosine cos(h'-mu, h-mu), and encoder-space cos(f(h'), g(z')) / g(z).

Two text-edit types for every text-conditioned method:  A = the warm verbalizer's full explanation of h with the concept word swapped
(z -> z');  B = a short concept-centred pair ("The model is about to name a {src}; the next words are about the {src}." -> {tgt}).
Per type the embedding movement is reported: pooled cos(g(z), g(z')) and prior-read cos(m(z), m(z')) (m = mean of K prior samples).

Methods (strength sweeps; the J-lens direction is the reference frontier):
  jadd_b{β}               h + β||h|| unit(v_tgt - v_src), v = J-lens token vectors at layer 42        jswap_a{α}   J-lens coordinate swap (Lens paper §2.5)
  rand_b{β}               random direction control
  {A,B}tdiff_a{α}         unCLIP pooled text diff: decode(invert(h|e) | renorm(e + α (g(z') - g(z)))), CFG 2 on t in --cfg-window
  {A,B}pdiff_a{α}         unCLIP prior-read diff: e' = renorm(e + α (m(z') - m(z))), CFG 2 windowed
  {A,B}prior_cfg{s}       unCLIP prior-sampled target e' ~ p(e | z') decoded from h's noise, CFG s windowed (s = 1: none)
  {A,B}gtext              decode under g(z') itself (α -> ∞ end of the pooled ladder), CFG 2 windowed
  {A,B}<critic>_b{β}      flow critics' conditional-mean direction E[h|z'] - E[h|z] (one-step x0 at t = 1, K draws) at β||h||   (sw_tokar, trunk_dn64)
  {A,B}ar_b{β}            MSE reconstructor's direction AR(z') - AR(z) at β||h||
  recon, var_1            unCLIP round trip and a variation (controls)
Reporting: full set and the SUBSET where the J-lens flips the next token at fluency-preserving strength (KL1 <= tau, tau = median KL1 of the
weakest jadd β reaching >= 50 % flips); per-family frontier (flip / mention / p_tgt vs median KL1 and vs continuation NLL); matched-KL table
(best flip rate per family under KL budgets).  usage: python scripts/unclip_steer_v2.py --decoder <snap> --prior <snap> --tag T
   -> /vol_glp/unclip/steer/<tag>.json  (plots: scripts/unclip_steer_v2_plot.py)"""
import argparse, json, os, re, sys, time
import numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import rhyme_plan_steer as R
import animal_swap_steer as A
from nla.unclip.steer_models import Encoder, Decoder, load_prior, prior_sample
pg = R.pg; DEV0, DEV1 = R.DEV0, R.DEV1; OUT = "/vol_glp/unclip/steer"
CRIT = {"sw_tokar": "/vol_glp/cond/sw_tokar/adapter_latest.pt", "trunk_dn64": "/vol_glp/cond/trunk_dn64/adapter_latest.pt"}
TEMPL_B = "The model is about to name a {w}; the next words are about the {w}."

# (text ending right before the concept token, source, target of another kind, kind)
CANDIDATES = [
    ("My dog loves chasing the ball in the park. Every morning, my", "dog", "shark", "animal"), ("The cat curled up on the windowsill. Later that night, the", "cat", "frog", "animal"),
    ("The elephant sprayed water over its back with its trunk. The keeper said the", "elephant", "mouse", "animal"), ("Woof woof! barked the", "dog", "eagle", "animal"),
    ("The horse galloped across the field. When it reached the fence, the", "horse", "spider", "animal"), ("The farmer milked the cow at dawn. By noon the", "cow", "dolphin", "animal"),
    ("In the savanna the lion stalked a zebra. Suddenly the", "lion", "owl", "animal"), ("The rabbit hopped into the garden. Then the", "rabbit", "shark", "animal"),
    ("Deep in the ocean the whale sang. Hours later the", "whale", "monkey", "animal"), ("The snake slithered across the rock. Then the", "snake", "elephant", "animal"),
    ("A frog sat on a lily pad. Then the", "frog", "bear", "animal"), ("The owl perched on the branch. At midnight the", "owl", "pig", "animal"),
    ("The mouse crept out of its hole. Then the", "mouse", "wolf", "animal"), ("The shark circled the boat. Then the", "shark", "sheep", "animal"),
    ("A bee buzzed between the flowers. Then the", "bee", "tiger", "animal"), ("The duck waddled to the pond. Then the", "duck", "camel", "animal"),
    ("The monkey swung through the trees. Then the", "monkey", "turtle", "animal"), ("The fox crept into the henhouse. Then the", "fox", "whale", "animal"),
    ("It purred and rubbed against my leg. What a lovely", "cat", "snake", "animal"), ("It barked at the mailman and wagged its tail. What a good", "dog", "eagle", "animal"),
    ("With its long trunk and big floppy ears, the", "elephant", "mouse", "animal"), ("It spun a web in the corner of the room. The", "spider", "dog", "animal"),
    ("Quack quack, said the", "duck", "camel", "animal"), ("Moo, said the", "cow", "dolphin", "animal"), ("Meow, said the", "cat", "frog", "animal"), ("Neigh! said the", "horse", "spider", "animal"),
    ("The zoo's new tiger paced in its enclosure. Visitors crowded to see the", "tiger", "penguin", "animal"), ("The bear hibernated all winter. In spring the", "bear", "shark", "animal"),
    ("The penguin waddled across the ice. Then the", "penguin", "lion", "animal"), ("A wolf howled at the moon. Then the", "wolf", "duck", "animal"),
    ("She parked the car in the driveway. The next morning the", "car", "boat", "object"), ("He tuned his guitar before the show. On stage he played the", "guitar", "piano", "object"),
    ("Every morning I brew a pot of coffee. I can't start the day without my", "coffee", "tea", "object"), ("We ordered a large pizza. When it arrived, the", "pizza", "salad", "object"),
    ("She opened the book and read a chapter. Then she closed the", "book", "movie", "object"), ("It rained all afternoon. By evening the", "rain", "snow", "object"),
    ("The train pulled into the station. Passengers boarded the", "train", "plane", "object"), ("He sliced the bread and buttered it. Then he ate the", "bread", "cheese", "object"),
    ("The knight drew his sword. He raised the", "sword", "gun", "object"), ("The castle stood on the hill. Tourists visited the", "castle", "church", "object"),
    ("The river flowed through the valley. We swam in the", "river", "desert", "object"), ("She poured a glass of wine. She sipped the", "wine", "beer", "object"),
    ("The doctor examined the patient. Then the", "doctor", "lawyer", "object"), ("He lit a candle. The flame of the", "candle", "lamp", "object"),
    ("The moon rose over the hills. In the light of the", "moon", "sun", "object"), ("He picked up his phone and unlocked it. He scrolled through his", "phone", "camera", "object"),
    ("The plane took off from the runway. An hour later the", "plane", "train", "object"), ("She put on her shoes and tied the laces. Then she took off her", "shoes", "hat", "object"),
    ("He put the key in the lock and turned the", "key", "coin", "object"), ("He opened the door and stepped through the", "door", "window", "object"),
    ("She turned on the computer and waited for the", "computer", "radio", "object"), ("The soldier loaded his gun. He aimed the", "gun", "sword", "object"),
    ("The chef chopped the onion. Then he fried the", "onion", "potato", "object"), ("He brewed a pot of tea and poured a cup of", "tea", "coffee", "object"),
    ("The boat drifted across the lake. The fisherman rowed the", "boat", "car", "object"), ("The bus stopped at the corner. We got on the", "bus", "train", "object"),
    ("She baked a cake for his birthday. Everyone loved the", "cake", "pizza", "object"), ("The clock on the wall struck noon. Everyone looked at the", "clock", "mirror", "object"),
    ("He strummed a chord on his guitar. Then he put down the", "guitar", "piano", "object"), ("The pizza was hot and cheesy. Everyone grabbed a slice of", "pizza", "salad", "object"),
]


# ------------------------------------------------------------------ stage A: prompt set with next-token targets
def build_cache(a, lens):
    tok, lm, st = pg.S["tok"], pg.S["lm"], pg.S["st"]; items = []; t0 = time.time(); dropped = []
    for n, (text, src, tgt, kind) in enumerate(CANDIDATES):
        if not (R.single(tok, src) and R.single(tok, tgt)): dropped.append((src, tgt, "multi-token")); continue
        ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEV0); T = ids.shape[1]
        with pg.S["gpu"], pg._base():
            st.update(cap=None, vec=None, decode_fn=None, prefill_fn=None)
            with torch.no_grad(): o = lm(input_ids=ids, output_hidden_states=True)
        lg = o.logits[0, -1].float(); h = o.hidden_states[43][0, -1].float(); ts_, tt_ = R.tid(tok, src), R.tid(tok, tgt); del o
        pr = torch.softmax(lg, -1); sr, tr = int(R.Lens.rank(lg, ts_)), int(R.Lens.rank(lg, tt_))
        if sr > a.max_src_rank: dropped.append((src, tgt, f"src rank {sr}, top: {tok.decode([int(lg.argmax())])!r}")); continue
        jl = lens.logits(h[None], 42)[0]
        items.append(dict(n=len(items), cand=n, text=text, src=src, tgt=tgt, kind=kind, T=T, base_p_src=float(pr[ts_]), base_p_tgt=float(pr[tt_]), base_rank_src=sr, base_rank_tgt=tr,
                          base_top5=[tok.decode([int(x)]) for x in lg.topk(5).indices], jl42_src_rank=int(R.Lens.rank(jl, ts_)), jl42_tgt_rank=int(R.Lens.rank(jl, tt_)),
                          _h=h.cpu(), _ids=ids.cpu(), _lg=lg.cpu()))
    # balance kinds, cap at --n
    per = {}; keep = []
    for it in items: per.setdefault(it["kind"], []).append(it)
    for kind in sorted(per): keep += per[kind][: a.n // 2]
    items = sorted(keep, key=lambda it: it["cand"])
    for i, it in enumerate(items): it["n"] = i
    print(f"[A] {len(items)} prompts kept ({', '.join(f'{k} {len(per[k])} found' for k in sorted(per))}), {len(dropped)} dropped: {dropped[:12]}", flush=True)
    H = torch.stack([it["_h"] for it in items]); vg = pg._verbalize(H, R.AV_WARM, temperature=0.0, max_new=200)
    for it, z0 in zip(items, vg):
        src, tgt = it["src"], it["tgt"]
        if A.ment(z0, src): z = z0; templ = False
        else: z = z0.rstrip() + f" The text is about a {src}; the continuation will keep talking about the {src}."; templ = True
        ze = z
        for f in sorted({src + "es", src + "s", src} | set(A.PLURAL.get(src, [])), key=len, reverse=True): ze = re.sub(rf"\b{re.escape(f)}\b", tgt + ("s" if f != src else ""), ze, flags=re.I)
        it.update(zA=z, zAe=ze, zA_templated=templ, zB=TEMPL_B.format(w=src), zBe=TEMPL_B.format(w=tgt))
        print(f"[A] {it['n']} {it['kind']} '{src}'->'{tgt}': next-token p(src) {it['base_p_src']:.2f} rank {it['base_rank_src']} | p(tgt) {it['base_p_tgt']:.4f} rank {it['base_rank_tgt']} | top5 {it['base_top5']} | verbalizer names src {not templ}", flush=True)
    print(f"[A] done {time.time() - t0:.0f}s", flush=True); return items


# ------------------------------------------------------------------ stage B: edits
@torch.no_grad()
def unclip_edits(a, items, enc, dec, prior):
    """-> V: name -> [N, d] raw (NOT yet norm-matched), C: name -> condition used, pd: per-type embedding diagnostics [N]"""
    N = len(items); H = torch.stack([it["_h"] for it in items]).to(dec.dev).float(); S = a.ode_steps; WIN = tuple(a.cfg_window); V, C, pd = {}, {}, {}
    E = enc.f(H); eps = dec.invert(H, E, S); V["recon"] = dec.decode(eps, E, S, 1.0); C["recon"] = E
    V["var_1"] = dec.decode(torch.randn(eps.shape, device=dec.dev, generator=torch.Generator(device=dec.dev).manual_seed(1234)), E, S, 1.0); C["var_1"] = E
    print(f"[B] unCLIP inversion: noise norm/sqrt(d) {float(eps.norm(dim=-1).mean() / eps.shape[1] ** 0.5):.3f}, recon rel err {float(((V['recon'] - H).norm(dim=-1) / H.norm(dim=-1)).median()):.4f}", flush=True)
    for T_ in ("A", "B"):
        zs, zes = [it[f"z{T_}"] for it in items], [it[f"z{T_}e"] for it in items]; Gz, Gze = enc.g(zs), enc.g(zes)
        pd[f"{T_}_cos_gz_gze"] = Encoder.cos(Gz, Gze).cpu(); pd[f"{T_}_cos_e_gz"] = Encoder.cos(E, Gz).cpu(); pd[f"{T_}_cos_e_gze"] = Encoder.cos(E, Gze).cpu()
        for al in a.alphas: Ep = enc.renorm(E + al * (Gze - Gz), E); V[f"{T_}tdiff_a{al:g}"] = dec.decode(eps, Ep, S, 2.0, WIN); C[f"{T_}tdiff_a{al:g}"] = Ep
        V[f"{T_}gtext"] = dec.decode(eps, Gze, S, 2.0, WIN); C[f"{T_}gtext"] = Gze
        if prior is not None:
            try:
                Mz = enc.project(prior_sample(prior, zs, n=a.K, seed=3).mean(1)); Mze = enc.project(prior_sample(prior, zes, n=a.K, seed=3).mean(1)); P1 = enc.project(prior_sample(prior, zes, n=1, seed=11)[:, 0])
                pd[f"{T_}_cos_mz_mze"] = Encoder.cos(Mz, Mze).cpu(); pd[f"{T_}_cos_e_mz"] = Encoder.cos(E, Mz).cpu(); pd[f"{T_}_cos_e_p1"] = Encoder.cos(E, P1).cpu(); pd[f"{T_}_cos_mze_gze"] = Encoder.cos(Mze, Gze).cpu()
                for al in a.alphas: Ep = enc.renorm(E + al * (Mze - Mz), E); V[f"{T_}pdiff_a{al:g}"] = dec.decode(eps, Ep, S, 2.0, WIN); C[f"{T_}pdiff_a{al:g}"] = Ep
                for s in a.cfgs: V[f"{T_}prior_cfg{s:g}"] = dec.decode(eps, P1, S, s, WIN if s != 1 else None); C[f"{T_}prior_cfg{s:g}"] = P1
            except Exception as ex:
                import traceback; traceback.print_exc(); print(f"[B] prior sampling failed for type {T_}: {ex}", flush=True)
        print(f"[B] type {T_}: cos(g(z), g(z')) {float(pd[f'{T_}_cos_gz_gze'].mean()):.3f}" + (f" | prior-read cos(m(z), m(z')) {float(pd[f'{T_}_cos_mz_mze'].mean()):.3f} cos(e, m(z)) {float(pd[f'{T_}_cos_e_mz'].mean()):.3f} cos(e, e'~p(e|z')) {float(pd[f'{T_}_cos_e_p1'].mean()):.3f}" if f"{T_}_cos_mz_mze" in pd else ""), flush=True)
    return V, C, pd


@torch.no_grad()
def flow_cmean_dirs(fb, items, K, seed=7):
    """conditional-mean directions E[h|z'] - E[h|z] (one-step x0 prediction from pure noise at t = 1, K draws) for both text types; raw units, per prompt."""
    out = {"A": [], "B": []}; d = fb.norm.mean.shape[0] if hasattr(fb.norm, "mean") else 5120
    for it in items:
        g_ = torch.Generator(device=DEV1).manual_seed(seed + it["n"]); E_ = torch.randn(K, d, device=DEV1, generator=g_); mu = {}
        for T_ in ("A", "B"):
            for key in (f"z{T_}", f"z{T_}e"):
                enc, mk, cv = fb.cond([it[key]]); rep = lambda x: None if x is None else x.expand(K, *x.shape[1:]) if x.shape[0] == 1 else x.repeat_interleave(K, 0)
                with torch.autocast("cuda", dtype=torch.bfloat16): v = fb.model(E_, torch.ones(K, device=DEV1), rep(enc), rep(mk), rep(cv)).float()
                mu[key] = fb.norm.denormalize((E_ - v).mean(0, keepdim=True))[0]
            out[T_].append((mu[f"z{T_}e"] - mu[f"z{T_}"]).cpu())
    return out


# ------------------------------------------------------------------ stage C: measurement
def next_metrics(lg, ts_, tt_, ref_lg):
    pr = torch.softmax(lg, -1)
    return dict(p_tgt=float(pr[tt_]), p_src=float(pr[ts_]), rank_tgt=int(R.Lens.rank(lg, tt_)), rank_src=int(R.Lens.rank(lg, ts_)), flip=bool(pr[tt_] > pr[ts_]), top1_tgt=bool(int(lg.argmax()) == tt_),
                kl1=float(F.kl_div(torch.log_softmax(ref_lg, -1), torch.log_softmax(lg, -1), log_target=True, reduction="sum")))


def family_of(nm):
    if nm.startswith("jadd"): return "jlens_add"
    if nm.startswith("jswap"): return "jlens_swap"
    if nm.startswith("rand"): return "random"
    if nm in ("none", "recon", "var_1"): return "control"
    T_ = nm[0]; core = nm[1:]
    for fam in ("tdiff", "pdiff", "prior", "gtext", "ar"):
        if core.startswith(fam): return f"unclip_{fam}_{T_}" if fam != "ar" else f"ar_{T_}"
    for c in CRIT:
        if core.startswith(c): return f"{c}_{T_}"
    return "other"


def summarize(rows, keys, subset=None):
    R_ = rows if subset is None else [r for r in rows if r["n"] in subset]; summ = {}
    if not R_: return summ
    for nm in keys:
        C_ = [r["conds"][nm] for r in R_ if nm in r["conds"]]
        if not C_: continue
        tg = np.array([x for c in C_ for x in c["tgt"]]); sr = np.array([x for c in C_ for x in c["src"]]); nll = [x for c in C_ for x in c["nll"] if x is not None]
        summ[nm] = dict(cond=nm, family=family_of(nm), n_prompts=len(C_), flip_rate=float(np.mean([c["flip"] for c in C_])), top1_tgt_rate=float(np.mean([c["top1_tgt"] for c in C_])),
                        p_tgt_mean=float(np.mean([c["p_tgt"] for c in C_])), p_src_mean=float(np.mean([c["p_src"] for c in C_])), rank_tgt_median=float(np.median([c["rank_tgt"] for c in C_])),
                        src_drop_mean=float(np.mean([r["conds"]["none"]["p_src"] - r["conds"][nm]["p_src"] for r in R_ if nm in r["conds"]])),
                        kl1_median=float(np.median([c["kl1"] for c in C_])), kl1_mean=float(np.mean([c["kl1"] for c in C_])), tgt_mention=float(tg.mean()), src_mention=float(sr.mean()), clean_swap=float((tg & ~sr).mean()),
                        nll_median=float(np.median(nll)) if nll else None, edit_rel_median=float(np.median([c["edit_rel"] for c in C_])), cos_c_mean=float(np.mean([c["cos_c"] for c in C_])),
                        jl_tgt_rank_median=float(np.median([c["jl_tgt_rank"] for c in C_])))
        for key in ("e_cos_tgt", "e_cos_src", "e_cos_cond"):
            if key in C_[0]: summ[nm][key + "_mean"] = float(np.mean([c[key] for c in C_]))
    return summ


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--decoder", required=True); p.add_argument("--prior", default="/vol_glp/unclip/prior/uprior_big_frozen/snap_2000000"); p.add_argument("--encoder-json", default="/vol_glp/unclip/encoder.json")
    p.add_argument("--tag", default="unclip_steer_v2"); p.add_argument("--n", type=int, default=48); p.add_argument("--max-src-rank", type=int, default=2); p.add_argument("--k", type=int, default=3); p.add_argument("--n-new", type=int, default=40)
    p.add_argument("--ode-steps", type=int, default=32); p.add_argument("--alphas", default="1,2,4,8"); p.add_argument("--cfgs", default="1,2,4"); p.add_argument("--cfg-window", default="0.2,0.8")
    p.add_argument("--betas", default="0.25,0.5,1,2"); p.add_argument("--jswap", default="1,2,4,8"); p.add_argument("--K", type=int, default=8); p.add_argument("--critics", default=",".join(CRIT))
    p.add_argument("--cache", default=f"{OUT}/cache_next.pt"); p.add_argument("--rebuild-cache", action="store_true"); p.add_argument("--out", default=OUT); p.add_argument("--smoke", action="store_true")
    a = p.parse_args(); fl = lambda s_: [float(x) for x in s_.split(",") if x]
    a.alphas, a.cfgs, a.cfg_window, a.betas, a.jswap = fl(a.alphas), fl(a.cfgs), fl(a.cfg_window), fl(a.betas), fl(a.jswap); crits = [c for c in a.critics.split(",") if c]
    if a.smoke: a.n, a.k, a.n_new, a.ode_steps, a.alphas, a.cfgs, a.betas, a.jswap, a.K = 4, 1, 12, 8, [2.0], [2.0], [1.0], [4.0], 2
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    pg._load(); lens = R.Lens(); tok = pg.S["tok"]
    if a.rebuild_cache or not os.path.exists(a.cache): items = build_cache(a, lens); torch.save(items, a.cache)
    else: items = torch.load(a.cache, map_location="cpu", weights_only=False); print(f"[A] {len(items)} prompts from {a.cache}", flush=True)
    items = items[: a.n]; N = len(items)
    # ---- unCLIP edits
    enc = Encoder(a.encoder_json, pg.S["snap"], DEV1); dec = Decoder(a.decoder, DEV1, base=pg.S["snap"]); prior = load_prior(a.prior, a.encoder_json, DEV1)
    dec_info = dict(path=dec.path, step=dec.step, samples=dec.samples); mu0 = dec.mu.to(DEV0)
    V, C, pd = unclip_edits(a, items, enc, dec, prior); del dec, prior; torch.cuda.empty_cache()
    dirs = {}                                                                   # name -> list of raw directions per prompt (cpu)
    for cname in crits:
        t1 = time.time()
        try:
            fb = R.load_flow(CRIT[cname]); fb.model.eval(); dd = flow_cmean_dirs(fb, items, a.K)
            for T_ in ("A", "B"): dirs[f"{T_}{cname}"] = dd[T_]
            del fb; torch.cuda.empty_cache(); print(f"[B] {cname} conditional-mean directions: {N} prompts x 2 types in {time.time() - t1:.0f}s", flush=True)
        except Exception as ex:
            import traceback; traceback.print_exc(); print(f"[B] critic {cname} failed: {ex}", flush=True); torch.cuda.empty_cache()
    critic = pg._critic(R.AR_NAME)
    for T_ in ("A", "B"): dirs[f"{T_}ar"] = [(pg.ar_pred(critic, it[f"z{T_}e"]) - pg.ar_pred(critic, it[f"z{T_}"])).float().cpu() for it in items]
    print(f"[B] edits done {time.time() - t0:.0f}s: {len(V)} unCLIP kinds, directions {sorted(dirs)}", flush=True)
    # ---- measurement
    rows = []
    for i, it in enumerate(items):
        h = it["_h"].to(DEV0).float(); ids = it["_ids"].to(DEV0); T = ids.shape[1]; src, tgt = it["src"], it["tgt"]; ts_, tt_ = R.tid(tok, src), R.tid(tok, tgt); hn = h.norm()
        V_ = torch.stack([lens.vec(ts_), lens.vec(tt_)], 1); d_jl = V_[:, 1] - V_[:, 0]
        g_ = torch.Generator(device=DEV0).manual_seed(99 + it["n"]); rnd = torch.randn(h.shape, device=DEV0, generator=g_)
        raw = {"none": h}
        for b in a.betas: raw[f"jadd_b{b:g}"] = R.resc(h, d_jl, b)
        for al in a.jswap: raw[f"jswap_a{al:g}"] = R.swap_fn(V_, al)(h[None])[0]
        for b in a.betas[1:]: raw[f"rand_b{b:g}"] = R.resc(h, rnd, b)
        for k_, v_ in V.items(): raw[k_] = v_[i].to(DEV0).float()
        for k_, dl in dirs.items():
            dd = dl[i].to(DEV0)
            for b in a.betas: raw[f"{k_}_b{b:g}"] = R.resc(h, dd, b)
        one = {k_: (v_ / v_.norm().clamp_min(1e-6) * hn if k_ != "none" else v_) for k_, v_ in raw.items()}          # direction only
        names = list(one); Eall = torch.stack([one[k_] for k_ in names]).float(); lg = lens.logits(Eall, 42); hc = h - mu0
        fe = enc.f(Eall.to(DEV1)); zA, zAe = enc.g([it["zA"]])[0], enc.g([it["zAe"]])[0]
        out = {}
        for j, nm in enumerate(names):
            out[nm] = dict(jl_tgt_rank=int(R.Lens.rank(lg[j], tt_)), jl_src_rank=int(R.Lens.rank(lg[j], ts_)), edit_rel=float((Eall[j] - h).norm() / hn), edit_rel_raw=float((raw[nm] - h).norm() / hn),
                           cos_raw=float(F.cosine_similarity(Eall[j], h, dim=0)), cos_c=float(F.cosine_similarity(Eall[j] - mu0, hc, dim=0)),
                           e_cos_tgt=float(Encoder.cos(fe[j:j + 1], zAe[None])[0]), e_cos_src=float(Encoder.cos(fe[j:j + 1], zA[None])[0]))
            if nm in C: out[nm]["e_cos_cond"] = float(Encoder.cos(fe[j:j + 1], C[nm][i:i + 1])[0])
        fmap = {nm: (lambda hb, v=one[nm]: v.to(hb.dtype).expand_as(hb)) for nm in names}; seqs_by = {}
        for greedy, KK in ((True, 1), (False, a.k)):
            gg, sl = A.run_batch(ids, names, fmap, 42, "anchor", KK, greedy, a.n_new); seqs = gg.sequences
            if greedy: first_lg = gg.scores[0].float()
            for nm, s_ in sl.items():
                out[nm].setdefault("conts", []).extend(A.cont_of(tok, seqs[j], T) for j in range(s_.start, s_.stop)); seqs_by.setdefault(nm, []).extend(seqs[j].cpu() for j in range(s_.start, s_.stop))
                if greedy: out[nm]["_lg1"] = first_lg[s_.start]
        ref = out["none"]["_lg1"]
        for nm, r in out.items():
            r.update(next_metrics(r.pop("_lg1"), ts_, tt_, ref)); r["tgt"] = [A.ment(c, tgt) for c in r["conts"]]; r["src"] = [A.ment(c, src) for c in r["conts"]]
        allseqs = [s_ for nm in names for s_ in seqs_by[nm]]; nll = A.nll_cont(ids, allseqs, T); q = 0
        for nm in names: out[nm]["nll"] = nll[q:q + len(seqs_by[nm])]; q += len(seqs_by[nm])
        rows.append(dict(n=it["n"], kind=it["kind"], text=it["text"], src=src, tgt=tgt, zA=it["zA"], zAe=it["zAe"], zB=it["zB"], zBe=it["zBe"], base_p_src=it["base_p_src"], base_p_tgt=it["base_p_tgt"],
                         prompt_diag={k: float(v[i]) for k, v in pd.items()}, conds=out))
        pick = [c for c in ["jadd_b0.5", "jadd_b1", "jswap_a4", "Btdiff_a4", "Bpdiff_a4", "Bprior_cfg2", "Bar_b1", "Bsw_tokar_b1", "Btrunk_dn64_b1", "Atdiff_a4"] if c in out]
        print(f"[C] {it['n']} {src}->{tgt} ({time.time() - t0:.0f}s) p(src) {it['base_p_src']:.2f}: " + " | ".join(f"{c} flip {int(out[c]['flip'])} p_t {out[c]['p_tgt']:.2f} kl {out[c]['kl1']:.2f} m {sum(out[c]['tgt'])}" for c in pick), flush=True)
        if (i + 1) % 8 == 0: json.dump(dict(partial=True, rows=rows), open(f"{a.out}/{a.tag}.partial.json", "w"))
    # ---- summaries: full set, J-lens fluency threshold, subset, frontier, matched-KL table
    keys = list(rows[0]["conds"]); full = summarize(rows, keys)
    jadd = sorted([k for k in keys if k.startswith("jadd")], key=lambda k: float(k.split("_b")[1]))
    weak = next((k for k in jadd if full[k]["flip_rate"] >= 0.5), jadd[-1]); tau = full[weak]["kl1_median"]
    jl_keys = [k for k in keys if k.startswith("jadd") or k.startswith("jswap")]
    subset = sorted(r["n"] for r in rows if any(r["conds"][k]["flip"] and r["conds"][k]["kl1"] <= tau for k in jl_keys))
    sub = summarize(rows, keys, set(subset))
    fams = {}
    for nm in keys: fams.setdefault(family_of(nm), []).append(nm)
    budgets = [0.5, 1.0, 2.0, 4.0, 8.0]; matched = {}
    for fam, ks in fams.items():
        matched[fam] = {}
        for B in budgets:
            ok = [k for k in ks if full[k]["kl1_median"] <= B]
            if ok:
                best = max(ok, key=lambda k: full[k]["flip_rate"]); matched[fam][str(B)] = dict(cond=best, flip_rate=full[best]["flip_rate"], tgt_mention=full[best]["tgt_mention"], p_tgt_mean=full[best]["p_tgt_mean"], kl1_median=full[best]["kl1_median"])
    pdm = {k: float(np.mean([r["prompt_diag"][k] for r in rows])) for k in rows[0]["prompt_diag"]}
    res = dict(tag=a.tag, decoder=dec_info, prior=a.prior, critics=crits, n=N, k=a.k, alphas=a.alphas, cfgs=a.cfgs, betas=a.betas, jswap=a.jswap, cfg_window=a.cfg_window, ode_steps=a.ode_steps,
               fluency=dict(weakest_jadd_with_half_flips=weak, tau_kl1=tau, subset=subset, n_subset=len(subset)), prompt_diag_mean=pdm, families=fams, summary=full, summary_subset=sub, matched_kl=matched, rows=rows)
    json.dump(res, open(f"{a.out}/{a.tag}.json", "w"))
    print(f"[summary] embedding movement per text type: " + " ".join(f"{k} {v:.3f}" for k, v in pdm.items()), flush=True)
    print(f"[summary] J-lens fluency threshold: {weak} reaches {100 * full[weak]['flip_rate']:.0f}% flips at median KL1 {tau:.2f} -> subset {len(subset)}/{N} prompts", flush=True)
    for nm in keys:
        s, ss = full[nm], sub.get(nm)
        print(f"[summary] {nm:22s} flip {s['flip_rate']:.2f} p_tgt {s['p_tgt_mean']:.3f} rank {s['rank_tgt_median']:.0f} src_drop {s['src_drop_mean']:.2f} kl1 {s['kl1_median']:.2f} nll {s['nll_median'] if s['nll_median'] is None else round(s['nll_median'], 2)} | mention tgt {s['tgt_mention']:.2f} src {s['src_mention']:.2f} | cosc {s['cos_c_mean']:.3f}"
              + (f" | subset flip {ss['flip_rate']:.2f} mention {ss['tgt_mention']:.2f}" if ss else ""), flush=True)
    for fam, m in matched.items(): print(f"[matched] {fam:18s} " + " ".join(f"KL<={B}: {m[str(B)]['cond']} flip {m[str(B)]['flip_rate']:.2f}" for B in budgets if str(B) in m), flush=True)
    print(f"[done] {N} prompts in {time.time() - t0:.0f}s -> {a.out}/{a.tag}.json", flush=True)


if __name__ == "__main__":
    main()
