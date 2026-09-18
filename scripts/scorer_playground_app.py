"""Scorer playground: MSE reconstructor vs conditional flow, side by side, on the same activation and the same candidate explanations.
Pick a held-out row (or paste text -> activation at its last token), collect candidates (typed, the gold explanation, samples from any AV
checkpoint, automatic NUMBER perturbations: wrong / hedged / removed, a shuffled-gold control), then rank them under
  * the frozen SFT MSE critic (ar_sft_merged): MSE to the gold activation (unit-L2), FVE-equivalent, cosine
  * the frozen conditional flow (655M prior + all-pairs adapter): exact log p(h|z) - log p(h) in bits (probability-flow ODE, paired probes),
    plus the uniform-t denoising-gain proxy in bits.
Reuses the checkpoint playground's state (base + AV adapters + critic + eval rows): scripts/playground_app.py."""
import hashlib, math, os, random, re, threading, time
import numpy as np
import torch
import playground_app as P

DEV = "cuda"; LOCK = threading.Lock()
PRIOR = os.environ.get("NLA_FLOW_PRIOR", "/vol_glp/glp27b_main/ckpts/snap_000655M")
ADAPTER = os.environ.get("NLA_FLOW_ADAPTER", "/vol_glp/cond/cond_655M_all/adapter_latest.pt")
STATS = os.environ.get("NLA_FLOW_STATS", "/vol_glp/glp27b_main/rep_statistics.pt")
ODE_STEPS, PROBES, GAIN_K = 40, 2, 16
NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![\w])")


def init():
    if P.S.get("flow") is not None:
        return
    P.init("qwen36_27b")
    from nla.flow.rl_critic import FlowCritic
    t0 = time.time()
    P.S["flow"] = FlowCritic(PRIOR, ADAPTER, STATS, P.S["model"], P.S["tok"], torch.device(DEV), enc_layer=P.S["F"]["extraction_layer"], train_adapter=False)
    P.S["lpu_cache"] = {}
    print(f"[scorer-pg] flow loaded in {time.time()-t0:.0f}s", flush=True)


def _perturb_number(m, rng, mode):
    s, frac = m.group(1), m.group(2) or ""
    n = int(s.replace(",", ""))
    if mode == "removed":
        return "some year" if (1900 <= n <= 2100 and not frac and "," not in s) else ("some" if n > 1 else "a")
    if 1900 <= n <= 2100 and not frac and "," not in s:                          # a year: move it by a few years, keep it a year
        alt = n + rng.choice([-1, 1]) * rng.randint(1, 30)
    else:
        alt = rng.choice([n * 2 + 1, max(0, n // 2 - 1), n + rng.randint(3, 17), n * 10 + 3, abs(n - rng.randint(11, 97))])
        if alt == n: alt = n + 7
    alt_s = f"{alt:,}" if "," in s else str(alt)
    if mode == "wrong":
        return alt_s + frac
    return f"{s}{frac} or {alt_s}{frac}"        # hedge


def number_variants(z, rng):
    if not NUM.search(z):
        return []
    return [(f"numbers {mode}", NUM.sub(lambda m: _perturb_number(m, rng, mode), z)) for mode in ("wrong", "hedged", "removed")]


@torch.no_grad()
def flow_scores(expls, act):
    """-> list of (pmi_bits, gain_bits) under the frozen flow; log p(h) cached per activation."""
    from nla.flow.eval_cond import exact_logp, denoise_gain
    fc = P.S["flow"]; d = fc.d
    a = torch.tensor(np.asarray(act), dtype=torch.float32, device=DEV).unsqueeze(0); x0 = fc.norm.normalize(a)
    key = hashlib.md5(a.cpu().numpy().tobytes()).hexdigest()
    if key not in P.S["lpu_cache"]:
        g = torch.Generator(device=DEV).manual_seed(1234)
        P.S["lpu_cache"][key] = float(exact_logp(fc.model, x0, None, None, n_steps=ODE_STEPS, probes=PROBES, gen=g)[0])
    lpu = P.S["lpu_cache"][key]
    out = [None] * len(expls); idx = [i for i, e in enumerate(expls) if e and e.strip()]
    for cs in range(0, len(idx), 16):
        ch = idx[cs: cs + 16]; enc, mk = fc.encode([expls[i] for i in ch]); xk = x0.expand(len(ch), -1).contiguous()
        g = torch.Generator(device=DEV).manual_seed(1234)                          # same probes as the unconditional pass -> paired
        lpc = exact_logp(fc.model, xk, enc, mk, n_steps=ODE_STEPS, probes=PROBES, gen=g)
        g2 = torch.Generator(device=DEV).manual_seed(99); lc, lu = denoise_gain(fc.model, xk, enc, mk, GAIN_K, g2)
        for r, i in enumerate(ch):
            out[i] = ((float(lpc[r]) - lpu) / math.log(2), float((lu[r] - lc[r]) / 2 * d / math.log(2)))
    return [o if o is not None else (float("nan"), float("nan")) for o in out]


def _spearman(x, y):
    n = len(x)
    if n < 3: return float("nan")
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y)); return float(1 - 6 * ((rx - ry) ** 2).sum() / (n * (n * n - 1)))


def run(mode, row_idx, custom_text, expl_text, add_gold, add_numvar, add_shuffled, gen_labels, k, temperature, numvar_source):
    with LOCK:
        return _run(mode, row_idx, custom_text, expl_text, add_gold, add_numvar, add_shuffled, gen_labels, k, temperature, numvar_source)


def _run(mode, row_idx, custom_text, expl_text, add_gold, add_numvar, add_shuffled, gen_labels, k, temperature, numvar_source):
    init()
    from nla.schema import extract_explanation
    S = P.S; F = S["F"]; rows = S["rows"]; rng = random.Random(int(row_idx) + 7)
    if mode == "eval row":
        r = rows[int(row_idx) % len(rows)]; act = r["activation"]; prompt_msgs = r["prompt"]; src = r["source"]; gold = extract_explanation(r["gold"] or "") or (r["gold"] or "")
    else:
        if not (custom_text or "").strip():
            return "Paste some text."
        act = P.activation_for_text(custom_text); prompt_msgs = rows[0]["prompt"]; src = custom_text[-P.MAX_CTX:]; gold = ""
    cands = []   # (label, text)
    for i, line in enumerate([l.strip() for l in (expl_text or "").splitlines() if l.strip()]):
        cands.append((f"typed #{i+1}", line))
    if add_gold and gold:
        cands.append(("gold explanation", gold.strip()))
    t_gen = 0.0
    for lab in (gen_labels or []):
        t0 = time.time(); gens = P.generate(lab, prompt_msgs, act, int(k), float(temperature)); t_gen += time.time() - t0
        short = lab.split(" — ")[-1]
        for j, (_, e) in enumerate(gens):
            if e: cands.append((f"{short} sample {j+1}", e.strip()))
    if add_numvar and cands:
        base_label, base_text = cands[0] if numvar_source == "first candidate" else (cands[-1] if numvar_source == "last candidate" else next(((l, t) for l, t in cands if l == "gold explanation"), cands[0]))
        vs = number_variants(base_text, rng)
        cands += [(f"{base_label} → {lab}", txt) for lab, txt in vs]
        if not vs: cands.append(("(no numbers found in the chosen candidate; no number variants)", ""))
    if add_shuffled:
        other = rows[(int(row_idx) + 137) % len(rows)]; og = extract_explanation(other["gold"] or "") or (other["gold"] or "")
        if og: cands.append(("control: gold of a DIFFERENT row", og.strip()))
    cands = [(l, t) for l, t in cands if t]
    if not cands:
        return "No candidates: type some explanations (one per line), tick 'gold', or pick a checkpoint to sample from."
    texts = [t for _, t in cands]
    t0 = time.time(); ms = P.score(S["critics"][S["frozen_key"]], texts, act); t_mse = time.time() - t0
    t0 = time.time(); fs = flow_scores(texts, act); t_flow = time.time() - t0
    base_mse = S["baseline"]
    mse_rank = np.argsort(np.argsort([m for m, _ in ms])) + 1                  # lower MSE = better = rank 1
    pmi_rank = np.argsort(np.argsort([-p for p, _ in fs])) + 1                 # higher PMI = better = rank 1
    rho = _spearman([-m for m, _ in ms], [p for p, _ in fs])
    order = np.argsort([-p for p, _ in fs])
    md = [f"**Source (last 500 chars; activation = layer {F['extraction_layer']} output at the final token):**\n\n```\n…{src[-500:]}\n```"]
    md.append(f"**Ranking under the two scorers** (sorted by flow log-probability). Spearman ρ between the scorers' rankings: **{rho:.2f}** "
              f"({len(cands)} candidates; MSE critic {t_mse:.1f}s, flow exact log p {t_flow:.1f}s{f', sampling {t_gen:.0f}s' if t_gen else ''}). "
              f"MSE critic: MSE of its reconstruction to the gold activation (unit-L2; FVE-equiv = 1 − MSE/{base_mse:.3f}). Flow: log p(h|z) − log p(h) in bits under the frozen flow "
              f"(ODE {ODE_STEPS} Heun steps, {PROBES} paired Hutchinson probes); gain = uniform-t denoising proxy ({GAIN_K} fixed noise draws).")
    md.append("| flow rank | MSE rank | candidate | flow PMI (bits) ↑ | denoising gain (bits) ↑ | critic MSE ↓ | FVE-equiv | cos |\n|---|---|---|---|---|---|---|---|")
    for i in order:
        lab, txt = cands[i]; m, c = ms[i]; p_, g_ = fs[i]
        t_show = txt.replace("|", "\\|").replace("\n", " "); t_show = t_show if len(t_show) <= 220 else t_show[:217] + "…"
        md.append(f"| {pmi_rank[i]} | {mse_rank[i]} | **{lab}**: {t_show} | {p_:.1f} | {g_:.0f} | {m:.3f} | {100*(1-m/base_mse):.0f}% | {c:.2f} |")
    return "\n".join(md)


def build_ui():
    import gradio as gr
    init()
    labels = [l for l, _ in P.S["catalog"]]
    defaults = [l for l in labels if l.endswith(("qwen36_av @ 7813",))][:1]
    with gr.Blocks(title="NLA scorer playground — MSE reconstructor vs conditional flow (Qwen3.6-27B, layer 42)") as demo:
        gr.Markdown("## MSE reconstructor vs conditional activation flow: how do they rank explanations of the same activation?\n"
                    "Frozen SFT MSE critic (43-block trunk + affine head) vs the frozen conditional flow (13.7B prior trained on 655M layer-42 activations + the all-pairs conditioning adapter). "
                    "Type candidates, add the gold explanation, sample from AV checkpoints, and add automatic **number** perturbations (wrong / hedged 'X or Y' / removed) to see which scorer prices a wrong number, a hedge, and an omission how.")
        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Radio(["eval row", "custom text"], value="eval row", label="activation source")
                row_idx = gr.Slider(0, len(P.S["rows"]) - 1, value=0, step=1, label="held-out eval row (activation at the marked token; gold = its SFT target)")
                custom_text = gr.Textbox(lines=4, label="custom text (activation at its LAST token; no gold)")
                expl_text = gr.Textbox(lines=6, label="candidate explanations, one per line", placeholder="e.g. The text is a 2019 financial report listing revenue of $4.2 billion ...")
                with gr.Row():
                    add_gold = gr.Checkbox(True, label="add the gold explanation"); add_shuffled = gr.Checkbox(True, label="add a shuffled-gold control (another row's gold)")
                with gr.Row():
                    add_numvar = gr.Checkbox(True, label="add number variants (wrong / hedged / removed)"); numvar_source = gr.Radio(["gold", "first candidate", "last candidate"], value="gold", label="perturb numbers of")
                gen_labels = gr.CheckboxGroup(labels, value=defaults, label="also sample explanations from these AV checkpoints (Karvonen injection of the same activation)")
                with gr.Row():
                    k = gr.Slider(1, 8, value=3, step=1, label="samples per checkpoint"); temperature = gr.Slider(0.0, 1.5, value=1.0, step=0.1, label="temperature")
                btn = gr.Button("Score", variant="primary")
            with gr.Column(scale=2):
                out = gr.Markdown()
        btn.click(run, [mode, row_idx, custom_text, expl_text, add_gold, add_numvar, add_shuffled, gen_labels, k, temperature, numvar_source], out, api_name="score")
    return demo
