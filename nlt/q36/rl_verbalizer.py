"""RL co-training of the two-activation CHANGE verbalizer and the direction critic (NLA-style), Qwen3.6-27B.

Policy  = Qwen3.6-27B + LoRA (the SFT verbalizer, common.change_prompt, u_i / u_j injected norm-matched at the two ㈜ markers after block 1).
Reward  = MINUS the co-trained critic's flow-matching loss on u_j given u_i and the rollout text, mean over a fixed t grid, with the SAME (t, eps) for every
          sample of a group (common random numbers; the no-text term cancels in the group advantage), minus lam * tokens (lam calibrated to 20% of the
          group std at the median length), minus a penalty for depth words (regex). Exact ODE bits are EVAL only (frozen critic, held-out dumps, twins, rp).
Critic  = co-trained every step on the policy's rollouts (true u_j) mixed 50/50 with replay rows of the warm-start trace pool (text dropout 0.1 keeps the
          unconditional path calibrated). A FROZEN copy of the warm-start critic scores the held-out greedy dumps every --eval-every steps (collusion guard),
          together with claim-twin P(true > twin) and random-pair bits under both critics.
Recipe  = the grad-olens ScaleRL bundle: CISPO (eps_max 5), batch-level advantage normalisation with zero-variance groups dropped, prompt-level aggregation,
          fp32 log-softmax, + KL(k3) to the frozen SFT policy. HF generate for rollouts (vLLM two-vector steering later). Data-parallel over ranks.

  torchrun --nproc_per_node 4 rl_verbalizer.py --data-dir /vol/q36/data --policy /vol/q36/verbalizer/v1/final --critic /vol/q36/critic/v1/ckpt_final.pt \
      --replay-text '/vol/q36/text/v1/train/craft_full__*.parquet' --twins '/vol/q36/text/v1/val/twins__*.parquet' --out /vol/q36/rl/v1 --steps 200 --batch 16 --group 8
"""
import argparse, copy, glob, json, math, os, re, sys, time, traceback
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"); os.environ["HF_HUB_OFFLINE"] = "0"
def _hook(t, v, tb): print(f"[rank {os.environ.get('RANK','0')}] UNCAUGHT: " + ''.join(traceback.format_exception(t, v, tb)), flush=True)
sys.excepthook = _hook
import numpy as np, pyarrow.parquet as pq, torch, torch.nn.functional as F, torch.distributed as dist
from peft import PeftModel, LoraConfig, get_peft_model
from common import D_MODEL, InjectMarkers, MARKER_ID, change_prompt, load_base, load_tokenizer, lora_target_re
from critic_data import Store, Directions, load_text_pairs, dm_partner

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True); ap.add_argument("--stats", default=None); ap.add_argument("--out", required=True); ap.add_argument("--band", default=None)
ap.add_argument("--policy", required=True, help="SFT adapter dir (PEFT); 'none' = fresh zero LoRA (mechanics smoke only)"); ap.add_argument("--critic", required=True, help="critic ckpt (train_critic.py); 'none' = random init (mechanics smoke only)"); ap.add_argument("--frozen-critic", default=None, help="ckpt for the FROZEN guard critic (default: a copy of --critic); use the best-calibrated checkpoint, not the lowest-FM-loss one")
ap.add_argument("--ref-text", default=None, help="glob(s) of the teacher text (val) -> teacher pmi/content on the held-out pairs under both critics at every eval (reference + collusion check)"); ap.add_argument("--replay-text", default=None, help="glob(s) of the warm-start trace pool (train split) for critic replay"); ap.add_argument("--twins", default=None, help="glob of val twins__*.parquet for the twin-P guard")
ap.add_argument("--steps", type=int, default=200); ap.add_argument("--batch", type=int, default=16, help="prompts per rank"); ap.add_argument("--group", type=int, default=8); ap.add_argument("--n-tok", type=int, default=176); ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--critic-lr", type=float, default=3e-5); ap.add_argument("--kl", type=float, default=0.02); ap.add_argument("--lam", type=float, default=-1.0, help="per-token cost in FM-loss units; < 0 = calibrate at step 1 (20% of the group std at the median length)"); ap.add_argument("--depth-penalty", type=float, default=0.05, help="FM-loss units subtracted per text with a depth word / empty text (the FM loss is O(1) per dim; group stds are ~1e-3..1e-2)")
ap.add_argument("--cispo-eps-max", type=float, default=5.0); ap.add_argument("--no-cotrain", action="store_true"); ap.add_argument("--replay-frac", type=float, default=0.5); ap.add_argument("--critic-micro", type=int, default=64)
ap.add_argument("--t-grid", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"); ap.add_argument("--eval-every", type=int, default=10); ap.add_argument("--heldout", type=int, default=128); ap.add_argument("--save-every", type=int, default=50)
ap.add_argument("--gen-chunk", type=int, default=32); ap.add_argument("--bwd-chunk", type=int, default=8); ap.add_argument("--no-grad-ckpt", action="store_true", help="disable gradient checkpointing on the policy (on by default: a 27B backward over 8 x 170 tokens OOMs a 140 GB H200 without it)"); ap.add_argument("--max-train-pos", type=int, default=None); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--wandb-project", default="nlt-qwen36-27b"); ap.add_argument("--wandb-entity", default="octahedral-systems"); ap.add_argument("--wandb-name", default=None); ap.add_argument("--no-wandb", action="store_true")
args = ap.parse_args()
RANK, WORLD, LRANK = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0)); is_dist, is_main = WORLD > 1, RANK == 0
if is_dist: dist.init_process_group("nccl", rank=RANK, world_size=WORLD)
dev = f"cuda:{LRANK}"; torch.cuda.set_device(dev); torch.manual_seed(args.seed + RANK); tok = load_tokenizer(); os.makedirs(args.out, exist_ok=True)
def P(*a):
    if is_main: print(*a, flush=True)
T_GRID = [float(x) for x in args.t_grid.split(",")]; PAD = tok.eos_token_id; EOT = tok.convert_tokens_to_ids("<|im_end|>")
PROMPT = change_prompt(tok); PLEN = len(PROMPT); PROMPT_T = torch.tensor([PROMPT], device=dev)

# ---- policy: "default" = trainable, "ref" = frozen SFT copy (KL reference) ----
base = load_base(dev)
if args.policy != "none":
    policy = PeftModel.from_pretrained(base, args.policy, adapter_name="default", is_trainable=True); policy.load_adapter(args.policy, adapter_name="ref"); policy.set_adapter("default")
else:
    policy = get_peft_model(base, LoraConfig(r=64, lora_alpha=16, use_rslora=True, lora_dropout=0.0, bias="none", target_modules=lora_target_re(None), task_type="CAUSAL_LM"), adapter_name="default")
    policy.add_adapter("ref", LoraConfig(r=64, lora_alpha=16, use_rslora=True, lora_dropout=0.0, bias="none", target_modules=lora_target_re(None), task_type="CAUSAL_LM")); policy.set_adapter("default")
trainable = [q for n_, q in policy.named_parameters() if q.requires_grad and ".default." in n_]
for q in trainable: q.data = q.data.float()
for n_, q in policy.named_parameters():
    if ".ref." in n_: q.requires_grad_(False)
if not args.no_grad_ckpt:
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); policy.enable_input_require_grads(); P("[rl] gradient checkpointing ON (policy)")
MPOS = [k for k, t in enumerate(PROMPT) if t == MARKER_ID]; INJ = InjectMarkers(policy, positions=MPOS)
P(f"[rl] policy {args.policy} ({sum(q.numel() for q in trainable) / 1e6:.0f}M trainable) | prompt {PLEN} tok | world {WORLD} x {args.batch} prompts x {args.group}")

# ---- critic (co-trained) + frozen copy + text encoder ----
sys.path.insert(0, "/root/easyNLA")
from nlt.prior.model import DiffusionPrior, build_prior
from nlt.critic.text_encoder import TextEncoder
from nlt.evals.regex_tags import hard_hits
EXTRA_HARD = [re.compile(p_, re.I) for p_ in (r"\b(?:layer|block|stage|step)s?\s*#?\s*\d{1,2}\b", r"\b\d{1,2}\s*(?:->|→|to)\s*\d{1,2}\b", r"\bL\d{1,2}\b", r"\bresidual stream\b", r"\bhidden states?\b", r"\bactivations?\b")]
def depth_hit(t): return bool(hard_hits(t)) or any(r_.search(t or "") for r_ in EXTRA_HARD)
dirs = Directions(args.stats or os.path.join(args.data_dir, "layer_stats.pt"), device=dev)          # radial settings are read from the critic checkpoint below
if args.critic != "none":
    ck = torch.load(args.critic, map_location="cpu"); critic = build_prior(ck["config"]); critic.load_state_dict(ck["model"]); cargs = ck["args"]
else:
    cargs = {"enc_model": "Qwen/Qwen3-0.6B", "enc_layer": 20, "enc_max_len": 192}; critic = DiffusionPrior(D_MODEL, 512, 4, 8, 8, 1024, 192, "v", 0.02, 0, 4, math.sqrt(D_MODEL))
critic.to(dev).float()
if args.frozen_critic and args.frozen_critic not in ("none", args.critic):
    fck = torch.load(args.frozen_critic, map_location="cpu"); frozen = build_prior(fck["config"]); frozen.load_state_dict(fck["model"]); frozen = frozen.to(dev).float().eval().requires_grad_(False)
    P(f"[rl] FROZEN guard critic = {args.frozen_critic} (step {fck.get('step')}); co-trained critic starts from {args.critic} (step {ck.get('step') if args.critic != 'none' else 'random'})")
else:
    frozen = copy.deepcopy(critic).eval().requires_grad_(False)
critic.train()
dirs.radial = cargs.get("radial", "lognormal"); dirs.sigma_iso = float(cargs.get("sigma_iso", 0.0)); dirs.sigma_r = float(cargs.get("sigma_r", dirs.sigma_r)); P(f"[rl] target convention: radial {dirs.radial} sigma_r {dirs.sigma_r} sigma_iso {dirs.sigma_iso}")
from huggingface_hub import snapshot_download
ENC_DIR = snapshot_download(cargs.get("enc_model", "Qwen/Qwen3-0.6B"), cache_dir="/vol/hf_cache/enc", token=os.environ.get("HF_TOKEN"))      # explicit cache_dir: HF_HOME points at the read-only 27B cache
encoder = TextEncoder(ENC_DIR, int(cargs.get("enc_layer", 20)), dev, int(cargs.get("enc_max_len", 192)))
c_opt = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr, betas=(0.9, 0.999), weight_decay=0.01)
P(f"[rl] critic {args.critic} ({critic.n_params() / 1e6:.0f}M) co-train {not args.no_cotrain} lr {args.critic_lr} | frozen copy kept for the guard")

# ---- data ----
BAND = [int(x) for x in args.band.split(",")] if args.band else None
store = Store(args.data_dir, "train", device=dev, layers=BAND, max_pos=args.max_train_pos, verbose=is_main); store_val = Store(args.data_dir, "val", device=dev, layers=BAND, verbose=False)
band = BAND or store.layers
vp = pq.read_table(os.path.join(args.data_dir, "pairs_val.parquet"), columns=["pair_id", "pos_idx", "i", "j"]).to_pandas(); vp = vp[vp["pos_idx"].isin(store_val.row_of) & vp["i"].isin(band) & vp["j"].isin(band)].iloc[: args.heldout].reset_index(drop=True)
Vh_rows = store_val.rows_for(vp["pos_idx"].values); Vh_i = torch.tensor(vp["i"].values.astype(np.int64)); Vh_j = torch.tensor(vp["j"].values.astype(np.int64))
REPLAY = None
if args.replay_text and not args.no_cotrain:
    files = sorted(sum((glob.glob(g) for g in args.replay_text.split(",")), [])); df = load_text_pairs(files, os.path.join(args.data_dir, "pairs_train.parquet")); df = df[df["pos_idx"].isin(store.row_of) & df["i"].isin(band) & df["j"].isin(band)].reset_index(drop=True)
    REPLAY = {"rows": store.rows_for(df["pos_idx"].values), "i": torch.tensor(df["i"].values.astype(np.int64)), "j": torch.tensor(df["j"].values.astype(np.int64)), "text": df["text"].astype(str).tolist()}; P(f"[rl] replay pool {len(df)} rows from {len(files)} files")
REF = None
if args.ref_text:
    dfr = load_text_pairs(sorted(sum((glob.glob(g) for g in args.ref_text.split(",")), [])), os.path.join(args.data_dir, "pairs_val.parquet"), pools_verbose=False)
    if "sample" in dfr and len(dfr): dfr = dfr[dfr["sample"].fillna(0).astype(int) == 0]
    ref_map = dfr.drop_duplicates("pair_id").set_index("pair_id")["text"].astype(str); REF = [ref_map.get(pid, "") for pid in vp["pair_id"].tolist()]
    P(f"[rl] teacher reference text for {sum(1 for t in REF if t)}/{len(REF)} held-out pairs (scored with the same (t, eps) as the policy's dumps)")
TWINS = None
if args.twins:
    import pandas as pd
    tw = pd.concat([pq.read_table(f).to_pandas() for f in sorted(glob.glob(args.twins))], ignore_index=True); vpa = pq.read_table(os.path.join(args.data_dir, "pairs_val.parquet"), columns=["pair_id", "pos_idx", "i", "j"]).to_pandas().set_index("pair_id")
    tw = tw[tw["pair_id"].isin(vpa.index)]; tw = tw[tw["pair_id"].map(lambda p_: (vpa.loc[p_, "pos_idx"] in store_val.row_of) and (int(vpa.loc[p_, "i"]) in band) and (int(vpa.loc[p_, "j"]) in band))]
    tru = tw[tw["variant"] == "true"].drop_duplicates("pair_id").set_index("pair_id")["text"]; twn = tw[tw["variant"] == "twin_shift"].drop_duplicates("pair_id").set_index("pair_id")["text"]
    com = [p_ for p_ in tru.index if p_ in twn.index][: args.heldout]
    if com:
        TWINS = {"rows": store_val.rows_for(vpa.loc[com, "pos_idx"].values), "i": torch.tensor(vpa.loc[com, "i"].values.astype(np.int64)), "j": torch.tensor(vpa.loc[com, "j"].values.astype(np.int64)), "true": tru.loc[com].tolist(), "twin": twn.loc[com].tolist()}; P(f"[rl] twins: {len(com)} val pairs (true vs twin_shift)")
rng = np.random.default_rng(args.seed + 17 * RANK); gen_t = torch.Generator(device="cpu").manual_seed(args.seed + 101 * RANK)
def sample_batch(B):
    """random positions, i<j uniform over the band -> (rows, i, j)"""
    rows = torch.randint(0, store.N, (B,), generator=gen_t); Ls = torch.tensor(sorted(band)); a = torch.randint(0, len(Ls), (B,), generator=gen_t); b = torch.randint(0, len(Ls) - 1, (B,), generator=gen_t); b = b + (b >= a).long()
    return rows, Ls[torch.minimum(a, b)], Ls[torch.maximum(a, b)]
def vecs_for(st, rows, i, j):
    u_i = dirs.unit(st.gather(rows, i, dev), i); u_j = dirs.unit(st.gather(rows, j, dev), j); return torch.stack([u_i, u_j], 1), dirs.source(st.gather(rows, i, dev), i), u_j
def y_of(u, seed):
    """the critic's TARGET for a unit direction u: sqrt(d) u s with the checkpoint's radial convention (lognormal s = exp(sigma_r eps_r), or fixed s = 1 + isotropic noise).
    Deterministic given the seed so a prompt's G rollouts (and the two critics at eval) see the same target. The critic was trained on this scale (rms ~ sqrt(d)); feeding the
    unit u itself (norm 1) puts x_t ~ t eps off-distribution and makes the reward text-blind (rl_v1 / rl_v2 bug)."""
    g = torch.Generator(device=dev).manual_seed(int(seed))
    if getattr(dirs, "radial", "lognormal") == "fixed": return dirs.sqrt_d * u + float(getattr(dirs, "sigma_iso", 0.0)) * torch.randn(u.shape, generator=g, device=dev)
    return dirs.sqrt_d * u * torch.exp(dirs.sigma_r * torch.randn(u.shape[0], generator=g, device=dev))[:, None]

with torch.no_grad():
    _y = y_of(dirs.unit(store_val.gather(Vh_rows[:64], Vh_j[:64], dev), Vh_j[:64]), 1); P(f"[rl] target per-dim rms over 64 held-out pairs = {float(_y.pow(2).mean().sqrt()):.2f}, norm = {float(_y.norm(dim=-1).mean()):.1f} (critic trained on norm ~sqrt(d) = {dirs.sqrt_d:.2f}, per-dim rms ~1; a unit u would give per-dim rms {1 / dirs.sqrt_d:.3f})")

# ---- critic scoring: proxy bits with shared (t, eps) ----
@torch.no_grad()
def proxy_bits(model, src, y, texts, eps_bank, dm_texts=None):
    """-> bits(text) [N] = (d/2) mean_t [L(no text) - L(text)] / ln2 ; eps_bank: list over t of [N, d]. Optional dm_texts -> content bits too."""
    d = y.shape[-1]; N = y.shape[0]; Lc = torch.zeros(len(T_GRID), N, device=dev); Lu = torch.zeros_like(Lc); Ld = torch.zeros_like(Lc)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        enc, mask = encoder(texts); enc_d, mask_d = encoder(dm_texts) if dm_texts is not None else (None, None)
    for ti, t in enumerate(T_GRID):
        tt = torch.full((N,), float(t), device=dev); x_t = (1 - tt)[:, None] * y + tt[:, None] * eps_bank[ti]; tgt = eps_bank[ti] - y
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vc = model(x_t, tt, src, enc=enc, enc_mask=mask).float(); vu = model(x_t, tt, src).float()
            Lc[ti] = ((vc - tgt) ** 2).mean(-1); Lu[ti] = ((vu - tgt) ** 2).mean(-1)
            if enc_d is not None: Ld[ti] = ((model(x_t, tt, src, enc=enc_d, enc_mask=mask_d).float() - tgt) ** 2).mean(-1)
    pmi = (d / 2) * (Lu - Lc).mean(0) / math.log(2); cont = (d / 2) * (Ld - Lc).mean(0) / math.log(2) if enc_d is not None else None
    return pmi, cont
def eps_for(N, d, seed):
    g = torch.Generator(device=dev).manual_seed(int(seed)); return [torch.randn(N, d, generator=g, device=dev) for _ in T_GRID]

# ---- policy sampling / log-probs ----
def gen(vecs, temperature):
    outs = []
    for a in range(0, vecs.shape[0], args.gen_chunk):
        vb = vecs[a:a + args.gen_chunk]; ids = PROMPT_T.repeat(vb.shape[0], 1); INJ.set(vb, ids)
        try:
            kw = dict(do_sample=temperature > 0, temperature=temperature if temperature > 0 else None, top_p=1.0 if temperature > 0 else None, top_k=0 if temperature > 0 else None)
            g = policy.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=args.n_tok, pad_token_id=PAD, eos_token_id=[EOT, PAD], suppress_tokens=[MARKER_ID], use_cache=True, **kw)
        finally: INJ.off()
        s = g[:, PLEN:]; s = F.pad(s, (0, args.n_tok - s.shape[1]), value=PAD) if s.shape[1] < args.n_tok else s; outs.append(s)
    return torch.cat(outs)
def sample_mask(samp):
    is_end = ((samp == PAD) | (samp == EOT)).long(); first = torch.cumsum(is_end, 1); return ((first == 0) | ((first == 1) & (is_end == 1))).float()
def seq_logp(vecs, samp, adapter="default"):
    policy.set_adapter(adapter); ids = torch.cat([PROMPT_T.repeat(vecs.shape[0], 1), samp], 1); INJ.set(vecs, ids)
    try: out = policy(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    finally: INJ.off(); policy.set_adapter("default")
    lg = out.logits[:, PLEN - 1:PLEN - 1 + samp.shape[1]].float(); return torch.log_softmax(lg, -1).gather(-1, samp[..., None])[..., 0]
def decode(samp):
    out = []
    for row in samp.tolist():
        ids = []
        for t in row:
            if t in (PAD, EOT): break
            ids.append(t)
        out.append(tok.decode(ids).strip())
    return out

# ---- reward: the flow-matching loss itself (user: "just use the flow loss as reward") ----
@torch.no_grad()
def fm_loss_text(model, src, y, texts, eps_bank):
    """mean over the t grid of the per-row FM loss (velocity MSE, mean over dims) of the critic given the text; eps_bank: list over t of [N, d]"""
    N = y.shape[0]; L = torch.zeros(len(T_GRID), N, device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16): enc, mask = encoder(texts)
    for ti, t in enumerate(T_GRID):
        tt = torch.full((N,), float(t), device=dev); x_t = (1 - tt)[:, None] * y + tt[:, None] * eps_bank[ti]; tgt = eps_bank[ti] - y
        with torch.autocast("cuda", dtype=torch.bfloat16): v = model(x_t, tt, src, enc=enc, enc_mask=mask).float()
        L[ti] = ((v - tgt) ** 2).mean(-1)
    return L.mean(0)
LAM = [args.lam]                                            # calibrated at step 1 when --lam < 0: 20% of the group std at the median length
def rewards_for(src, u_j, texts, G, seed):
    """texts grouped by prompt (B*G, prompt-major); the SAME (t grid, eps) for every rollout of a prompt (common random numbers) -> reward = -FM loss - lam*tokens - depth penalty.
    The no-text term is a constant within a group (shared noise) and cancels in the group-normalised advantage, so it is not computed."""
    N = len(texts); Bp = N // G; eps_p = eps_for(Bp, D_MODEL, seed); eps_bank = [e.repeat_interleave(G, 0) for e in eps_p]
    fm = fm_loss_text(critic.eval(), src, u_j, [t if t else " " for t in texts], eps_bank); critic.train()
    ntok = torch.tensor([len(tok(t, add_special_tokens=False).input_ids) for t in texts], device=dev, dtype=torch.float32); hits = torch.tensor([float(depth_hit(t)) for t in texts], device=dev)
    empty = torch.tensor([float(not t.strip()) for t in texts], device=dev)
    if LAM[0] < 0:
        gstd = float((-fm).view(Bp, G).std(1).mean()); med = float(ntok.median().clamp_min(1)); LAM[0] = 0.2 * gstd / med
        P(f"[rl] lambda calibrated: group std of -FM {gstd:.4f}, median tokens {med:.0f} -> lam {LAM[0]:.6f} per token (20% of the group std at the median length)")
    r = -fm - LAM[0] * ntok - args.depth_penalty * hits - args.depth_penalty * empty
    return r, -fm, ntok, hits

# ---- critic co-training step ----
def critic_step(src, u_j, texts, G):
    if args.no_cotrain: return float("nan")
    N = len(texts); n_rep = int(N * args.replay_frac / (1 - args.replay_frac)) if REPLAY is not None else 0
    S, Y, T = [src], [u_j], list(texts)
    if n_rep:
        idx = torch.randint(0, len(REPLAY["text"]), (n_rep,), generator=gen_t); _, s_r, y_r = vecs_for(store, REPLAY["rows"][idx], REPLAY["i"][idx], REPLAY["j"][idx]); S.append(s_r); Y.append(y_r); T += [REPLAY["text"][k] for k in idx.tolist()]
    S = torch.cat(S); Y = torch.cat(Y); y = dirs.sqrt_d * Y * torch.exp(dirs.sigma_r * torch.randn(Y.shape[0], device=dev))[:, None] if dirs.radial != "fixed" else dirs.sqrt_d * Y + dirs.sigma_iso * torch.randn_like(Y)
    keep = torch.rand(len(T), device=dev) >= 0.1; tot = 0.0; c_opt.zero_grad(set_to_none=True); n_all = len(T)
    for s0 in range(0, n_all, args.critic_micro):
        sl = slice(s0, min(n_all, s0 + args.critic_micro)); nb = sl.stop - sl.start
        with torch.autocast("cuda", dtype=torch.bfloat16): enc, mask = encoder([t if t else " " for t in T[sl]])
        mask = mask & keep[sl][:, None]; t = torch.rand(nb, device=dev); eps = torch.randn_like(y[sl])
        with torch.autocast("cuda", dtype=torch.bfloat16): l, _ = critic.loss(y[sl], S[sl], t, eps, enc, mask)
        (l.mean() * nb / n_all).backward(); tot += float(l.mean()) * nb / n_all
    if is_dist:
        for q in critic.parameters():
            if q.grad is None: q.grad = torch.zeros_like(q)
            dist.all_reduce(q.grad); q.grad.div_(WORLD)
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0); c_opt.step(); return tot

# ---- eval: greedy dumps on held-out pairs, both critics, twins, rp ----
@torch.no_grad()
def evaluate(step):
    policy.eval(); vecs, src, u_j = vecs_for(store_val, Vh_rows, Vh_i, Vh_j); y_j = y_of(u_j, 4321); g = gen(vecs, 0.0); texts = decode(g); N = len(texts)
    eps_bank = eps_for(N, D_MODEL, 4321); dmp = dm_partner(Vh_i, Vh_j); dm_texts = [texts[q] for q in dmp]; rp_texts = [texts[(q + N // 2) % N] for q in range(N)]
    out = {"step": step, "tokens": float(np.mean([len(tok(t, add_special_tokens=False).input_ids) for t in texts])), "depth_hit_rate": float(np.mean([depth_hit(t) for t in texts])), "empty_rate": float(np.mean([not t.strip() for t in texts]))}
    for name, model in (("cotrained", critic.eval()), ("frozen", frozen)):
        pmi, cont = proxy_bits(model, src, y_j, [t if t else " " for t in texts], eps_bank, [t if t else " " for t in dm_texts]); rp, _ = proxy_bits(model, src, y_j, [t if t else " " for t in rp_texts], eps_bank)
        out[f"{name}/pmi_bits"] = float(pmi.mean()); out[f"{name}/content_bits"] = float(cont.mean()); out[f"{name}/p_z_gt_dm"] = float((cont > 0).float().mean()); out[f"{name}/rp_bits"] = float(rp.mean())
        if REF is not None:
            ok = torch.tensor([bool(t) for t in REF], device=dev); pmi_r, cont_r = proxy_bits(model, src, y_j, [t if t else " " for t in REF], eps_bank, [REF[q] if REF[q] else " " for q in dmp])
            out[f"{name}/teacher_pmi_bits"] = float(pmi_r[ok].mean()); out[f"{name}/teacher_content_bits"] = float(cont_r[ok].mean()); out[f"{name}/teacher_p_z_gt_dm"] = float((cont_r[ok] > 0).float().mean())
        if TWINS is not None:
            _, s_t, u_t = vecs_for(store_val, TWINS["rows"], TWINS["i"], TWINS["j"]); y_t = y_of(u_t, 999); eb = eps_for(len(TWINS["true"]), D_MODEL, 999)
            bt, _ = proxy_bits(model, s_t, y_t, TWINS["true"], eb); bw, _ = proxy_bits(model, s_t, y_t, TWINS["twin"], eb); out[f"{name}/twin_p_true_gt_twin"] = float((bt > bw).float().mean()); out[f"{name}/twin_true_bits"] = float(bt.mean())
    critic.train(); policy.train()
    P(f"  [eval {step}] cotrained pmi {out['cotrained/pmi_bits']:.1f} content {out['cotrained/content_bits']:.1f} P {out['cotrained/p_z_gt_dm']:.2f} rp {out['cotrained/rp_bits']:.1f} | FROZEN pmi {out['frozen/pmi_bits']:.1f} content {out['frozen/content_bits']:.1f} P {out['frozen/p_z_gt_dm']:.2f} rp {out['frozen/rp_bits']:.1f}" + (f" | twins co {out['cotrained/twin_p_true_gt_twin']:.2f} fr {out['frozen/twin_p_true_gt_twin']:.2f}" if TWINS is not None else "") + f" | tokens {out['tokens']:.0f} depth-hits {out['depth_hit_rate']:.2%} empty {out['empty_rate']:.2%}" + (f" | TEACHER content co {out['cotrained/teacher_content_bits']:.1f} (P {out['cotrained/teacher_p_z_gt_dm']:.2f}) fr {out['frozen/teacher_content_bits']:.1f} (P {out['frozen/teacher_p_z_gt_dm']:.2f})" if REF is not None else ""))
    for q in range(2): P(f"    [{int(Vh_i[q])}->{int(Vh_j[q])}] {texts[q][:300]!r}")
    out["examples"] = texts[:8]; return out

optim = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
wb = None
if is_main and not args.no_wandb and os.environ.get("WANDB_API_KEY"):
    import wandb; os.environ.setdefault("WANDB_DIR", "/root/wandb"); os.makedirs("/root/wandb", exist_ok=True); wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name or "rl_" + os.path.basename(args.out.rstrip("/")), config=vars(args))
if is_main: ev0 = evaluate(0); json.dump(ev0, open(f"{args.out}/eval_0000.json", "w"), indent=1); wb and wb.log({"step": 0, **{"eval/" + k: v for k, v in ev0.items() if isinstance(v, float)}})
if is_dist: dist.barrier()
t0 = time.time(); B, G = args.batch, args.group
for step in range(1, args.steps + 1):
    rows, i, j = sample_batch(B); vecs, src, u_j = vecs_for(store, rows, i, j); vG = vecs.repeat_interleave(G, 0); sG = src.repeat_interleave(G, 0); uG = u_j.repeat_interleave(G, 0); yG = y_of(u_j, args.seed + 7919 * step + RANK).repeat_interleave(G, 0)
    with torch.no_grad():
        policy.eval(); samp = gen(vG, args.temp); policy.train(); texts = decode(samp)
        rew, pmi, ntok, hits = rewards_for(sG, yG, texts, G, seed=args.seed + 1000 * step + RANK); rg = rew.view(B, G); adv = rg - rg.mean(1, keepdim=True)
        nz = (rg.std(1) > 1e-6); keep = nz.repeat_interleave(G); advf = adv.view(-1) * keep
        stats = torch.tensor([advf[keep].double().pow(2).sum().item(), advf[keep].double().sum().item(), float(keep.sum())], dtype=torch.float64, device=dev)
        if is_dist: dist.all_reduce(stats)
        n_all = stats[2].item(); std = math.sqrt(max(stats[0].item() / n_all - (stats[1].item() / n_all) ** 2, 0.0)) if n_all > 1 else 1.0; adv = (advf / (std + 1e-6)).view(-1)
        mask = sample_mask(samp) * keep.float()[:, None]; m3 = mask.view(B, G, -1); tok_g = m3.sum((1, 2)); n_eff_g = max(int((tok_g > 0).sum()), 1)
        w = (m3 / tok_g.clamp(min=1)[:, None, None] / n_eff_g).view(B * G, -1)
        old_lp = torch.cat([seq_logp(vG[a:a + args.bwd_chunk], samp[a:a + args.bwd_chunk]) for a in range(0, B * G, args.bwd_chunk)])
        ref_lp = torch.cat([seq_logp(vG[a:a + args.bwd_chunk], samp[a:a + args.bwd_chunk], "ref") for a in range(0, B * G, args.bwd_chunk)]) if args.kl > 0 else None
    optim.zero_grad(set_to_none=True); pg_tot = 0.0; kl_tot = 0.0
    for a in range(0, B * G, args.bwd_chunk):
        sl = slice(a, a + args.bwd_chunk); lp = seq_logp(vG[sl], samp[sl]); rho = torch.exp(lp.detach() - old_lp[sl]).clamp(max=args.cispo_eps_max)
        loss_tok = -(rho * adv[sl, None] * lp)
        if ref_lp is not None:
            k3 = torch.exp(ref_lp[sl] - lp) - (ref_lp[sl] - lp) - 1; loss_tok = loss_tok + args.kl * k3; kl_tot += float((k3.detach() * mask[sl]).sum() / mask[sl].sum().clamp_min(1))
        pg = (loss_tok * w[sl]).sum(); pg.backward(); pg_tot += pg.item()
    if is_dist:
        wsum = torch.tensor([float(n_eff_g)], device=dev); dist.all_reduce(wsum)
        for q in trainable:
            if q.grad is None: q.grad = torch.zeros_like(q)
            q.grad.mul_(float(n_eff_g)); dist.all_reduce(q.grad); q.grad.div_(wsum.item())
    gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0); optim.step()
    c_loss = critic_step(sG, uG, texts, G)
    if is_main:
        log = {"step": step, "reward": rew.mean().item(), "neg_fm_loss": pmi.mean().item(), "lam": LAM[0], "tokens": ntok.mean().item(), "depth_hit_rate": hits.mean().item(), "groups_kept": int(nz.sum()), "kl_k3": kl_tot / max(1, (B * G) // args.bwd_chunk), "grad_norm": float(gn), "critic_loss": c_loss, "min": (time.time() - t0) / 60}
        print(f"step {step:04d} | reward {log['reward']:.4f} | -fm {log['neg_fm_loss']:.4f} | tokens {log['tokens']:.0f} | depth-hits {log['depth_hit_rate']:.1%} | groups {log['groups_kept']}/{B} | kl {log['kl_k3']:.4f} | critic {c_loss:.4f} | gn {float(gn):.2f} | {log['min']:.1f} min", flush=True)
        if wb: wb.log(log)
        if step % args.eval_every == 0:
            ev = evaluate(step); json.dump(ev, open(f"{args.out}/eval_{step:04d}.json", "w"), indent=1)
            if wb: wb.log({"step": step, **{"eval/" + k: v for k, v in ev.items() if isinstance(v, float)}})
        if step % args.save_every == 0 or step == args.steps:
            policy.save_pretrained(f"{args.out}/step_{step:04d}", selected_adapters=["default"]); torch.save({"model": critic.state_dict(), "config": critic.config(), "args": cargs, "step": step, "d_enc": encoder.d_enc}, f"{args.out}/critic_step_{step:04d}.pt")
    if is_dist and (step % args.eval_every == 0 or step % args.save_every == 0): dist.barrier()
P("RL_DONE")
if is_dist: dist.destroy_process_group()
