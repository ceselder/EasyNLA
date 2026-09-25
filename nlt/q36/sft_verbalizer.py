"""SFT of the two-marker CHANGE verbalizer: Qwen3.6-27B + LoRA (r64, a16, rsLoRA, lr 3e-5), the earlier state u_i injected at the first ㈜ marker and
the later state u_j at the second (Karvonen norm-matched add at the block-1 output, common.InjectMarkers), constant prompt (common.change_prompt),
no layer / depth / gap label anywhere. Targets = crafted change texts (craft_text.py pools). Data-parallel over ranks (per-rank shard of the text rows).

  torchrun --nproc_per_node 4 sft_verbalizer.py --data-dir /vol/q36/data --text '/vol/q36/text/v1/train/craft_full__*.parquet' --out /vol/q36/verbalizer/v1 \
      --steps 400 --batch 8 --grad-accum 4 --lr 3e-5 --val-text '/vol/q36/text/v1/val/craft_full__*.parquet'
Injected vectors are DIRECTIONS u = unit(h - mu_layer) (critic_data.Directions; norm-matching makes the scale moot anyway).
"""
import argparse, glob, json, os, time
import numpy as np, pyarrow.parquet as pq, torch, torch.nn.functional as F
from peft import LoraConfig, get_peft_model, PeftModel
from common import D_MODEL, InjectMarkers, MARKER_ID, change_prompt, load_base, load_tokenizer, lora_target_re
from critic_data import Store, Directions, load_text_pairs

p = argparse.ArgumentParser()
p.add_argument("--data-dir", required=True); p.add_argument("--text", required=True, help="comma list of globs (train pools)"); p.add_argument("--val-text", default=None); p.add_argument("--out", required=True)
p.add_argument("--steps", type=int, default=400); p.add_argument("--batch", type=int, default=8, help="per-GPU micro-batch"); p.add_argument("--grad-accum", type=int, default=4); p.add_argument("--lr", type=float, default=3e-5)
p.add_argument("--lora-r", type=int, default=64); p.add_argument("--lora-alpha", type=int, default=16); p.add_argument("--max-len", type=int, default=160); p.add_argument("--init-adapter", default=None)
p.add_argument("--band", default=None, help="comma list of layers to load from the store (RAM)"); p.add_argument("--samples", default=None, help="comma list of readout sample ids to keep (default all)"); p.add_argument("--val-rows", type=int, default=512); p.add_argument("--eval-every", type=int, default=100); p.add_argument("--save-every", type=int, default=200); p.add_argument("--warmup", type=int, default=20)
p.add_argument("--seed", type=int, default=0); p.add_argument("--wandb-project", default="nlt-qwen36-27b"); p.add_argument("--wandb-entity", default="octahedral-systems"); p.add_argument("--wandb-name", default=None); p.add_argument("--no-wandb", action="store_true")
args = p.parse_args()
import torch.distributed as dist
RANK, WORLD, LRANK = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0)); is_dist, is_main = WORLD > 1, RANK == 0
if is_dist: dist.init_process_group("nccl", rank=RANK, world_size=WORLD)
dev = f"cuda:{LRANK}"; torch.cuda.set_device(dev); torch.manual_seed(args.seed)
if is_main: os.makedirs(args.out, exist_ok=True)
tok = load_tokenizer(); pad_id = tok.eos_token_id; EOT = tok.convert_tokens_to_ids("<|im_end|>")
PROMPT = change_prompt(tok); PLEN = len(PROMPT); PROMPT_T = torch.tensor(PROMPT, dtype=torch.long)
def P(*a):
    if is_main: print(*a, flush=True)
P(f"[sft] prompt {PLEN} tokens: {tok.decode(PROMPT)!r}")

model = load_base(dev)
if args.init_adapter: model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
else: model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, use_rslora=True, lora_dropout=0.0, bias="none", target_modules=lora_target_re(None), task_type="CAUSAL_LM"))
if is_main: model.print_trainable_parameters()
inj = InjectMarkers(model)

# ---- data: text rows joined to pairs; activations from the store (CPU) ----
dirs = Directions(os.path.join(args.data_dir, "layer_stats.pt"), device=dev)
def load_rows(paths, split, store):
    files = sorted(sum((glob.glob(g) for g in paths.split(",")), [])); assert files, paths
    df = load_text_pairs(files, os.path.join(args.data_dir, f"pairs_{split}.parquet")); df = df[df["pos_idx"].isin(store.row_of)].reset_index(drop=True)
    if args.samples and "sample" in df: df = df[df["sample"].isin([int(x) for x in args.samples.split(",")])].reset_index(drop=True)
    return df
BAND = [int(x) for x in args.band.split(",")] if args.band else None
store = Store(args.data_dir, "train", device=dev, layers=BAND, verbose=is_main)          # on the GPU: 16 layers x 100k positions = 17 GB per rank, too much for CPU RAM x 4 ranks; df = load_rows(args.text, "train", store); df = df.iloc[RANK::WORLD].reset_index(drop=True)
P(f"[sft] {len(df)} train text rows per rank (sources {df['source'].value_counts().to_dict()}); eff batch {args.batch * args.grad_accum * WORLD}")
store_val = dfv = None
if args.val_text and is_main:
    store_val = Store(args.data_dir, "val", device=dev, layers=BAND, verbose=False); dfv = load_rows(args.val_text, "val", store_val).drop_duplicates("pair_id").iloc[: args.val_rows].reset_index(drop=True)
    P(f"[sft] {len(dfv)} val rows")

def make_batch(sub, st):
    rows = st.rows_for(sub["pos_idx"].values); i = torch.tensor(sub["i"].values.astype(np.int64)); j = torch.tensor(sub["j"].values.astype(np.int64))
    u_i = dirs.unit(st.gather(rows, i, dev), i); u_j = dirs.unit(st.gather(rows, j, dev), j); vec = torch.stack([u_i, u_j], 1)                       # [B, 2, d]
    tgt = [tok(str(t).strip(), add_special_tokens=False).input_ids[: args.max_len] + [EOT] for t in sub["text"].values]
    T = PLEN + max(len(t) for t in tgt); B = len(tgt)
    ids = torch.full((B, T), pad_id, dtype=torch.long); lab = torch.full((B, T), -100, dtype=torch.long); attn = torch.zeros((B, T), dtype=torch.long)
    for r, t in enumerate(tgt): ids[r, :PLEN] = PROMPT_T; ids[r, PLEN:PLEN + len(t)] = torch.tensor(t); lab[r, PLEN:PLEN + len(t)] = torch.tensor(t); attn[r, :PLEN + len(t)] = 1
    return ids.to(dev), attn.to(dev), lab.to(dev), vec
def ce_loss(sub, st):
    ids, attn, lab, vec = make_batch(sub, st); inj.set(vec, ids)
    try: out = model(input_ids=ids, attention_mask=attn, use_cache=False)
    finally: inj.off()
    lg = out.logits[:, :-1].float(); return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), lab[:, 1:].reshape(-1), ignore_index=-100)
@torch.no_grad()
def evaluate():
    if dfv is None: return float("nan"), []
    model.eval(); ces = [ce_loss(dfv.iloc[a:a + args.batch], store_val).item() for a in range(0, len(dfv), args.batch)]
    sub = dfv.iloc[:3]; ids, attn, lab, vec = make_batch(sub, store_val); ids = PROMPT_T[None].repeat(3, 1).to(dev); inj.set(vec, ids)
    try: g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=96, do_sample=False, pad_token_id=pad_id)
    finally: inj.off()
    model.train(); return float(np.mean(ces)), [(str(sub["text"].values[q])[:200], tok.decode(g[q, PLEN:], skip_special_tokens=True)) for q in range(3)]

trainable = [q for q in model.parameters() if q.requires_grad]
optim = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: min(1.0, s / max(1, args.warmup)) * max(0.05, 1 - s / args.steps))
wb = None
if is_main and not args.no_wandb and os.environ.get("WANDB_API_KEY"):
    import wandb; os.environ.setdefault("WANDB_DIR", "/root/wandb"); os.makedirs("/root/wandb", exist_ok=True)
    wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name or os.path.basename(args.out), config={**vars(args), "eff_batch": args.batch * args.grad_accum * WORLD, "gpus": WORLD})
t0 = time.time(); model.train(); rng = np.random.default_rng(args.seed + 1 + RANK); order = rng.permutation(len(df)); ptr = 0
for step in range(args.steps):
    optim.zero_grad(set_to_none=True); tot = 0.0
    for _ in range(args.grad_accum):
        if ptr + args.batch > len(order): order = rng.permutation(len(df)); ptr = 0
        idx = order[ptr:ptr + args.batch]; ptr += args.batch
        loss = ce_loss(df.iloc[idx], store); (loss / args.grad_accum).backward(); tot += loss.item() / args.grad_accum
    if is_dist:
        for q in trainable:
            if q.grad is None: q.grad = torch.zeros_like(q)
            dist.all_reduce(q.grad); q.grad.div_(WORLD)
    gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0); optim.step(); sched.step()
    if is_main:
        if step % 10 == 0: print(f"step {step:05d} | ce {tot:.4f} | gn {float(gn):.2f} | injected {inj.n_writes} | {(time.time() - t0) / 60:.1f} min", flush=True)
        log = {"step": step, "ce": tot, "lr": sched.get_last_lr()[0], "grad_norm": float(gn)}
        if step > 0 and step % args.eval_every == 0:
            ev, samples = evaluate(); log["eval_ce"] = ev; print(f"  [eval {step}] ce {ev:.4f}", flush=True)
            for tb_, ge in samples[:2]: print(f"    TARGET {tb_!r}\n    GEN    {ge!r}", flush=True)
        if wb: wb.log(log)
        if step > 0 and step % args.save_every == 0: model.save_pretrained(f"{args.out}/step_{step:06d}")
    if is_dist and step > 0 and step % args.eval_every == 0: dist.barrier()
if is_main:
    model.save_pretrained(f"{args.out}/final"); ev, samples = evaluate()
    json.dump({"args": vars(args), "final_eval_ce": ev, "prompt": tok.decode(PROMPT), "prompt_len": PLEN, "elapsed_min": (time.time() - t0) / 60, "world": WORLD}, open(f"{args.out}/meta.json", "w"), indent=1)
    print(f"SFT_DONE final_ce {ev:.4f}", flush=True)
if is_dist: dist.barrier(); dist.destroy_process_group()
