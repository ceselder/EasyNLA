"""Dump the two-marker change verbalizer on held-out pairs (HF generate, greedy by default) -> parquet [pair_id, text, source] in the critic's text-set
format, so eval_bits.py scores the verbalizer next to the crafted teacher text on the SAME pairs.

  python dump_verbalizer.py --data-dir /vol/q36/data --adapter /vol/q36/verbalizer/v1/final --pairs-text '/vol/q36/text/v1/val/craft_full__*.parquet' \
      --n 1024 --out /vol/q36/dumps/verbalizer_v1.parquet [--sample --temperature 1.0]
--pairs-text restricts the dumped pairs to those that have a teacher text (paired comparison); without it the first --n val pairs are used.
"""
import argparse, glob, json, os, time
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, torch
from peft import PeftModel
from common import InjectMarkers, change_prompt, load_base, load_tokenizer
from critic_data import Store, Directions, load_text_pairs

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True); ap.add_argument("--adapter", required=True); ap.add_argument("--out", required=True); ap.add_argument("--split", default="val")
ap.add_argument("--pairs-text", default=None); ap.add_argument("--n", type=int, default=1024); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--max-new", type=int, default=112)
ap.add_argument("--band", default=None); ap.add_argument("--sample", action="store_true"); ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--source", default="verbalizer"); ap.add_argument("--base-only", action="store_true", help="no adapter: the base model with the injected markers (control)")
args = ap.parse_args(); dev = "cuda"; t0 = time.time(); tok = load_tokenizer(); pad_id = tok.eos_token_id
PROMPT = change_prompt(tok); PLEN = len(PROMPT); PROMPT_T = torch.tensor(PROMPT, dtype=torch.long, device=dev)
model = load_base(dev)
if not args.base_only: model = PeftModel.from_pretrained(model, args.adapter); model.eval()
inj = InjectMarkers(model); dirs = Directions(os.path.join(args.data_dir, "layer_stats.pt"), device=dev); store = Store(args.data_dir, args.split, device="cpu", layers=[int(x) for x in args.band.split(",")] if args.band else None)
vp = pq.read_table(os.path.join(args.data_dir, f"pairs_{args.split}.parquet"), columns=["pair_id", "pos_idx", "i", "j"]).to_pandas(); vp = vp[vp["pos_idx"].isin(store.row_of)]
if args.pairs_text:
    have = set(load_text_pairs(sorted(sum((glob.glob(g) for g in args.pairs_text.split(",")), [])), os.path.join(args.data_dir, f"pairs_{args.split}.parquet"))["pair_id"]); vp = vp[vp["pair_id"].isin(have)]
vp = vp.iloc[: args.n].reset_index(drop=True); n = len(vp); print(f"[dump] {n} pairs, adapter {args.adapter if not args.base_only else 'NONE (base)'}", flush=True)
texts = []
for s in range(0, n, args.batch):
    sub = vp.iloc[s:s + args.batch]; B = len(sub); rows = store.rows_for(sub["pos_idx"].values); i = torch.tensor(sub["i"].values.astype(np.int64)); j = torch.tensor(sub["j"].values.astype(np.int64))
    vec = torch.stack([dirs.unit(store.gather(rows, i, dev), i), dirs.unit(store.gather(rows, j, dev), j)], 1); ids = PROMPT_T[None].repeat(B, 1); inj.set(vec, ids)
    with torch.no_grad():
        try: g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=args.max_new, do_sample=args.sample, temperature=args.temperature if args.sample else None, top_p=1.0 if args.sample else None, top_k=0 if args.sample else None, pad_token_id=pad_id)
        finally: inj.off()
    texts += [tok.decode(g[q, PLEN:], skip_special_tokens=True).strip() for q in range(B)]
    if (s // args.batch) % 8 == 0: print(f"[dump] {min(n, s + B)}/{n} | {(time.time() - t0) / 60:.1f} min | e.g. {texts[-1][:160]!r}", flush=True)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
pq.write_table(pa.table({"pair_id": vp["pair_id"].tolist(), "text": texts, "source": [args.source] * n, "sample": pa.array([0] * n, pa.int32())}), args.out)
ntok = [len(tok(t, add_special_tokens=False).input_ids) for t in texts]
json.dump({"n": n, "adapter": args.adapter, "mean_tokens": float(np.mean(ntok)), "empty": int(sum(1 for t in texts if not t)), "elapsed_min": (time.time() - t0) / 60, "examples": texts[:8]}, open(args.out.replace(".parquet", "_meta.json"), "w"), indent=1, ensure_ascii=False)
print(f"DUMP_DONE {args.out} n={n} mean_tokens={np.mean(ntok):.1f}", flush=True)
