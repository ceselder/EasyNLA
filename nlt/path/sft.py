"""Controlled SFT of the PATH verbalizer (DECISIONS v1.26): same init (V0 LoRA), same rows (V0b's 20k), same hyper-parameters as V0b
(nlt.verbalizer.sft); only the INPUT differs: h_i, then the writes between i and j (one norm-matched marker each), then h_j.

  python -m nlt.path.sft --data-dir /vol/data/qwen3_8b --text /vol/z/v0b_mix/train/rows.parquet --val-text /vol/z/v0b_mix/val/rows.parquet \
      --init lora:/vol/rl/sft/v0_ao_tsv1/lora --path-mode delta --out /vol/rl/sft/v0b_path_d --tag v0b_path_d
  python -m nlt.path.sft ... --path-mode attn_mlp --path-dir /vol/path/qwen3_8b --out /vol/rl/sft/v0b_path --tag v0b_path
  python -m nlt.path.sft ... --path-mode none --init lora:/vol/rl/sft/v0b_mix/lora --eval-only --out /vol/rl/sft/v0b_evalonly   (V0b's held-out loss, same code path)

Row filtering is copied from nlt.verbalizer.sft (hard-regex hits, 4-gram copy > 0.05 and empty texts dropped) so the row SET is identical.
"""
from __future__ import annotations
import argparse, json, math, os, time
import numpy as np, torch, torch.nn.functional as F


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--out", required=True); p.add_argument("--tag", default="path_sft")
    p.add_argument("--text", required=True); p.add_argument("--val-text", default=None)
    p.add_argument("--base", default="Qwen/Qwen3-8B"); p.add_argument("--init", default="ao"); p.add_argument("--question", default=None)
    p.add_argument("--path-mode", default="delta", choices=["none", "count", "delta", "attn_mlp"]); p.add_argument("--path-dir", default="/vol/path/qwen3_8b"); p.add_argument("--fixed-markers", type=int, default=0, help="pad the middle with zero markers to this fixed count (count carries no gap info)"); p.add_argument("--rows-direct", action="store_true", help="text parquets already carry pos_idx/i/j (nlt.path.facts): skip the pairs join and the copy filter"); p.add_argument("--ablate-mid", default="none", choices=["none", "zero", "shuffle", "noise"], help="eval-time ablation of the middle vectors (diagnostic; applies to train too, so use with --eval-only)")
    p.add_argument("--epochs", type=float, default=1.0); p.add_argument("--lr", type=float, default=3e-5); p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--batch", type=int, default=32); p.add_argument("--micro", type=int, default=8); p.add_argument("--max-resp-tokens", type=int, default=96)
    p.add_argument("--max-rows", type=int, default=None); p.add_argument("--copy-thresh", type=float, default=0.05)
    p.add_argument("--verbosity", default=None); p.add_argument("--sources", default=None)
    p.add_argument("--lora-r", type=int, default=64); p.add_argument("--lora-alpha", type=int, default=16); p.add_argument("--train-store-device", default="cpu"); p.add_argument("--train-store-max-pos", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=25); p.add_argument("--save-every", type=int, default=200); p.add_argument("--val-rows", type=int, default=768); p.add_argument("--eval-only", action="store_true")
    p.add_argument("--wandb-project", default="nlt-qwen3-8b"); p.add_argument("--wandb-entity", default="octahedral-systems"); p.add_argument("--no-wandb", action="store_true"); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); os.makedirs(a.out, exist_ok=True)
    from nlt.data.dataset import ActStore
    from nlt.critic.train import load_text_pairs
    from nlt.evals.regex_tags import hard_hits
    from nlt.evals.copy_rate import copy_rate_ngram
    from nlt.verbalizer.prompt import response_ids
    from nlt.verbalizer.model import load_tokenizer, load_policy, save_adapter
    from nlt.path.prompt import build_path_prompt
    from nlt.path.inject import MultiMarkerInjector, pack_slot
    from nlt.path.vectors import PathStore, path_inputs, n_mid_of
    dev = "cuda"; tok = load_tokenizer(a.base); pad = tok.pad_token_id
    spec0 = build_path_prompt(tok, 0 if a.path_mode == "none" else 2, a.question); print(f"[path-sft] mode {a.path_mode}; example prompt ({spec0.n} tokens, markers at {spec0.positions}): {spec0.text!r}", flush=True)

    def load_rows(paths, split, store):      # identical to nlt.verbalizer.sft.load_rows -> identical row set
        import glob as _glob
        files = sorted(sum([_glob.glob(x) if any(c in x for c in "*?[") else [x] for x in paths.split(",")], []))
        assert files, f"no text files match {paths}"
        verb = [int(v) for v in a.verbosity.split(",")] if a.verbosity else None
        if a.rows_direct:
            import pandas as pd, pyarrow.parquet as pq
            df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True); df = df[df["pos_idx"].isin(store.row_of)]
            n0 = len(df); df = df[~df["text"].astype(str).map(lambda t: bool(hard_hits(t)))]
            print(f"[path-sft:{split}] direct rows {n0} -> regex {len(df)} used; sources {df['source'].value_counts().to_dict()}", flush=True)
            return df.reset_index(drop=True)
        df = load_text_pairs(files, os.path.join(a.data_dir, f"pairs_{split}.parquet"), verbosity=verb); df = df[df["pos_idx"].isin(store.row_of)]
        if a.sources: df = df[df["source"].isin(a.sources.split(","))]
        n0 = len(df); df = df[~df["text"].astype(str).map(lambda t: bool(hard_hits(t)))]; n1 = len(df)
        docs = store.load_docs(a.data_dir); keep = []
        if not docs: print(f"[path-sft:{split}] WARNING no docs parquet in {a.data_dir}/{split}: copy filter skipped", flush=True)
        for pos_idx, t in zip(df["pos_idx"].values, df["text"].values):
            ids = tok.encode(str(t).strip(), add_special_tokens=False)
            ok = len(ids) > 0 and (not docs or copy_rate_ngram(ids, store.context_ids(int(pos_idx), 256), 4) <= a.copy_thresh); keep.append(ok)
        df = df[np.array(keep, bool)]; n2 = len(df)
        print(f"[path-sft:{split}] rows {n0} -> regex {n1} -> copy/empty {n2} -> {len(df)} used; sources {df['source'].value_counts().to_dict()}", flush=True)
        store.docs = {}                     # free the docs (only needed for the copy filter)
        return df.reset_index(drop=True)

    store = None; df = None
    if not a.eval_only:
        store = ActStore(a.data_dir, "train", device=a.train_store_device, max_pos=a.train_store_max_pos); df = load_rows(a.text, "train", store)
        if a.max_rows: df = df.iloc[: a.max_rows]
    store_val = dfv = None
    if a.val_text:
        store_val = ActStore(a.data_dir, "val", device="cpu"); dfv = load_rows(a.val_text, "val", store_val).iloc[: a.val_rows]
    pstore = pstore_val = None
    if a.path_mode == "attn_mlp":
        if df is not None: pstore = PathStore(a.path_dir, "train", pos_idx_needed=df["pos_idx"].values)
        if dfv is not None: pstore_val = PathStore(a.path_dir, "val", pos_idx_needed=dfv["pos_idx"].values)
        if df is not None:
            ok = df["pos_idx"].map(lambda q: int(q) in pstore.row_of).values; print(f"[path-sft] train rows with path vectors: {ok.sum()}/{len(df)}", flush=True); df = df[ok].reset_index(drop=True)
        if dfv is not None:
            ok = dfv["pos_idx"].map(lambda q: int(q) in pstore_val.row_of).values; print(f"[path-sft] val rows with path vectors: {ok.sum()}/{len(dfv)}", flush=True); dfv = dfv[ok].reset_index(drop=True)
    policy = load_policy(a.base, a.init, r=a.lora_r, alpha=a.lora_alpha, device=dev); policy.train()
    inj = MultiMarkerInjector(policy, spec0.marker_id)
    params = [q for q in policy.parameters() if q.requires_grad]; opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    n_steps = 0 if a.eval_only else int(math.ceil(len(df) * a.epochs / a.batch)); print(f"[path-sft] {0 if df is None else len(df)} rows, {n_steps} steps of {a.batch}", flush=True)
    run = None
    if not a.no_wandb and not a.eval_only:
        import wandb
        run = wandb.init(project=a.wandb_project, entity=a.wandb_entity, name=f"sft_{a.tag}", group="sft", config=vars(a))

    abl_gen = torch.Generator().manual_seed(a.seed + 12345)

    def batch_tensors(sub, st, ps):
        vecs = path_inputs(st, sub["pos_idx"].values, sub["i"].values, sub["j"].values, a.path_mode, ps, a.fixed_markers, a.ablate_mid, abl_gen)
        seqs, poss, plens = [], [], []
        for b, (i_, j_, t) in enumerate(zip(sub["i"].values, sub["j"].values, sub["text"].values)):
            sp = build_path_prompt(tok, n_mid_of(i_, j_, a.path_mode, a.fixed_markers), a.question); assert len(sp.positions) == vecs[b].shape[0]
            seqs.append(torch.tensor(sp.ids + response_ids(tok, str(t), a.max_resp_tokens))); poss.append(sp.positions); plens.append(sp.n)
        L = max(s.numel() for s in seqs); ids = torch.full((len(seqs), L), pad, dtype=torch.long); lab = torch.full((len(seqs), L), -100, dtype=torch.long); am = torch.zeros_like(ids)
        for r, s in enumerate(seqs): ids[r, : s.numel()] = s; am[r, : s.numel()] = 1; lab[r, plens[r]: s.numel()] = s[plens[r]:]
        return ids, am, lab, pack_slot(vecs, poss)

    def loss_on(ids, am, lab, slot, grad=True):
        ids, am, lab = ids.to(dev), am.to(dev), lab.to(dev); inj.ref[0] = (slot[0].to(dev), slot[1].to(dev)); inj.reset_count()
        try:
            ctx = torch.enable_grad() if grad else torch.no_grad()
            with ctx:
                logits = policy(input_ids=ids, attention_mask=am, use_cache=False).logits[:, :-1].float()
                l = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
        finally:
            inj.ref[0] = None
        assert inj.n_writes == int((slot[1] >= 0).sum()), f"marker writes {inj.n_writes} != {int((slot[1] >= 0).sum())}"
        return l, int((lab[:, 1:] != -100).sum())

    @torch.no_grad()
    def evaluate():
        if dfv is None: return {}
        policy.eval(); tot = 0.0; ntk = 0; by_src = {}
        for s in range(0, len(dfv), a.micro):
            sub = dfv.iloc[s: s + a.micro]; ids, am, lab, slot = batch_tensors(sub, store_val, pstore_val); l, k = loss_on(ids, am, lab, slot, grad=False); tot += float(l); ntk += k
            for src in set(sub["source"]):        # per-source loss (cheap second pass per source group)
                ss = sub[sub["source"] == src]; ids2, am2, lab2, slot2 = batch_tensors(ss, store_val, pstore_val); l2, k2 = loss_on(ids2, am2, lab2, slot2, grad=False)
                acc = by_src.setdefault(src, [0.0, 0]); acc[0] += float(l2); acc[1] += k2
        policy.train(); out = {"val/loss_per_token": tot / max(1, ntk), "val/tokens": ntk, "val/rows": len(dfv)}
        for src, (l_, k_) in by_src.items(): out[f"val/loss_per_token_{src}"] = l_ / max(1, k_)
        return out

    if a.eval_only:
        ev = evaluate(); ev.update({"init": a.init, "path_mode": a.path_mode, "val_text": a.val_text, "ablate_mid": a.ablate_mid, "fixed_markers": a.fixed_markers}); print("[path-sft] EVAL-ONLY " + json.dumps(ev), flush=True)
        json.dump(ev, open(os.path.join(a.out, "eval.json"), "w"), indent=1); return
    order = np.random.default_rng(a.seed).permutation(len(df)); ptr = 0; t_start = time.time(); last_eval = {}
    for step in range(n_steps):
        for g in opt.param_groups: g["lr"] = a.lr * min(1.0, (step + 1) / max(1, a.warmup)) * (0.5 * (1 + math.cos(math.pi * step / max(1, n_steps))) if step >= a.warmup else 1.0)
        if ptr + a.batch > len(order): order = np.random.default_rng(a.seed + step).permutation(len(df)); ptr = 0
        idx = order[ptr: ptr + a.batch]; ptr += a.batch; sub = df.iloc[idx]
        opt.zero_grad(set_to_none=True); tot = 0.0; ntk = 0
        for s in range(0, len(sub), a.micro):
            ids, am, lab, slot = batch_tensors(sub.iloc[s: s + a.micro], store, pstore); l, k = loss_on(ids, am, lab, slot); tot += float(l.detach()); ntk += k
            (l / max(1, k) * (len(ids) / len(sub))).backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        if math.isfinite(gn): opt.step()
        log = {"step": step, "loss_per_token": tot / max(1, ntk), "tokens": ntk, "grad_norm": gn, "lr": opt.param_groups[0]["lr"], "time": time.time() - t_start}
        if step % a.eval_every == 0 or step == n_steps - 1:
            last_eval = evaluate(); log.update(last_eval)
            print(f"path-sft {step:5d}/{n_steps} | loss/tok {log['loss_per_token']:.4f} | val {log.get('val/loss_per_token', float('nan')):.4f} | gn {gn:.2f} | {log['time']:.0f}s", flush=True)
        elif step % 10 == 0: print(f"path-sft {step:5d}/{n_steps} | loss/tok {log['loss_per_token']:.4f} | gn {gn:.2f} | {log['time']:.0f}s", flush=True)
        if run is not None: run.log(log, step=step)
        if (step + 1) % a.save_every == 0 or step == n_steps - 1:
            save_adapter(policy, os.path.join(a.out, "lora"))
            json.dump({"step": step + 1, "rows": len(df), "init": a.init, "path_mode": a.path_mode, "question": a.question, "example_prompt": spec0.text, "text": a.text, "last_eval": last_eval, "args": vars(a)},
                      open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    if run is not None: run.finish()
    print(f"done -> {a.out}/lora; final eval {json.dumps(last_eval)}", flush=True)


if __name__ == "__main__":
    main()
