"""SFT of the two-marker verbalizer on critic-selected (pair_id, text) rows (DECISIONS D4: V0 = AO init + SFT on the per-pair winners).

  python -m nlt.verbalizer.sft --data-dir /vol/data/qwen3_8b --text /vol/z/pool_v1/train_winners.parquet --init ao --out /vol/rl/sft/v0 --tag v0

Text rows follow the board #6/#18 interface ([pair_id, text, verbosity, source, sample_idx]); they are joined to pairs_train.parquet on
pair_id (nlt.critic.train.load_text_pairs). Hard-regex hits, 4-gram copy > 0.05 and empty texts are dropped at load time (no verbatim
past, ever). Loss = CE on the response tokens only, with h_i, h_j injected at the two markers. Compare --init ao vs base on the same rows.
"""
from __future__ import annotations
import argparse, json, math, os, time
import numpy as np, torch, torch.nn.functional as F


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--out", required=True); p.add_argument("--tag", default="sft")
    p.add_argument("--text", required=True, help="comma list of text files (train split)"); p.add_argument("--val-text", default=None, help="comma list (val split)")
    p.add_argument("--base", default="Qwen/Qwen3-8B"); p.add_argument("--init", default="ao"); p.add_argument("--question", default=None)
    p.add_argument("--epochs", type=float, default=1.0); p.add_argument("--lr", type=float, default=3e-5); p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--batch", type=int, default=32); p.add_argument("--micro", type=int, default=8); p.add_argument("--max-resp-tokens", type=int, default=96)
    p.add_argument("--max-rows", type=int, default=None); p.add_argument("--one-per-pair", action="store_true"); p.add_argument("--copy-thresh", type=float, default=0.05)
    p.add_argument("--lora-r", type=int, default=64); p.add_argument("--lora-alpha", type=int, default=16); p.add_argument("--train-store-device", default="cpu")
    p.add_argument("--eval-every", type=int, default=50); p.add_argument("--save-every", type=int, default=200); p.add_argument("--val-rows", type=int, default=512)
    p.add_argument("--wandb-project", default="nlt-qwen3-8b"); p.add_argument("--wandb-entity", default="octahedral-systems"); p.add_argument("--no-wandb", action="store_true"); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); os.makedirs(a.out, exist_ok=True)
    import wandb
    from nlt.data.dataset import ActStore
    from nlt.critic.train import load_text_pairs
    from nlt.evals.regex_tags import hard_hits
    from nlt.evals.copy_rate import copy_rate_ngram
    from nlt.verbalizer.prompt import build_prompt, response_ids, DEFAULT_QUESTION
    from nlt.verbalizer.model import load_tokenizer, load_policy, save_adapter
    from nlt.verbalizer.inject import TwoMarkerInjector
    dev = "cuda"; tok = load_tokenizer(a.base); spec = build_prompt(tok, a.question or DEFAULT_QUESTION); pad = tok.pad_token_id

    def load_rows(paths, split, store):
        df = load_text_pairs(paths.split(","), os.path.join(a.data_dir, f"pairs_{split}.parquet")); df = df[df["pos_idx"].isin(store.row_of)]
        n0 = len(df); df = df[~df["text"].astype(str).map(lambda t: bool(hard_hits(t)))]; n1 = len(df)
        docs = store.load_docs(a.data_dir); keep = []
        if not docs: print(f"[sft:{split}] WARNING no docs parquet in {a.data_dir}/{split}: copy filter skipped", flush=True)
        for pos_idx, t in zip(df["pos_idx"].values, df["text"].values):
            ids = tok.encode(str(t).strip(), add_special_tokens=False)
            ok = len(ids) > 0 and (not docs or copy_rate_ngram(ids, store.context_ids(int(pos_idx), 256), 4) <= a.copy_thresh); keep.append(ok)
        df = df[np.array(keep, bool)]; n2 = len(df)
        if a.one_per_pair: df = df.sample(frac=1.0, random_state=a.seed).drop_duplicates("pair_id")
        print(f"[sft:{split}] rows {n0} -> regex {n1} -> copy/empty {n2} -> {len(df)} used; sources {df['source'].value_counts().to_dict()}", flush=True)
        return df.reset_index(drop=True)

    store = ActStore(a.data_dir, "train", device=a.train_store_device); df = load_rows(a.text, "train", store)
    if a.max_rows: df = df.iloc[: a.max_rows]
    store_val = dfv = None
    if a.val_text:
        store_val = ActStore(a.data_dir, "val", device="cpu"); dfv = load_rows(a.val_text, "val", store_val).iloc[: a.val_rows]
    policy = load_policy(a.base, a.init, r=a.lora_r, alpha=a.lora_alpha, device=dev); policy.train(); inj = TwoMarkerInjector(policy, spec.marker_id)
    params = [q for q in policy.parameters() if q.requires_grad]; opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    n_steps = int(math.ceil(len(df) * a.epochs / a.batch)); print(f"[sft] {len(df)} rows, {n_steps} steps of {a.batch}", flush=True)
    run = None if a.no_wandb else wandb.init(project=a.wandb_project, entity=a.wandb_entity, name=f"sft_{a.tag}", group="sft", config=vars(a))

    def batch_tensors(sub, st):
        rows = st.rows_for(sub["pos_idx"].values); acts = torch.stack([st.gather(rows, torch.as_tensor(sub["i"].values).long()), st.gather(rows, torch.as_tensor(sub["j"].values).long())], 1).float()
        seqs = [torch.tensor(spec.ids + response_ids(tok, str(t), a.max_resp_tokens)) for t in sub["text"].values]
        L = max(s.numel() for s in seqs); ids = torch.full((len(seqs), L), pad, dtype=torch.long); lab = torch.full((len(seqs), L), -100, dtype=torch.long); am = torch.zeros_like(ids)
        for r, s in enumerate(seqs): ids[r, : s.numel()] = s; am[r, : s.numel()] = 1; lab[r, spec.n: s.numel()] = s[spec.n:]
        return ids, am, lab, acts

    def loss_on(ids, am, lab, acts, grad=True):
        ids, am, lab = ids.to(dev), am.to(dev), lab.to(dev); inj.ref[0] = acts.to(dev)
        try:
            ctx = torch.enable_grad() if grad else torch.no_grad()
            with ctx:
                logits = policy(input_ids=ids, attention_mask=am, use_cache=False).logits[:, :-1].float()
                l = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
        finally:
            inj.ref[0] = None
        return l, int((lab[:, 1:] != -100).sum())

    @torch.no_grad()
    def evaluate():
        if dfv is None: return {}
        policy.eval(); tot = 0.0; ntk = 0
        for s in range(0, len(dfv), a.micro):
            ids, am, lab, acts = batch_tensors(dfv.iloc[s: s + a.micro], store_val); l, k = loss_on(ids, am, lab, acts, grad=False); tot += float(l); ntk += k
        policy.train(); return {"val/loss_per_token": tot / max(1, ntk), "val/tokens": ntk}

    order = np.random.default_rng(a.seed).permutation(len(df)); ptr = 0; t_start = time.time()
    for step in range(n_steps):
        for g in opt.param_groups: g["lr"] = a.lr * min(1.0, (step + 1) / max(1, a.warmup)) * (0.5 * (1 + math.cos(math.pi * step / max(1, n_steps))) if step >= a.warmup else 1.0)
        if ptr + a.batch > len(order): order = np.random.default_rng(a.seed + step).permutation(len(df)); ptr = 0
        idx = order[ptr: ptr + a.batch]; ptr += a.batch; sub = df.iloc[idx]
        opt.zero_grad(set_to_none=True); tot = 0.0; ntk = 0
        for s in range(0, len(sub), a.micro):
            ids, am, lab, acts = batch_tensors(sub.iloc[s: s + a.micro], store); l, k = loss_on(ids, am, lab, acts); tot += float(l); ntk += k
            (l / max(1, len(sub))).backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        if math.isfinite(gn): opt.step()
        log = {"step": step, "loss_per_token": tot / max(1, ntk), "tokens": ntk, "grad_norm": gn, "lr": opt.param_groups[0]["lr"], "time": time.time() - t_start}
        if step % a.eval_every == 0 or step == n_steps - 1:
            log.update(evaluate()); print(f"sft {step:5d}/{n_steps} | loss/tok {log['loss_per_token']:.4f} | val {log.get('val/loss_per_token', float('nan')):.4f} | gn {gn:.2f} | {log['time']:.0f}s", flush=True)
        if run is not None: run.log(log, step=step)
        if (step + 1) % a.save_every == 0 or step == n_steps - 1:
            save_adapter(policy, os.path.join(a.out, "lora")); json.dump({"step": step + 1, "rows": len(df), "init": a.init, "prompt": spec.text, "text": a.text}, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    if run is not None: run.finish()
    print(f"done -> {a.out}/lora", flush=True)


if __name__ == "__main__":
    main()
