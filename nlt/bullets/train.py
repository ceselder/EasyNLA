"""Train the reconstructor R(h_i, text) -> delta_hat on one text source, with BULLET DROPOUT (each bullet dropped w.p. --p-drop) and
--p-empty rows with empty text (the same net is then the h_i-only baseline and leave-one-out stays in distribution).

  python -m nlt.bullets.train --data-dir /vol/data/qwen3_8b --text '/vol/z/bullets-sonnet-v1/train/part_*.parquet' \
      --val-text '/vol/z/bullets-sonnet-v1/val/part_*.parquet' --out /vol/bullets/R_bullets --tag R_bullets [--pair-ids ids.json]

Loss = mean_i ||dh_i - d_i||^2 / ||d_i||^2 + cos_w (1 - cos)  (delta in the pooled-normalised space; scale-free, j-agnostic).
Writes ckpt_final.pt, curve.json (train loss, val FVE(all) / FVE(empty) every --eval-every), train_pairs.json (the pair ids used).
"""
from __future__ import annotations
import argparse, json, math, os, time
import numpy as np, torch
from nlt.data.dataset import GlobalNorm
from nlt.bullets.data import load_pairs, join_text, gather_acts, join_bullets
from nlt.bullets import model as M


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--out", required=True); p.add_argument("--tag", default="R")
    p.add_argument("--text", required=True); p.add_argument("--val-text", required=True)
    p.add_argument("--verbosity", default=None, help="comma list of verbosity levels to keep"); p.add_argument("--val-verbosity", default=None)
    p.add_argument("--pair-ids", default=None, help="json list of train pair_ids to restrict to (matched runs)"); p.add_argument("--n-val", type=int, default=1536)
    p.add_argument("--max-train", type=int, default=None); p.add_argument("--epochs", type=float, default=0, help="if > 0, steps = epochs * n_train / batch")
    p.add_argument("--freeze-lora", action="store_true", help="frozen text encoder (no LoRA): small-data regime"); p.add_argument("--bottleneck", type=int, default=0); p.add_argument("--text-dropout", type=float, default=0.0)
    p.add_argument("--select-rows", default=None, help="a:b val rows used ONLY to pick ckpt_best.pt (report on the other rows)")
    p.add_argument("--init-from", default=None, help="warm start from a reconstructor checkpoint (architecture args are taken from it)")
    p.add_argument("--steps", type=int, default=2000); p.add_argument("--batch", type=int, default=64); p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--lr-lora", type=float, default=1e-4); p.add_argument("--lr-head", type=float, default=5e-4); p.add_argument("--wd", type=float, default=0.01); p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--p-drop", type=float, default=0.3); p.add_argument("--p-empty", type=float, default=0.1); p.add_argument("--cos-w", type=float, default=0.5)
    p.add_argument("--enc-model", default="Qwen/Qwen3-0.6B"); p.add_argument("--enc-layer", type=int, default=20); p.add_argument("--lora-r", type=int, default=64); p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--max-len", type=int, default=320); p.add_argument("--d-model", type=int, default=1024); p.add_argument("--n-q", type=int, default=4); p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--hidden", type=int, default=4096); p.add_argument("--n-hidden", type=int, default=2)
    p.add_argument("--wandb", default="nlt-qwen3-8b"); p.add_argument("--wandb-entity", default="octahedral-systems"); p.add_argument("--no-wandb", action="store_true"); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = "cuda"; os.makedirs(a.out, exist_ok=True)
    norm = GlobalNorm.load(os.path.join(a.data_dir, "stats.pt"), "affine").to(dev)

    verb = [int(v) for v in a.verbosity.split(",")] if a.verbosity else None
    vverb = [int(v) for v in a.val_verbosity.split(",")] if a.val_verbosity else verb
    tr = join_text(a.text, load_pairs(a.data_dir, "train"), verbosity=verb)
    if a.pair_ids:
        ids = set(json.load(open(a.pair_ids))); tr = tr[tr["pair_id"].isin(ids)].reset_index(drop=True)
    if a.max_train:
        tr = tr.iloc[: a.max_train].reset_index(drop=True)
    pv = load_pairs(a.data_dir, "val").iloc[: a.n_val]
    va = join_text(a.val_text, pv, verbosity=vverb)
    print(f"[train] {len(tr)} train rows ({tr['n_bullets'].mean():.1f} bullets, {tr['n_tokens'].mean() if 'n_tokens' in tr else -1:.0f} tokens avg); {len(va)} val rows of the first {a.n_val} fixed val pairs", flush=True)
    json.dump(sorted(tr["pair_id"].tolist()), open(os.path.join(a.out, "train_pairs.json"), "w"))
    t0 = time.time()
    Htr = gather_acts(a.data_dir, "train", tr["pos_idx"].values, tr["i"].values, tr["j"].values)
    Hva = gather_acts(a.data_dir, "val", va["pos_idx"].values, va["i"].values, va["j"].values)
    print(f"[train] gathered activations in {time.time() - t0:.0f}s", flush=True)
    Dtr = norm.normalize(Htr["h_j"].to(dev)) - norm.normalize(Htr["h_i"].to(dev)); Xtr = norm.normalize(Htr["h_i"].to(dev))
    Dva = norm.normalize(Hva["h_j"].to(dev)) - norm.normalize(Hva["h_i"].to(dev)); Xva = norm.normalize(Hva["h_i"].to(dev))
    del Htr, Hva

    if a.epochs > 0: a.steps = max(50, int(round(a.epochs * len(tr) / a.batch)))
    sel = None
    if a.select_rows:
        lo, hi = [int(x) for x in a.select_rows.split(":")]; sel_ids = set(pv["pair_id"].iloc[lo:hi]); sel = torch.as_tensor(va["pair_id"].isin(sel_ids).values, device=dev)
        print(f"[train] selection rows {lo}:{hi} -> {int(sel.sum())} val rows for ckpt_best; {int((~sel).sum())} rows left for the report", flush=True)
    print(f"[train] steps {a.steps} (batch {a.batch}, {a.steps * a.batch / max(1, len(tr)):.1f} epochs)", flush=True)
    if a.init_from:
        model, margs = M.load(a.init_from, dev); model.train()
        for k in ("enc_model", "enc_layer", "lora_r", "lora_alpha", "max_len", "d_model", "n_q", "n_heads", "hidden", "n_hidden", "bottleneck", "text_dropout", "freeze_lora"):
            if k in margs: setattr(a, k, margs[k])
        print(f"[train] warm start from {a.init_from} (step {margs.get('best_step', 'final')})", flush=True)
    else:
        model = M.build(a, dev); model.train()
    opt = torch.optim.AdamW(M.trainable_groups(model, a.lr_lora, a.lr_head, a.wd), betas=(0.9, 0.95))
    base_lrs = [g["lr"] for g in opt.param_groups]
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad); print(f"[train] trainable params {n_tr / 1e6:.1f}M", flush=True)
    run = None
    if not a.no_wandb:
        try:
            import wandb; run = wandb.init(project=a.wandb, entity=a.wandb_entity, name=f"bullets_{a.tag}", group="bullets", config=vars(a))
        except Exception as e:
            print(f"[train] wandb off: {str(e)[:100]}", flush=True)
    bullets_tr = tr["bullets"].tolist(); bullets_va = va["bullets"].tolist()

    def augment(bl):
        if rng.random() < a.p_empty or len(bl) == 0:
            return ""
        keep = [b for b in bl if rng.random() >= a.p_drop]
        if not keep:
            keep = [bl[rng.integers(len(bl))]]
        return join_bullets(keep)

    @torch.no_grad()
    def evaluate():
        model.eval(); B = 128; se_all = []; se_emp = []; en = []
        for s in range(0, len(va), B):
            x = Xva[s:s + B]; d = Dva[s:s + B]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p_all = model(x, [join_bullets(b) for b in bullets_va[s:s + B]]).float(); p_emp = model(x, [""] * len(x)).float()
            se_all.append(((p_all - d) ** 2).sum(-1)); se_emp.append(((p_emp - d) ** 2).sum(-1)); en.append((d ** 2).sum(-1))
        se_all, se_emp, en = map(torch.cat, (se_all, se_emp, en)); model.train()
        def blk(m, pre):
            return {f"{pre}/fve_all": float(1 - se_all[m].sum() / en[m].sum()), f"{pre}/fve_empty": float(1 - se_emp[m].sum() / en[m].sum()),
                    f"{pre}/relmse_all": float((se_all[m] / en[m]).mean()), f"{pre}/relmse_empty": float((se_emp[m] / en[m]).mean()),
                    f"{pre}/gain_per_ex": float(((se_emp[m] - se_all[m]) / en[m]).mean()), f"{pre}/p_text_beats_empty": float((se_all[m] < se_emp[m]).float().mean())}
        out = blk(torch.ones_like(en, dtype=torch.bool), "val")
        if sel is not None: out.update(blk(sel, "sel")); out.update(blk(~sel, "rep"))
        return out

    curve = []; t0 = time.time(); N = len(tr); best = float("inf"); best_step = 0
    for step in range(a.steps):
        f = min(1.0, (step + 1) / max(1, a.warmup)) * (0.5 * (1 + math.cos(math.pi * step / a.steps)) * 0.95 + 0.05)
        for g, b in zip(opt.param_groups, base_lrs): g["lr"] = b * f
        idx = rng.integers(0, N, a.batch); x = Xtr[idx]; d = Dtr[idx]; texts = [augment(bullets_tr[k]) for k in idx]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x, texts).float()
        loss, rm, cos = M.recon_loss(pred, d, a.cos_w)
        opt.zero_grad(set_to_none=True); loss.backward(); gn = float(torch.nn.utils.clip_grad_norm_([q for g in opt.param_groups for q in g["params"]], 1.0)); opt.step()
        log = {"train/loss": float(loss.detach()), "train/relmse": float(rm.detach().mean()), "train/cos": float(cos.detach().mean()), "train/grad_norm": gn, "train/lr_head": opt.param_groups[-1]["lr"]}
        if step % 50 == 0:
            print(f"[train] {step}/{a.steps} loss {log['train/loss']:.4f} relmse {log['train/relmse']:.4f} cos {log['train/cos']:.3f} gn {gn:.2f} {(time.time() - t0) / (step + 1):.2f}s/step", flush=True)
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            ev = evaluate(); log.update(ev); curve.append({"step": step + 1, **log}); print(f"[eval@{step + 1}] {json.dumps({k: round(v, 4) for k, v in ev.items()})}", flush=True)
            json.dump(curve, open(os.path.join(a.out, "curve.json"), "w"), indent=1)
            crit = ev.get("sel/relmse_all", ev["val/relmse_all"])
            if crit < best: best, best_step = crit, step + 1; M.save(model, os.path.join(a.out, "ckpt_best.pt"), vars(a) | {"best_step": best_step}); print(f"[train] ckpt_best <- step {best_step} ({crit:.4f})", flush=True)
        if run is not None: run.log(log, step=step)
    M.save(model, os.path.join(a.out, "ckpt_final.pt"), vars(a))
    json.dump({"args": vars(a), "final": curve[-1] if curve else {}, "best_step": best_step, "best_crit": best, "n_train": N, "n_val": len(va), "seconds": time.time() - t0}, open(os.path.join(a.out, "train_summary.json"), "w"), indent=1)
    if run is not None: run.finish()
    print(f"[train] DONE -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
