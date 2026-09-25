"""SMOKE TEST of the RL v6 per-bullet credit (orchestrator 2026-09-25 15:50): does the leave-one-bullet-out FM gain have the right SIGN on the teacher?
For N twinsL pairs (teacher 'true' text and its twin_new / twin_shift): gain(b) = reward(full) - reward(text without bullet b) under the judge, same noise, with the reward's t grid
(v6: 0.5/0.7/0.9 x K draws, weighted) and, for comparison, the uniform 5-t grid. Checks: (i) mean gain of true bullets > 0 (every teacher bullet helps); (ii) P(gain(original bullet in
the true text) > gain(swapped-in bullet in the twin text)) - the swapped slot is found as the differing item - with position-bootstrap CIs; (iii) fraction of true bullets with gain > 0.
  python bullet_credit_smoke.py --data-dir /vol/q36/data --ckpt /vol/q36/critic/v5/ckpt_step000500.pt --twins '...' --pair-ids-file /vol/q36/twinsL/pairs.txt --n 256 --out /vol/q36/results/bullet_smoke_v5s500.json
"""
import argparse, glob, json, math, os, sys, time

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from critic_data import Store, Directions

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--twins", required=True); ap.add_argument("--pair-ids-file", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=256); ap.add_argument("--t-grid", default="0.5,0.7,0.9"); ap.add_argument("--t-weights", default="0.28,0.37,0.36"); ap.add_argument("--eps-draws", type=int, default=4); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--batch", type=int, default=64)
a = ap.parse_args(); dev = "cuda"; torch.manual_seed(a.seed); T0 = time.time()
from eval_bits import load_critic
from nlt.critic.text_encoder import TextEncoder
model, aa, step = load_critic(a.ckpt, dev); sigma_r = float(aa.get("sigma_r", 0.1))
dirs = Directions(aa.get("stats_path") or os.path.join(a.data_dir, "layer_stats.pt"), sigma_r, dev, radial=aa.get("radial", "lognormal"), sigma_iso=float(aa.get("sigma_iso", 0.0)))
store = Store(a.data_dir, "val", device=dev); d = store.d; encoder = TextEncoder(aa.get("enc_model", "Qwen/Qwen3-0.6B"), int(aa.get("enc_layer", 20)), dev, int(aa.get("enc_max_len", 192)))
ids = [l.strip() for l in open(a.pair_ids_file) if l.strip()][: a.n]; parts = [x.split(":") for x in ids]
pos = torch.tensor([int(x[1]) for x in parts]); I = torch.tensor([int(x[2]) for x in parts]); J = torch.tensor([int(x[3]) for x in parts]); rows = store.rows_for(pos.numpy())
tw = __import__("pandas").concat([pq.read_table(f).to_pandas() for pat in a.twins.split(";") for f in sorted(glob.glob(pat))], ignore_index=True); tw = tw[tw["pair_id"].isin(set(ids))]
T = {(r.pair_id, r.variant): r.text for r in tw.itertuples()}
g = torch.Generator().manual_seed(a.seed + 1); S = torch.exp(sigma_r * torch.randn(len(ids), generator=g)); ISO = torch.randn(len(ids), d, generator=g)
def inputs(kk):
    r = rows[kk]; i = I[kk]; j = J[kk]; h_i = store.gather(r, i, dev); h_j = store.gather(r, j, dev); src = dirs.source(h_i, i); x0, _ = dirs.target(h_j, j, s=S[kk].to(dev), eps_iso=ISO[kk].to(dev)); return x0, src

LINES = ["Now present", "Faded", "Shift"]
def bullets(raw):
    out = []; pos_ = 0
    for ln in raw.split("\n"):
        for lab in LINES:
            if ln.startswith(lab + ":"):
                body_start = pos_ + len(lab) + 1; body = ln[len(lab) + 1:]; c = 0
                for item in body.split(";"):
                    st = c; en = c + len(item); it = item.rstrip("."); lead = len(item) - len(item.lstrip()); trail = len(it) - len(it.rstrip())
                    if it.strip(): out.append((body_start + st + lead, body_start + en - (len(item) - len(it)) - trail, lab))
                    c = en + 1
                break
        pos_ += len(ln) + 1
    return out
def without(raw, s, e):
    if raw[e:e + 1] == ";": e2 = e + 1; e2 += (raw[e2:e2 + 1] == " "); return raw[:s] + raw[e2:]
    k = raw.rfind(";", 0, s)
    if k >= 0 and raw.rfind("\n", 0, s) < k: return raw[:k] + raw[e:]
    ls = raw.rfind("\n", 0, s) + 1; le = raw.find("\n", e); le = len(raw) if le < 0 else le + 1; return raw[:ls] + raw[le:]

GRIDS = {"v6": ([float(x) for x in a.t_grid.split(",")], [float(x) for x in a.t_weights.split(",")], a.eps_draws), "uniform5": ([0.1, 0.3, 0.5, 0.7, 0.9], [1.0] * 5, 1)}
@torch.no_grad()
def reward(kk, texts, grid, w, K, eps):
    """-sum_t w_t mean_k FM_{t,k}; eps: dict (t,k) -> [N_all, d] shared per pair"""
    x0, src = inputs(kk); N = len(kk); out = torch.zeros(N, device=dev); w = [x / sum(w) for x in w]
    with torch.autocast("cuda", dtype=torch.bfloat16): enc, mask = encoder(texts)
    for ti, t in enumerate(grid):
        for k in range(K):
            e = eps[(t, k)][kk].to(dev); tt = torch.full((N,), t, device=dev); x_t = (1 - tt)[:, None] * x0 + tt[:, None] * e
            with torch.autocast("cuda", dtype=torch.bfloat16): v = model(x_t, tt, src, enc=enc, enc_mask=mask)
            out = out + (w[ti] / K) * ((v.float() - (e - x0)) ** 2).mean(-1)
    return -out
res = {"ckpt": a.ckpt, "step": step, "n_pairs": len(ids), "grids": {}}
for gname, (grid, w, K) in GRIDS.items():
    ge = torch.Generator().manual_seed(a.seed + 7); eps = {(t, k): torch.randn(len(ids), d, generator=ge) for t in grid for k in range(K)}
    # jobs: (pair k, variant, text) for full texts and each leave-one-out text
    jobs = []; meta = []
    for k, pid in enumerate(ids):
        for var in ("true", "twin_new", "twin_shift"):
            raw = T.get((pid, var));
            if raw is None: continue
            jobs.append((k, raw)); meta.append((k, var, "full", None))
            for (s, e, lab) in bullets(raw): jobs.append((k, without(raw, s, e).strip() or " ")); meta.append((k, var, "loo", (s, e, lab, raw[s:e])))
    R = torch.zeros(len(jobs))
    for b0 in range(0, len(jobs), a.batch):
        kk = torch.tensor([j[0] for j in jobs[b0:b0 + a.batch]]); R[b0:b0 + a.batch] = reward(kk, [j[1] for j in jobs[b0:b0 + a.batch]], grid, w, K, eps).cpu()
        if (b0 // a.batch) % 20 == 0: print(f"[smoke/{gname}] {b0}/{len(jobs)} jobs, {time.time() - T0:.0f}s", flush=True)
    full = {(m[0], m[1]): float(R[q]) for q, m in enumerate(meta) if m[2] == "full"}
    gains = {}                                                   # (k, var) -> list of (bullet_text, lab, gain)
    for q, m in enumerate(meta):
        if m[2] == "loo": gains.setdefault((m[0], m[1]), []).append((m[3][3], m[3][2], full[(m[0], m[1])] - float(R[q])))
    true_g = [g_ for (k, var), L in gains.items() if var == "true" for (_, _, g_) in L]
    by_lab = {lab: [g_ for (k, var), L in gains.items() if var == "true" for (_, l_, g_) in L if l_ == lab] for lab in LINES}
    # sign test on the swapped slot: the twin's items differ from the true's in exactly one slot -> compare that slot's gain in true vs twin
    comp = {"twin_new": [], "twin_shift": []}; pos_of = {}
    for k, pid in enumerate(ids):
        tg = gains.get((k, "true"))
        if not tg: continue
        for var in comp:
            vg = gains.get((k, var))
            if not vg or len(vg) != len(tg): continue
            diff = [q for q, ((bt, lt, _), (bv, lv, _)) in enumerate(zip(tg, vg)) if bt != bv]
            if len(diff) != 1: continue
            q = diff[0]; comp[var].append((int(pid.split(":")[1]), tg[q][2], vg[q][2]))
    def boot(vals, clusters, B=2000):
        vals = np.asarray(vals, dtype=np.float64); cl = np.asarray(clusters); u, inv = np.unique(cl, return_inverse=True); C = len(u); rng_ = np.random.default_rng(3)
        if C < 2: return [None, None]
        sums = np.zeros(C); cnt = np.zeros(C); np.add.at(sums, inv, vals); np.add.at(cnt, inv, 1); idx = rng_.integers(0, C, size=(B, C)); W = np.zeros((B, C))
        for b_ in range(B): np.add.at(W[b_], idx[b_], 1)
        p_ = (W @ sums) / (W @ cnt); return [float(np.percentile(p_, 2.5)), float(np.percentile(p_, 97.5))]
    out = {"t_grid": grid, "t_weights": w, "eps_draws": K, "true_bullets": {"n": len(true_g), "gain_mean": float(np.mean(true_g)), "gain_sem": float(np.std(true_g) / math.sqrt(len(true_g))), "frac_positive": float(np.mean([g_ > 0 for g_ in true_g]))},
           "by_line": {lab: {"n": len(v), "gain_mean": float(np.mean(v)) if v else None, "frac_positive": float(np.mean([g_ > 0 for g_ in v])) if v else None} for lab, v in by_lab.items()}, "swapped_slot": {}}
    for var, L in comp.items():
        if not L: continue
        win = [float(g_t > g_v) for (_, g_t, g_v) in L]; dlt = [g_t - g_v for (_, g_t, g_v) in L]; cl = [p for (p, _, _) in L]
        out["swapped_slot"][var] = {"n": len(L), "p_true_bullet_gain_gt_swapped": float(np.mean(win)), "ci95": boot(win, cl), "mean_gain_true": float(np.mean([g_t for (_, g_t, _) in L])), "mean_gain_swapped": float(np.mean([g_v for (_, _, g_v) in L])), "mean_delta": float(np.mean(dlt))}
    res["grids"][gname] = out
    print(f"[smoke/{gname}] true bullets: gain {out['true_bullets']['gain_mean']:+.5f} ± {out['true_bullets']['gain_sem']:.5f} (frac > 0 {out['true_bullets']['frac_positive']:.3f}, n {out['true_bullets']['n']}) | " + " | ".join(f"{var}: P(true slot > swapped slot) {x['p_true_bullet_gain_gt_swapped']:.3f} {x['ci95']} n {x['n']}" for var, x in out["swapped_slot"].items()), flush=True)
res["elapsed_min"] = (time.time() - T0) / 60; json.dump(res, open(a.out, "w"), indent=1); print("SMOKE_DONE", flush=True)
