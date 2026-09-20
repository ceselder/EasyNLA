"""Which noise levels carry the specifics? Per-t flow-matching loss on the controlled number test.

Re-uses the perturbed explanation texts of a finished halluc_classify run (orig / near / far / hedge / removed per row). For every t on a
fine grid and K shared noise draws, computes the conditional FM loss  ||v(x_t, t | z) - (eps - x0)||^2  for each variant, and reports the
paired detection accuracy P(loss(orig) < loss(variant)) per t and per perturbation type, plus for a few t-grids (the RL reward's uniform
0.1..0.9 grid, low-t only, high-t only). Tells us whether the RL reward should weight noise levels differently to reward exact specifics.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np, torch, pyarrow.parquet as pq
from nla.flow.scoring import FlowBundle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="sw_tokar"); p.add_argument("--classify-json", default=None, help="halluc_classify output with variant texts (default /vol_glp/cond/halluc_classify_numbers_<adapter>.json)")
    p.add_argument("--val-parquet", default="/vol_q36/data/sft/av_sft_val.parquet"); p.add_argument("--k", type=int, default=8); p.add_argument("--n", type=int, default=512)
    p.add_argument("--ts", default="0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98"); p.add_argument("--out", required=True); p.add_argument("--critic", default="/vol/ckpts/qwen36_27b/ar_sft_merged")
    a = p.parse_args(); dev = "cuda:0"; ts = [float(x) for x in a.ts.split(",")]
    from huggingface_hub import snapshot_download
    snap = snapshot_download("Qwen/Qwen3.6-27B", token=os.environ.get("HF_TOKEN"), local_dir="/root/base_snap", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.py", "*.model", "*.tiktoken"])
    ap = f"/vol_glp/cond/{a.adapter}/adapter_latest.pt"; aa = torch.load(ap, map_location="cpu")["args"]
    pco = os.path.join(os.path.dirname(ap), "prior_cotrained_latest.pt")
    fb = FlowBundle(aa["prior"], ap, aa["stats"], dev, base=snap, enc_layer=aa.get("enc_layer", 42), ar_ckpt=aa.get("ar_ckpt", a.critic), prior_override=pco if os.path.exists(pco) else None)
    cj = json.load(open(a.classify_json or f"/vol_glp/cond/halluc_classify_numbers_{a.adapter}.json")); items = cj["items"][: a.n]; modes = [m for m in cj["modes"] if m != "orig"]
    t = pq.read_table(a.val_parquet, columns=["activation_vector"]); N = t.num_rows
    acts = torch.tensor(np.asarray(t.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(N, -1)); d = acts.shape[1]
    K = a.k; V = ["orig"] + modes; res = []
    print(f"[per-t] adapter {a.adapter}, {len(items)} items, variants {V}, K={K}, ts {ts}", flush=True)
    for j, it in enumerate(items):
        t0 = time.time(); r = it["row"]; x0 = fb.norm.normalize(acts[r][None].to(dev)).float()          # [1, d]
        texts = [it["variants"][v]["text"] for v in V]; enc, mk, cv = fb.cond(texts)                    # [V, L, D], [V, L]
        g = torch.Generator(device=dev).manual_seed(5000 + r); eps = torch.randn(K, d, device=dev, generator=g); target = eps - x0      # [K, d]
        B = len(V) * K
        encB = enc.repeat_interleave(K, 0) if enc is not None else None; mkB = mk.repeat_interleave(K, 0) if mk is not None else None; cvB = cv.repeat_interleave(K, 0) if cv is not None else None
        rec = {"row": r, "number": it["number"], "loss": {}, "loss_uncond": {}}
        for tt in ts:
            xt = (1 - tt) * x0 + tt * eps                                                              # [K, d]
            xB = xt.repeat(len(V), 1); tB = torch.full((B,), tt, device=dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                v = fb.model(xB, tB, encB, mkB, cvB).float(); vu = fb.model(xt, torch.full((K,), tt, device=dev)).float()
            loss = ((v - target.repeat(len(V), 1)) ** 2).mean(-1).view(len(V), K).mean(1)              # [V]
            rec["loss"][str(tt)] = {vn: loss[i].item() for i, vn in enumerate(V)}; rec["loss_uncond"][str(tt)] = ((vu - target) ** 2).mean(-1).mean().item()
        rec["pmi_bits_exact"] = {vn: it["variants"][vn].get("pmi_bits") for vn in V}
        res.append(rec)
        if (j + 1) % 32 == 0 or j == 0: print(f"[per-t] {j+1}/{len(items)} ({time.time()-t0:.1f}s/item)", flush=True); json.dump({"adapter": a.adapter, "ts": ts, "k": K, "modes": modes, "items": res}, open(a.out, "w"))
    # ---- summary: detection accuracy per t and per mode, and for a few grids
    def acc(mode, tsel):
        wins = [np.mean([rr["loss"][str(tt)]["orig"] for tt in tsel]) < np.mean([rr["loss"][str(tt)][mode] for tt in tsel]) for rr in res]
        return float(np.mean(wins))
    grids = {"rl_grid_0.1-0.9": [x for x in ts if 0.1 <= x <= 0.9], "low_t<=0.3": [x for x in ts if x <= 0.3], "mid_0.4-0.6": [x for x in ts if 0.4 <= x <= 0.6], "high_t>=0.7": [x for x in ts if x >= 0.7], "all": ts}
    summ = {"per_t": {str(tt): {m: acc(m, [tt]) for m in modes} for tt in ts}, "grids": {g: {m: acc(m, sel) for m in modes} for g, sel in grids.items()},
            "exact_logp_reference": {m: float(np.mean([rr["pmi_bits_exact"]["orig"] > rr["pmi_bits_exact"][m] for rr in res if rr["pmi_bits_exact"].get(m) is not None and rr["pmi_bits_exact"].get("orig") is not None])) for m in modes},
            "mean_gap_per_t": {str(tt): {m: float(np.mean([rr["loss"][str(tt)][m] - rr["loss"][str(tt)]["orig"] for rr in res])) for m in modes} for tt in ts},
            "mean_loss_orig_per_t": {str(tt): float(np.mean([rr["loss"][str(tt)]["orig"] for rr in res])) for tt in ts}, "mean_loss_uncond_per_t": {str(tt): float(np.mean([rr["loss_uncond"][str(tt)] for rr in res])) for tt in ts}}
    json.dump({"adapter": a.adapter, "ts": ts, "k": K, "modes": modes, "summary": summ, "items": res}, open(a.out, "w"))
    print("[per-t] SUMMARY", json.dumps({"grids": summ["grids"], "exact": summ["exact_logp_reference"]}), flush=True)


if __name__ == "__main__":
    main()
