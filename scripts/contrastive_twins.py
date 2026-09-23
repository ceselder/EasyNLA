"""Opus's contrastive-PMI reward, zero-training test (flow-noise groups + matched twins) and the fabrication-deletion test.

stage capture (1 GPU): layer-42 activation at the last token of every row's original text (re-captured; checked against the stored vector), of
               every twin and of the placebo (data/flow_noise/twins.json) -> /vol_glp/cond/flow_noise/twin_acts.pt
stage score   (1 GPU, one critic): with ONE shared set of (t, eps) per row (D draws x 5 t, common random numbers across the row's activations,
               explanations and the unconditional pass), the draw-averaged per-t velocity losses
                 Lu[a, t]      unconditional, for every activation a in [h_stored, h_recap, twins..., placebo]
                 Lc[z, a, t]   conditional, for every sampled explanation z of the row (both AV groups) and every activation a
                 deletion items: Lc for (z, z_remove_false, z_remove_true) at h_stored
               -> /vol_glp/cond/flow_noise/contrastive_<critic>.json. PMI proxy i(a; z) = (d/2) * sum_t w_t (Lu[a,t] - Lc[z,a,t]) is formed offline.
"""
import argparse, json, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
OUT = "/vol_glp/cond/flow_noise"; TS = [0.1, 0.3, 0.5, 0.7, 0.9]
CRITICS = {"sw_tokar": "/vol_glp/cond/sw_tokar/adapter_latest.pt", "trunk_dn64": "/vol_glp/cond/trunk_dn64/adapter_latest.pt"}


def stage_capture(a):
    import intervene_playground_app as pg
    pg._load(); tw = json.load(open(f"{OUT}/twins.json")); acts = pg.S["rows"]["act"]; texts = pg.S["rows"]["text"]; out = {}
    for it in tw["items"]:
        r = it["row"]; _, _, h_re, _ = pg.capture_h(texts[r]); h_st = acts[r].float()
        tv = [pg.capture_h(e["text"])[2].cpu() for e in it["twins"]]; pv = pg.capture_h(it["placebo"]["text"])[2].cpu() if it["placebo"] else None
        out[r] = {"h_stored": h_st, "h_recap": h_re.cpu(), "twins": tv, "placebo": pv}
        cos = torch.nn.functional.cosine_similarity(h_st, h_re.cpu(), dim=0).item()
        sh = [((v - h_re.cpu()).norm() / h_re.norm().cpu()).item() for v in tv]; ps = ((pv - h_re.cpu()).norm() / h_re.norm().cpu()).item() if pv is not None else float("nan")
        print(f"[capture] row {r}: cos(stored, recap) {cos:.4f}; twin shifts {[round(s, 3) for s in sh]}; placebo {ps:.3f}", flush=True)
    torch.save(out, f"{OUT}/twin_acts.pt"); print("[capture] wrote twin_acts.pt", flush=True)


def stage_score(a):
    from nla.flow.scoring import FlowBundle
    from huggingface_hub import snapshot_download
    snap = snapshot_download("Qwen/Qwen3.6-27B", token=os.environ.get("HF_TOKEN"), local_dir="/root/base_snap", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.py", "*.model", "*.tiktoken"])
    gen = json.load(open(f"{OUT}/gen.json")); A = torch.load(f"{OUT}/twin_acts.pt"); dele = json.load(open(f"{OUT}/deletions.json"))["items"]; dev = a.device
    ap = CRITICS.get(a.critic, f"/vol_glp/cond/{a.critic}/adapter_latest.pt"); aa = torch.load(ap, map_location="cpu")["args"]
    pco = os.path.join(os.path.dirname(ap), "prior_cotrained_latest.pt")   # from-scratch / co-trained critics: the prior weights live next to the adapter
    fb = FlowBundle(aa["prior"], ap, aa["stats"], dev, base=snap, enc_layer=aa.get("enc_layer", 42), ar_ckpt=aa.get("ar_ckpt", "/vol/ckpts/qwen36_27b/ar_sft_merged"),
                    prior_override=pco if os.path.exists(pco) else None); fb.model.eval()
    T = len(TS); tt = torch.tensor(TS, device=dev); t0 = time.time(); res = {"critic": a.critic, "ts": TS, "D": a.D, "rows": {}, "deletions": []}

    def build(X0, eps):                                    # X0 [nA, d], eps [D, d] -> x_t, t, target flattened over (a, d, t)
        xt = (1 - tt)[None, None, :, None] * X0[:, None, None, :] + tt[None, None, :, None] * eps[None, :, None, :]   # [nA, D, T, d]
        tgt = (eps[None, :, None, :] - X0[:, None, None, :]).expand_as(xt)
        nA, D = X0.shape[0], eps.shape[0]; return xt.reshape(-1, X0.shape[1]), tt.repeat(nA * D), tgt.reshape(-1, X0.shape[1]), (nA, D, T)

    @torch.no_grad()
    def uncond(X0, eps):
        xt, tv, tgt, shp = build(X0, eps)
        with torch.autocast("cuda", dtype=torch.bfloat16): v = fb.model(xt, tv).float()
        return fb.fm_err(v, tgt, a.metric).view(*shp).mean(1)                                  # [nA, T] draw-averaged

    @torch.no_grad()
    def cond(texts, X0, eps):
        xt, tv, tgt, shp = build(X0, eps); R = xt.shape[0]; out = torch.zeros(len(texts), shp[0], shp[2], device=dev)
        enc, mk, cv = fb.cond(texts)
        for j in range(len(texts)):
            ej = enc[j:j + 1] if enc is not None else None; mj = mk[j:j + 1] if mk is not None else None; cj = cv[j:j + 1] if cv is not None else None
            L = torch.zeros(R, device=dev)
            for s in range(0, R, a.chunk):
                n = min(a.chunk, R - s)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    v = fb.model(xt[s:s + n], tv[s:s + n], ej.expand(n, *ej.shape[1:]) if ej is not None else None, mj.expand(n, *mj.shape[1:]) if mj is not None else None,
                                 cj.expand(n, *cj.shape[1:]) if cj is not None else None).float()
                L[s:s + n] = fb.fm_err(v, tgt[s:s + n], a.metric)
            out[j] = L.view(*shp).mean(1)
        return out                                                                              # [nZ, nA, T]

    for g, row in enumerate(gen["rows"]):
        e = A[row]; acts = [e["h_stored"], e["h_recap"]] + list(e["twins"]) + ([e["placebo"]] if e["placebo"] is not None else [])
        X0 = fb.norm.normalize(torch.stack(acts).to(dev).float()).float()
        gen_ = torch.Generator(device=dev).manual_seed(7_000_003 + row); eps = torch.randn(a.D, X0.shape[1], device=dev, generator=gen_)
        Lu = uncond(X0, eps); rec = {"n_twins": len(e["twins"]), "has_placebo": e["placebo"] is not None, "Lu": Lu.cpu().numpy().round(6).tolist(), "Lc": {}}
        for av, v in gen["avs"].items():
            texts = [z if z else "(empty)" for z in v["explanations"][g]]
            rec["Lc"][av] = cond(texts, X0, eps).cpu().numpy().round(6).tolist()
        # deletion items of this row: (z, remove_false, remove_true) at h_stored only, same draws
        for it in [d for d in dele if d["row"] == row]:
            Lc = cond([it["z"], it["remove_false"] or "(empty)", it["remove_true"] or "(empty)"], X0[:1], eps)
            res["deletions"].append({"av": it["av"], "g": it["g"], "i": it["i"], "row": row, "n_removed": it["n_removed"], "n_false": it["n_false"], "Lu": Lu[0].cpu().numpy().round(6).tolist(), "Lc": Lc[:, 0].cpu().numpy().round(6).tolist()})
        res["rows"][str(row)] = rec
        print(f"[score {a.critic}] row {g + 1}/{len(gen['rows'])} ({len(acts)} activations) {time.time() - t0:.0f}s", flush=True)
    od = a.out_dir or OUT; os.makedirs(od, exist_ok=True); res["metric"] = a.metric; sfx = "" if a.metric == "train" else "_model_space"
    json.dump(res, open(f"{od}/contrastive_{a.critic}{sfx}.json", "w")); print(f"[score] wrote {od}/contrastive_{a.critic}{sfx}.json in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--stage", required=True, choices=["capture", "score"]); p.add_argument("--critic", default="sw_tokar")
    p.add_argument("--D", type=int, default=8); p.add_argument("--chunk", type=int, default=240); p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", default=None, help="output dir (default /vol_glp/cond/flow_noise; the PriorGrad comparison writes to .../flow_noise/pg)")
    p.add_argument("--metric", default="train", choices=["train", "model"], help="velocity-error metric: the critic's training metric (= its RL reward) or its model space (nats proxy); differ only for --whiten-loss original critics")
    a = p.parse_args(); (stage_capture if a.stage == "capture" else stage_score)(a)
