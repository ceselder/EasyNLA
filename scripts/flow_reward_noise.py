"""Is the flow reward mostly noise? Real GRPO-style groups, scored many times.

stage gen   (2 GPUs: LM + verbalizer adapters + MSE critic on cuda:0): for N held-out activations (clean1 val rows) sample G explanations each
            (temperature 1, 200 tokens — the RL sampling setting) from each of the chosen verbalizer checkpoints; score each with the MSE reward
            (-MSE of the SFT reconstructor's prediction, NLA unit-L2 units = the MSE arms' reward). -> /vol_glp/cond/flow_noise/gen.json
stage score (1 GPU): for one flow critic, score every (activation, explanation) pair under D independent noise draws eps_d at every t of a 9-point
            grid, with eps SHARED across the G samples of a group and across t within a draw (exactly the trainer's scheme: FlowCritic.score draws
            one eps per group per step). Stores the per-(group, sample, draw, t) flow-matching loss, from which every estimator (1 vs K draws,
            shared vs independent noise, t subsets / weightings) is evaluated offline. -> /vol_glp/cond/flow_noise/score_<critic>.json
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
OUT = "/vol_glp/cond/flow_noise"
CRITICS = {"sw_tokar": "/vol_glp/cond/sw_tokar/adapter_latest.pt", "trunk_dn64": "/vol_glp/cond/trunk_dn64/adapter_latest.pt",
           "trunk_rl400": "/vol/ckpts/qwen36_27b/rlQ36_trunk128/flow_latest/adapter_latest.pt"}
AVS = {"warm": "warm start (SFT verbalizer, iter_0007813)", "trunk400": "fast 128×8, whole-trunk flow critic, step 400"}
TS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def stage_gen(a):
    import intervene_playground_app as pg
    import pyarrow.parquet as pq
    from nla.schema import normalize_activation
    pg._load(); rows = pg.S["rows"]; t = pq.read_table(pg.ROWS_PARQUET, columns=["n_raw_tokens", "doc_id"])
    ntok = t.column("n_raw_tokens").to_pylist(); rng = np.random.default_rng(a.seed); seen, pick = set(), []
    for i in rng.permutation(len(ntok)):
        if 60 <= ntok[i] <= 400 and rows["doc"][i] not in seen: pick.append(int(i)); seen.add(rows["doc"][i])
        if len(pick) == a.n: break
    critic = pg._critic("SFT reconstructor (ar_sft_merged, pre-RL)"); msf = pg.S["msf"]
    out = {"rows": pick, "G": a.G, "avs": {}}
    for key in a.avs.split(","):
        t0 = time.time(); vecs = rows["act"][pick].repeat_interleave(a.G, 0)
        ex = pg._verbalize(vecs, AVS[key], temperature=1.0, max_new=200, bs=a.bs)
        mse = []
        for j, z in enumerate(ex):
            h = rows["act"][pick[j // a.G]].to(pg.DEV0)
            p = pg.ar_pred(critic, z) if z else None
            mse.append(None if p is None else -float(((normalize_activation(p[None], msf) - normalize_activation(h[None].float(), msf)) ** 2).mean()))
        out["avs"][key] = {"explanations": [ex[g * a.G:(g + 1) * a.G] for g in range(len(pick))], "mse_reward": [mse[g * a.G:(g + 1) * a.G] for g in range(len(pick))]}
        print(f"[gen] {key}: {len(ex)} explanations in {time.time() - t0:.0f}s; empty {sum(1 for z in ex if not z)}", flush=True)
    os.makedirs(OUT, exist_ok=True); json.dump(out, open(f"{OUT}/gen.json", "w")); print("[gen] wrote", f"{OUT}/gen.json", flush=True)


def stage_score(a):
    import pyarrow.parquet as pq
    from nla.flow.scoring import FlowBundle
    from huggingface_hub import snapshot_download
    snap = snapshot_download("Qwen/Qwen3.6-27B", token=os.environ.get("HF_TOKEN"), local_dir="/root/base_snap", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.py", "*.model", "*.tiktoken"])
    gen = json.load(open(f"{OUT}/gen.json")); dev = a.device
    t = pq.read_table("/vol_q36/data/sft/av_sft_val_clean1.parquet", columns=["activation_vector"]); N = t.num_rows
    acts = torch.tensor(np.asarray(t.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(N, -1))
    ap = CRITICS.get(a.critic, f"/vol_glp/cond/{a.critic}/adapter_latest.pt"); aa = torch.load(ap, map_location="cpu")["args"]; pco = os.path.join(os.path.dirname(ap), "prior_cotrained_latest.pt")
    fb = FlowBundle(aa["prior"], ap, aa["stats"], dev, base=snap, enc_layer=aa.get("enc_layer", 42), ar_ckpt=aa.get("ar_ckpt", "/vol/ckpts/qwen36_27b/ar_sft_merged"), prior_override=pco if os.path.exists(pco) else None)
    fb.model.eval(); d = acts.shape[1]; T = len(TS); tt_all = torch.tensor(TS, device=dev); two = fb.err_map is not None   # plain-loss critics: also keep the model-space loss
    res = {"critic": a.critic, "adapter": ap, "ts": TS, "D": a.D, "avs": {}}; t0 = time.time()
    for key, v in gen["avs"].items():
        L_all, Lm_all = [], []
        for g, row in enumerate(gen["rows"]):
            texts = [z if z else "(empty)" for z in v["explanations"][g]]; G = len(texts)
            x0 = fb.norm.normalize(acts[row][None].to(dev)).float()                                   # [1, d]
            gen_ = torch.Generator(device=dev).manual_seed(a.seed * 1_000_003 + row)
            eps = torch.randn(a.D, d, device=dev, generator=gen_)                                     # draw d shared by the whole group and all t
            with torch.no_grad():
                enc, mk, cv = fb.cond(texts)
                L = torch.zeros(G, a.D, T, device=dev); Lm = torch.zeros(G, a.D, T, device=dev) if two else None
                for d0 in range(0, a.D, a.chunk):
                    E = eps[d0:d0 + a.chunk]; Dc = E.shape[0]
                    xt = (1 - tt_all)[None, :, None] * x0[None] + tt_all[None, :, None] * E[:, None, :]   # [Dc, T, d]
                    xt = xt.reshape(Dc * T, d); tv = tt_all.repeat(Dc); tgt = (E[:, None, :] - x0[None]).expand(Dc, T, d).reshape(Dc * T, d)
                    R = Dc * T                                                                          # rows per explanation
                    xB = xt.repeat(G, 1); tB = tv.repeat(G); tgtB = tgt.repeat(G, 1)
                    encB = enc.repeat_interleave(R, 0) if enc is not None else None; mkB = mk.repeat_interleave(R, 0) if mk is not None else None
                    cvB = cv.repeat_interleave(R, 0) if cv is not None else None
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        vv = fb.model(xB, tB, encB, mkB, cvB).float()
                    L[:, d0:d0 + Dc, :] = fb.fm_err(vv, tgtB).view(G, Dc, T)                              # the critic's training metric (= its reward)
                    if two: Lm[:, d0:d0 + Dc, :] = fb.fm_err(vv, tgtB, metric="model").view(G, Dc, T)
            L_all.append(L.cpu().numpy().round(5).tolist())
            if two: Lm_all.append(Lm.cpu().numpy().round(5).tolist())
            if g % 10 == 0: print(f"[score {a.critic}] {key} group {g + 1}/{len(gen['rows'])} {time.time() - t0:.0f}s", flush=True)
        res["avs"][key] = {"L": L_all, **({"L_model_space": Lm_all} if two else {})}
    od = a.out_dir or OUT; os.makedirs(od, exist_ok=True)
    fn = f"{od}/score_{a.critic.replace('/', '__')}.json"   # snapshot tags <run>/snap_<pairs> -> flat file names
    json.dump(res, open(fn, "w")); print(f"[score] wrote {fn} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--stage", required=True, choices=["gen", "score"]); p.add_argument("--n", type=int, default=40); p.add_argument("--G", type=int, default=8)
    p.add_argument("--avs", default="warm,trunk400"); p.add_argument("--bs", type=int, default=40); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--critic", default="sw_tokar"); p.add_argument("--D", type=int, default=20); p.add_argument("--chunk", type=int, default=4); p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", default=None, help="score output dir (default /vol_glp/cond/flow_noise; the PriorGrad comparison writes to .../flow_noise/pg so plot_flow_noise.py's glob is unaffected)")
    a = p.parse_args(); (stage_gen if a.stage == "gen" else stage_score)(a)
