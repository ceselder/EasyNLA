"""Score the paraphrase control: Δbits = log2 p(h|z_para) - log2 p(h|z_orig) per row under each conditional flow (paired ODE probes), plus the
frozen MSE critic's ΔMSE. A content critic gives ~0 and P(orig preferred) ~50 %. Rows index /vol_q36/data/sft/av_sft_val.parquet."""
import sys, os, json, math, gc, torch, numpy as np, pyarrow.parquet as pq
sys.path.insert(0, ".")
from nla.flow.scoring import FlowBundle
from nla.flow.eval_cond import exact_logp
dev = "cuda"; STEPS = 32; B = 64
P = json.load(open("/vol_glp/cond/paraphrase_512.json")); items = P["items"]; rows = [it["row"] for it in items]
tab = pq.read_table("/vol_q36/data/sft/av_sft_val.parquet", columns=["activation_vector"]); acts = torch.tensor(np.stack([tab.column("activation_vector")[r].as_py() for r in rows]), dtype=torch.float32)
tags = sys.argv[1:] or ["cond_655M_all", "sw_both", "sw_tokbase", "sw_tokar_frozen", "sw_tokar"]
out_path = "/vol_glp/cond/paraphrase_control.json"; out = json.load(open(out_path)) if os.path.exists(out_path) else {}
for tag in tags:
    ap = f"/vol_glp/cond/{tag}/adapter_latest.pt"
    if not os.path.exists(ap): continue
    aa = torch.load(ap, map_location="cpu")["args"]
    fb = FlowBundle(aa["prior"], ap, aa["stats"], dev, base="Qwen/Qwen3.6-27B", enc_layer=aa.get("enc_layer", 42), ar_ckpt=aa.get("ar_ckpt", "/vol/ckpts/qwen36_27b/ar_sft_merged"))
    x0 = fb.norm.normalize(acts.to(dev)); lp = {"orig": [], "para": []}
    for i in range(0, len(items), B):
        xx = x0[i:i + B]
        for k in ("orig", "para"):
            enc, mk, cv = fb.cond([it[k] for it in items[i:i + B]]); gx = torch.Generator(device=dev).manual_seed(11 + i)
            lp[k].append(exact_logp(fb.model, xx, enc, mk, n_steps=STEPS, probes=1, gen=gx, cvec=cv).cpu())
    d = ((torch.cat(lp["para"]) - torch.cat(lp["orig"])) / math.log(2))
    out[tag] = dict(adapter_step=torch.load(ap, map_location="cpu").get("step"), cond_mode=fb.cond_mode, n=len(items), delta_bits_mean=d.mean().item(), delta_bits_median=d.median().item(),
                    delta_bits_sem=(d.std() / math.sqrt(len(d))).item(), p_orig_preferred=(d < 0).float().mean().item(), per_row=d.tolist())
    print(tag, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out[tag].items() if k != "per_row"}, flush=True)
    json.dump(out, open(out_path, "w")); del fb; gc.collect(); torch.cuda.empty_cache()
print("done")
