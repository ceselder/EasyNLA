"""Reconstructor comparison on IDENTICAL text: FVE (NLA unit-L2 convention, same as score_dumps) of several critics on (a) every dumped
token-matched verbalizer eval (evalTM_*: same 736 held-out activations, each verbalizer's own explanations) and (b) the gold Opus
explanations of the same rows. Critics: the July critic, the Sonnet-trained and the Opus-trained token-matched reconstructors."""
import sys, os, glob, json, math, gc, torch, numpy as np, pyarrow.parquet as pq
sys.path.insert(0, ".")
from transformers import AutoTokenizer
from nla.models import NLACriticModel
from nla.utils.critic import critic_predict
from nla.schema import normalize_activation, extract_explanation
import nla.flow.score_dumps as SD
compute_predict_mean_baselines, load_nla_config, resolve_target_scale = SD.compute_predict_mean_baselines, SD.load_nla_config, SD.resolve_target_scale
dev = "cuda"; C = "/vol/ckpts/qwen36_27b"; B = 32
critics = sys.argv[1:] or ["ar_sft_merged", "sft_ar_sonnet_merged", "sft_ar_opustm_merged"]
dumps = sorted(glob.glob(f"{C}/evalTM_*/eval_rollouts/step_*_r*.pt"))
gold = pq.read_table("/vol_q36/data/sft/av_sft_val_clean1.parquet", columns=["activation_vector", "response"])
gold_acts = torch.tensor(np.stack(gold.column("activation_vector").to_pylist()), dtype=torch.float32)
gold_z = [extract_explanation(r) or r for r in gold.column("response").to_pylist()]
out_path = "/vol_glp/cond/tm_cross_critic.json"; out = json.load(open(out_path)) if os.path.exists(out_path) else {}
def fve(critic, tok, template, msf, pad, acts, expls):
    valid = [i for i, z in enumerate(expls) if z]; _, baseline = compute_predict_mean_baselines(acts, msf); mses = []
    for cs in range(0, len(valid), B):
        ch = valid[cs: cs + B]
        ids_l = [tok.encode(template.format(explanation=expls[i]), add_special_tokens=False)[:1024] for i in ch]
        T = max(len(x) for x in ids_l); bx = torch.full((len(ch), T), pad, dtype=torch.long, device=dev); am = torch.zeros((len(ch), T), dtype=torch.long, device=dev)
        for r, x in enumerate(ids_l): bx[r, :len(x)] = torch.tensor(x, device=dev); am[r, :len(x)] = 1
        with torch.no_grad(): pred = critic_predict(critic, bx, am, msf)
        mse = ((normalize_activation(pred.float(), msf) - normalize_activation(acts[ch].to(dev), msf)) ** 2).mean(1)
        mses += [v for v in mse.tolist() if math.isfinite(v)]
    return dict(fve=100 * (1 - sum(mses) / len(mses) / baseline), n=len(mses), baseline_mse=baseline)
for cname in critics:
    cdir = f"{C}/{cname}"
    tok = AutoTokenizer.from_pretrained(cdir); pad = tok.eos_token_id
    cfg = load_nla_config("/vol_q36/data/rl/rl_shuf.parquet", tok); template = cfg.critic_prompt_template; msf = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    critic = NLACriticModel.from_pretrained(cdir, torch_dtype=torch.bfloat16).to(dev).eval(); critic.requires_grad_(False)
    res = out.get(cname, {})
    res["gold_clean1"] = fve(critic, tok, template, msf, pad, gold_acts, gold_z); print(cname, "gold", res["gold_clean1"], flush=True)
    for f in dumps:
        name = f.split("/")[-3]; dd = torch.load(f, map_location="cpu")
        res[name] = fve(critic, tok, template, msf, pad, dd["activations"].float(), list(dd["explanations"])); print(cname, name, res[name], flush=True)
    out[cname] = res; json.dump(out, open(out_path, "w"), indent=1)
    del critic; gc.collect(); torch.cuda.empty_cache()
print("done", {c: {k: round(v["fve"], 1) for k, v in r.items()} for c, r in out.items()})
