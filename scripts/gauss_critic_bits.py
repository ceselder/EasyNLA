"""Reference for the flow's bits: how many bits does the MSE critic's mean prediction E[h|z] carry under a GAUSSIAN residual model?
Runs the frozen critic (ar_sft_merged) on gold explanations of the eval rows, then
  bits_iso  = -(d/2) log2(1 - FVE)                      (isotropic Gaussian, what FVE 'would be worth' if variance were spread evenly)
  bits_diag = 1/2 sum_i log2(var_i(h) / var_i(h - E[h|z]))  in the raw basis and in the PCA basis of h (top-k components)
These are scorer-independent yardsticks for the flow's exact PMI = log p(h|z) - log p(h)."""
import sys, json, math, torch, numpy as np, pyarrow.parquet as pq
sys.path.insert(0, ".")
from nla.models import NLACriticModel
from nla.schema import extract_explanation, normalize_activation
from transformers import AutoTokenizer
dev = "cuda"; ar_dir = "/vol/ckpts/qwen36_27b/ar_sft_merged"
crit = NLACriticModel.from_pretrained(ar_dir, dtype=torch.bfloat16).to(dev).eval()
tok = AutoTokenizer.from_pretrained(ar_dir); tok.padding_side = "right"
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
msf = math.sqrt(crit.value_head.weight.shape[0]); tmpl = "Summary of the following text: <text>{explanation}</text> <summary>"
@torch.no_grad()
def predict(zs, bs=16):
    out = []
    for i in range(0, len(zs), bs):
        enc = tok([tmpl.format(explanation=z) for z in zs[i:i + bs]], return_tensors="pt", padding=True, truncation=True, max_length=384, add_special_tokens=False)
        ids, am = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
        last = crit(input_ids=ids, attention_mask=am).backbone_last_hidden[torch.arange(ids.shape[0], device=dev), am.sum(1) - 1].float()
        out.append(crit.value_head(normalize_activation(last, msf).to(crit.value_head.weight.dtype)).float().cpu())
    return torch.cat(out)
def run(parquet, n):
    t = pq.read_table(parquet, columns=["activation_vector", "response"]).slice(0, n)
    h = torch.tensor(np.stack(t.column("activation_vector").to_pylist()), dtype=torch.float32)
    zs = [extract_explanation(r) or r for r in t.column("response").to_pylist()]
    pred = predict(zs); d = h.shape[1]; r = h - pred
    var = ((h - h.mean(0)) ** 2).mean().item(); mse = (r ** 2).mean().item(); fve = 1 - mse / var
    s2 = h.var(0, unbiased=False); r2 = r.var(0, unbiased=False); eps = 1e-6 * s2.mean()
    bits_raw = (0.5 * torch.log2((s2 + eps) / (r2 + eps))).sum().item()
    hc = h - h.mean(0); U, S, Vt = torch.linalg.svd(hc, full_matrices=False)
    res = dict(n=int(h.shape[0]), d=d, fve=fve, bits_iso_from_fve=-0.5 * d * math.log2(max(1 - fve, 1e-9)), bits_diag_raw=bits_raw)
    tot = (S ** 2).sum().item()
    for k in (64, 256, 512, min(700, len(S) - 1)):
        P = Vt[:k]; hp = hc @ P.T; rp = (r - r.mean(0)) @ P.T
        res[f"bits_diag_pca{k}"] = (0.5 * torch.log2((hp.var(0) + eps) / (rp.var(0) + eps))).sum().item()
        res[f"var_share_top{k}"] = ((S[:k] ** 2).sum() / tot).item()
    res["bits_per_dim_iso"] = res["bits_iso_from_fve"] / d
    return res
out = {"clean1_gold_736": run("/vol_q36/data/sft/av_sft_val_clean1.parquet", 736), "mixed_gold_first1024": run("/vol_q36/data/sft/av_sft_val.parquet", 1024)}
json.dump(out, open("/vol_glp/cond/gauss_critic_bits.json", "w"), indent=1); print(json.dumps(out, indent=1))
