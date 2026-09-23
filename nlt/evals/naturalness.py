"""Reference-LM naturalness of z (EVALS 7a): per-token NLL of z under the FIXED base Qwen3-8B given the text-only prompt that DECISIONS v1.2
uses as the KL reference (no markers, no injection). Reports median / mean NLL per token, the share of z above 2x the corpus median, and the
ratio to a reference z table (e.g. the warm start) when given.

  python -m nlt.evals.naturalness --z z.parquet --out natural.json [--ref-z warmstart.parquet --model Qwen/Qwen3-8B --n 1024]
"""
from __future__ import annotations
import argparse, json
import numpy as np, torch, torch.nn.functional as F

PROMPT = "Describe what changed inside a language model while it processed a passage.\n"


@torch.no_grad()
def nll_per_token(model, tok, texts, dev, batch=16):
    out = []
    p_ids = tok(PROMPT, add_special_tokens=False)["input_ids"]
    for s in range(0, len(texts), batch):
        chunk = texts[s:s + batch]; seqs = [p_ids + tok(z, add_special_tokens=False)["input_ids"][:256] for z in chunk]
        T = max(len(x) for x in seqs); pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        ids = torch.full((len(seqs), T), pad, dtype=torch.long); am = torch.zeros((len(seqs), T), dtype=torch.long)
        for r, x in enumerate(seqs): ids[r, :len(x)] = torch.tensor(x); am[r, :len(x)] = 1
        ids, am = ids.to(dev), am.to(dev)
        logits = model(input_ids=ids, attention_mask=am).logits.float()
        lp = F.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:, None])[..., 0]
        for r, x in enumerate(seqs):
            n_z = len(x) - len(p_ids)
            out.append(float(-lp[r, len(p_ids) - 1: len(x) - 1].mean()) if n_z > 0 else float("nan"))
    return np.asarray(out)


def main():
    from nlt.evals.common import load_table
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser(); ap.add_argument("--z", required=True); ap.add_argument("--out", required=True); ap.add_argument("--ref-z"); ap.add_argument("--model", default="Qwen/Qwen3-8B"); ap.add_argument("--n", type=int, default=1024); ap.add_argument("--device", default="cuda"); ap.add_argument("--dtype", default="bfloat16", help="bfloat16 on GPU; float32 on CPU")
    a = ap.parse_args(); dev = a.device
    tok = AutoTokenizer.from_pretrained(a.model); model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), attn_implementation="sdpa").to(dev).eval()
    z = load_table(a.z); texts = z["text"].fillna("").astype(str); texts = texts[texts.str.strip().str.len() > 0].tolist()[: a.n]
    nll = nll_per_token(model, tok, texts, dev); res = {"n": int(np.isfinite(nll).sum()), "nll_per_token_median": float(np.nanmedian(nll)), "nll_per_token_mean": float(np.nanmean(nll)),
                                                      "share_above_2x_median": float(np.nanmean(nll > 2 * np.nanmedian(nll)))}
    if a.ref_z:
        rz = load_table(a.ref_z); rt = rz["text"].fillna("").astype(str); rt = rt[rt.str.strip().str.len() > 0].tolist()[: a.n]
        rn = nll_per_token(model, tok, rt, dev); res["ref_nll_per_token_median"] = float(np.nanmedian(rn)); r = res["nll_per_token_median"] / res["ref_nll_per_token_median"]
        res["ratio_to_ref"] = float(r); res["verdict_7a"] = "PASS" if r <= 1.3 else ("WARN" if r <= 1.8 else "FAIL")
    json.dump(res, open(a.out, "w"), indent=1); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
