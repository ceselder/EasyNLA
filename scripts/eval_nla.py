"""Standalone held-out evaluation of an NLA (AV + AR) on a fixed test parquet.

  1. vLLM generates one explanation per test row through the vllm-lens injection path
     (rollout_batch_vllm, group_size=1) at --temperature (1.0 = the training/rollout
     distribution; 0.0 = greedy).
  2. The AR critic scores every explanation -> FVE (paper definition, variance-around-
     mean baseline computed on the SAME rows) + extraction rate + length.
  3. Sonnet-5 judges (nla.utils.halluc_eval + text_judges) on the first --judge-n rows.
  4. Optional: score the same explanations with a SECOND critic (--ar-ckpt-b) so two
     ARs are compared on identical text (e.g. AR_sft vs AR_filtered on AV_sft rollouts).
  5. Optional: --gold-fve also reports each AR's FVE on the parquet's GOLD explanations
     (the `response` column) — the "how well does this AR read Opus text" number.

Writes <out>.json (metrics) and <out>.samples.parquet (per-row explanations + scores).
"""
import argparse
import json
import math
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from nla.config import load_nla_config
from nla.models import NLACriticModel
from nla.schema import (compute_predict_mean_baselines, extract_explanation,
                        normalize_activation, resolve_target_scale)
from nla.utils import build_prompt_text, critic_predict


def load_rows(parquet, n, offset=0):
    pf = pq.ParquetFile(parquet)
    cols = ["prompt", "activation_vector", "doc_id", "detokenized_text_truncated"]
    if "response" in pf.schema_arrow.names:
        cols.append("response")
    rows, seen = [], 0
    for rg in range(pf.num_row_groups):
        if len(rows) >= n:
            break
        t = pf.read_row_group(rg, columns=cols)
        m = t.num_rows
        if seen + m <= offset:
            seen += m
            continue
        pr = t.column("prompt").to_pylist()
        ac = np.asarray(t.column("activation_vector").combine_chunks().flatten(),
                        dtype=np.float32).reshape(m, -1)
        dd = t.column("doc_id").to_pylist()
        sr = t.column("detokenized_text_truncated").to_pylist()
        rs = t.column("response").to_pylist() if "response" in cols else [None] * m
        for j in range(max(0, offset - seen), m):
            rows.append({"prompt": pr[j], "activation": ac[j], "doc_id": dd[j],
                         "source": sr[j] or "", "gold": rs[j]})
            if len(rows) >= n:
                break
        seen += m
    return rows


@torch.no_grad()
def score(critic, tok, template, expls, acts, mse_scale_f, device, bs=32):
    """per-row normalized MSE (nan for failed/over-length) under one critic."""
    out = [float("nan")] * len(expls)
    ids_list = [None] * len(expls)
    for i, e in enumerate(expls):
        if e is None:
            continue
        ids = tok.encode(template.format(explanation=e), add_special_tokens=False)
        if 0 < len(ids) <= 1024:
            ids_list[i] = ids
    valid = [i for i in range(len(expls)) if ids_list[i] is not None]
    pad = tok.eos_token_id
    for cs in range(0, len(valid), bs):
        chunk = valid[cs:cs + bs]
        L = max(len(ids_list[i]) for i in chunk)
        bx = torch.full((len(chunk), L), pad, dtype=torch.long, device=device)
        am = torch.zeros((len(chunk), L), dtype=torch.long, device=device)
        for r, i in enumerate(chunk):
            bx[r, :len(ids_list[i])] = torch.tensor(ids_list[i], device=device)
            am[r, :len(ids_list[i])] = 1
        pred = critic_predict(critic, bx, am, mse_scale_f)
        gold = torch.tensor(np.stack([acts[i] for i in chunk]), dtype=torch.float32, device=device)
        mse = ((normalize_activation(pred, mse_scale_f) - normalize_activation(gold, mse_scale_f)) ** 2).mean(1)
        for r, i in enumerate(chunk):
            out[i] = float(mse[r].item())
    return out


def fve_from(mses, baseline):
    v = [m for m in mses if not math.isnan(m)]
    return (1.0 - float(np.mean(v)) / baseline) * 100.0 if v else float("nan"), len(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--av-ckpt", required=True, help="merged bf16 HF dir for vLLM")
    p.add_argument("--ar-ckpt", required=True, help="HF critic dir (train_sft AR / critic_latest)")
    p.add_argument("--ar-ckpt-b", default=None, help="second critic scored on the same text")
    p.add_argument("--ar-ckpts", nargs="*", default=[],
                   help="more critics as name=path; each is scored on the same generations "
                        "(metrics fve_<name>, gold_fve_<name>)")
    p.add_argument("--parquet", required=True, help="test AV-format parquet")
    p.add_argument("--sidecar", default=None)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--judge-n", type=int, default=256)
    p.add_argument("--judge-model", default="claude-sonnet-5")
    p.add_argument("--no-text-judges", action="store_true")
    p.add_argument("--gold-fve", action="store_true")
    p.add_argument("--vllm-gpu-mem", type=float, default=0.6)
    p.add_argument("--out", required=True)
    p.add_argument("--tag", default="")
    args = p.parse_args()

    from transformers import AutoTokenizer
    from nla.train_rl_vllm import rollout_batch_vllm
    tok = AutoTokenizer.from_pretrained(args.av_ckpt)
    cfg = load_nla_config(args.sidecar or args.parquet, tok)
    mse_scale_f = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    template = cfg.critic_prompt_template
    device = "cuda"
    rows = load_rows(args.parquet, args.n, args.offset)
    acts = [r["activation"] for r in rows]
    _, baseline = compute_predict_mean_baselines(
        torch.tensor(np.stack(acts), dtype=torch.float32), mse_scale_f)
    print(f"[eval] {len(rows)} rows, baseline mse={baseline:.4f}", flush=True)

    # ---- 1. generate
    from vllm import LLM
    from nla.utils.vllm_steer import vllm_attn_kwargs
    llm = LLM(**vllm_attn_kwargs(), model=args.av_ckpt, tokenizer=args.av_ckpt, dtype="bfloat16",
              gpu_memory_utilization=args.vllm_gpu_mem, max_model_len=1024,
              enforce_eager=True, disable_log_stats=True, enable_prefix_caching=False)
    pwa = [(build_prompt_text(r["prompt"], cfg.injection_char, tok),
            torch.tensor(r["activation"], dtype=torch.float32)) for r in rows]
    t0 = time.time()
    resp = rollout_batch_vllm(llm, tok, pwa, cfg.injection_token_id, 1, args.max_new_tokens,
                              args.temperature, left_id=cfg.injection_left_neighbor_id,
                              right_id=cfg.injection_right_neighbor_id)
    by = {r["prompt_idx"]: r for r in resp}
    expls, lens, trunc, unver = [], [], 0, 0
    for i in range(len(rows)):
        r = by[i]
        e = extract_explanation(r["text"])
        if r.get("truncated"):
            e = None; trunc += 1
        unver += not r.get("steer_verified", True)
        expls.append(e); lens.append(int(r["n_resp"]))
    gen_s = time.time() - t0
    del llm
    torch.cuda.empty_cache()
    print(f"[eval] generated in {gen_s:.0f}s; extraction {np.mean([e is not None for e in expls]):.1%} "
          f"trunc={trunc} unverified={unver}", flush=True)

    # ---- 2. critic(s)
    metrics = {"n": len(rows), "temperature": args.temperature, "baseline_mse": baseline,
               "extraction_rate": float(np.mean([e is not None for e in expls])),
               "truncated": trunc, "steer_unverified": unver,
               "resp_len_mean": float(np.mean(lens)), "gen_s": gen_s, "tag": args.tag,
               "av_ckpt": args.av_ckpt, "ar_ckpt": args.ar_ckpt, "parquet": args.parquet}
    per = {"doc_id": [r["doc_id"] for r in rows], "explanation": expls, "n_tokens": lens}
    crit_specs = [("a", args.ar_ckpt)] + ([("b", args.ar_ckpt_b)] if args.ar_ckpt_b else [])
    for spec in args.ar_ckpts:
        name, path = spec.split("=", 1)
        crit_specs.append((name, path))
    golds = [extract_explanation(r["gold"]) if r.get("gold") else None for r in rows]
    for key, path in crit_specs:
        critic = NLACriticModel.from_pretrained(path, torch_dtype=torch.bfloat16).to(device).eval()
        mses = score(critic, tok, template, expls, acts, mse_scale_f, device)
        fve, nv = fve_from(mses, baseline)
        metrics[f"fve_{key}"] = fve; metrics[f"fve_{key}_n"] = nv
        per[f"mse_{key}"] = mses
        print(f"[eval] critic {key} ({path}): FVE {fve:.2f}% on {nv} rows", flush=True)
        if args.gold_fve and any(g for g in golds):
            gm = score(critic, tok, template, golds, acts, mse_scale_f, device)
            gf, gn = fve_from(gm, baseline)
            metrics[f"gold_fve_{key}"] = gf; metrics[f"gold_fve_{key}_n"] = gn
            per[f"gold_mse_{key}"] = gm
            print(f"[eval] critic {key} on GOLD explanations: FVE {gf:.2f}% ({gn})", flush=True)
        del critic
        torch.cuda.empty_cache()

    # ---- 3. judges (Sonnet 5)
    if args.judge_n > 0:
        from nla.utils.halluc_eval import judge_hallucination
        jn = min(args.judge_n, len(rows))
        srcs = [r["source"] for r in rows[:jn]]
        t1 = time.time()
        hm, hs = judge_hallucination(expls[:jn], srcs, model=args.judge_model, concurrency=32,
                                     total_timeout_s=1800)
        metrics.update({f"judge/{k}": v for k, v in hm.items()})
        per["halluc"] = [d.get("halluc") for d in hs] + [None] * (len(rows) - jn)
        per["inform"] = [d.get("inform") for d in hs] + [None] * (len(rows) - jn)
        hv = [d.get("halluc") for d in hs if isinstance(d.get("halluc"), int)]
        metrics["judge/halluc_le3_frac"] = float(np.mean([h <= 3 for h in hv])) if hv else float("nan")
        metrics["judge/halluc_ge7_frac"] = float(np.mean([h >= 7 for h in hv])) if hv else float("nan")
        print(f"[eval] halluc {hm.get('hallucination_mean'):.2f} inform {hm.get('informativeness_mean'):.2f} "
              f"(n={hm.get('n_judged')}, fail {hm.get('judge_fail_rate'):.0%}, {time.time()-t1:.0f}s)", flush=True)
        if not args.no_text_judges:
            from nla.utils.text_judges import judge_explanations
            t2 = time.time()
            tm, ts = judge_explanations(expls[:jn], srcs, model=args.judge_model, concurrency=32,
                                        total_timeout_s=1800)
            metrics.update({f"text/{k}": v for k, v in tm.items()})
            for dim in ("writing_quality", "coherence", "specificity", "unique_info",
                        "repetitiveness", "interestingness", "source_match"):
                per[f"tj_{dim}"] = [d.get(dim) for d in ts] + [None] * (len(rows) - jn)
            print(f"[eval] text judges: " + " ".join(f"{k}={v:.2f}" for k, v in tm.items()
                                                      if isinstance(v, float) and not math.isnan(v))
                  + f" ({time.time()-t2:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(metrics, open(args.out if args.out.endswith(".json") else args.out + ".json", "w"), indent=2)
    cols = {}
    for k, v in per.items():
        cols[k] = pa.array(v)
    pq.write_table(pa.table(cols), (args.out[:-5] if args.out.endswith(".json") else args.out) + ".samples.parquet")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
