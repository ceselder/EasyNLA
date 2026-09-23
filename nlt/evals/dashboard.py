"""Aggregate every eval output for a set of z sources into ONE verdict table (EVALS.md rows) as JSON + Markdown.
Inputs are the files the other modules write (any subset may be missing):
  text_<tag>.json                (run_text_evals)         -> 4c, 5a, 5b, 6a, 6b, 7d, 7e, next-token mention rate
  manifest_<tag>_scored.parquet  (infra score_manifest)    -> 1a/1b/1e, 3a-3d, 4e, 5c, src, 7d-bits (via summarize_scored)
  para_<tag>_scored.parquet      (paraphrase_eval)         -> 2a-2c, 9f, mask drop
  natural_<tag>.json             (naturalness)             -> 7a
  judged_<tag>_<task>.jsonl      (judge_batch)             -> 7c, 9e, 5d, 9a, 9b, 9c, 8b ; magnitude -> 8a via reader_tables.correlate (corr_<tag>.json)
  bits_<critics>.json            (infra eval_bits.run)     -> 4a/4b told-depth gain (exact / proxy)

  python -m nlt.evals.dashboard --dir /vol/evals --tags lensdiff_L1,teacher_sonnet_v1,rl_step100 --out /vol/evals/dashboard.json [--md dashboard.md]
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np

ROWS = ["1a", "1b", "1e", "2a", "2b", "2c", "3a", "3b", "3c", "3d", "4a", "4c", "4e", "5a", "5b", "5c", "5d", "6a", "6b", "7a", "7c", "7d", "7d_bits", "7e", "8a", "8b", "9a", "9b", "9c", "9e", "9f", "src"]


def _load(path):
    try: return json.load(open(path))
    except Exception: return None


def collect(d, tag):
    out = {"tag": tag, "verdicts": {}, "numbers": {}}
    t = _load(f"{d}/text_{tag}.json")
    if t:
        out["verdicts"].update({k.replace("verdict_", ""): v for k, v in t.get("verdicts", {}).items()})
        out["numbers"].update(hard_per_1000=t["regex"]["hard_hits_per_1000_z"], soft_per_z=t["regex"]["soft_hits_per_z"], tokens_median=t["diversity"]["tokens_median"], distinct4=t["diversity"]["distinct_4gram_ratio"],
                              self_bleu=t["diversity"]["self_bleu4"], n_texts=t["n"])
        if "depth" in t: out["numbers"].update(mi_z_j_bits=t["depth"]["j"].get("mi_bits"), gap_mae_ratio=t["depth"]["gap"].get("ratio"))
        if "copy" in t: out["numbers"].update(copy4=t["copy"]["copy_rate_4gram_mean"], lcs_p95=t["copy"]["lcs_tokens_p95"], next_token_mention=t["copy"].get("next_token_mention_rate"))
    sc = glob.glob(f"{d}/manifest_{tag}*_scored.parquet") + glob.glob(f"{d}/scored_{tag}*.parquet")
    if sc:
        try:
            import pandas as pd
            from nlt.evals.summarize_scored import summarize
            s = summarize(pd.read_parquet(sc[0]))
            out["verdicts"].update({k.replace("verdict_", ""): v for k, v in s.items() if k.startswith("verdict_")})
            out["numbers"].update(bits_orig=s["orig"]["bits_mean"], bits_orig_ci=s["orig"]["ci95"], bits_per_token=s["orig"].get("bits_per_token_mean"),
                                  bits_dm=s.get("dm", {}).get("bits_mean"), bits_rp=s.get("rp", {}).get("bits_mean"), bits_copy=s.get("copy", {}).get("bits_mean"), bits_src=s.get("src_desc", {}).get("bits_mean"),
                                  p_orig_gt_wrong_j=s.get("wrong_j", {}).get("p_orig_higher"), proxy_over_exact=s.get("proxy_over_exact_orig"), workspace_share=s.get("workspace_share_of_positive_bits"),
                                  bits_by_band={b: v["bits_mean"] for b, v in s.get("by_band", {}).get("orig", {}).items()})
        except Exception as e: out["numbers"]["scored_error"] = str(e)[:120]
    pe = glob.glob(f"{d}/para_{tag}*_scored.parquet")
    if pe:
        try:
            import pandas as pd
            from nlt.evals.paraphrase_eval import summarize as psum
            s = psum(pd.read_parquet(pe[0]))
            for v in ("para_light", "para_strong", "twin", "mask_next"):
                if v in s:
                    out["verdicts"].update({k.replace("verdict_", ""): vv for k, vv in s[v].items() if k.startswith("verdict_")})
                    out["numbers"][f"{v}_retention"] = s[v].get("retention_median"); out["numbers"][f"{v}_p_orig"] = s[v].get("p_orig_preferred")
        except Exception as e: out["numbers"]["para_error"] = str(e)[:120]
    n = _load(f"{d}/natural_{tag}.json")
    if n:
        out["numbers"].update(nll_per_token=n["nll_per_token_median"], nll_ratio_to_ref=n.get("ratio_to_ref"))
        if "verdict_7a" in n: out["verdicts"]["7a"] = n["verdict_7a"]
    for task, row in (("fluency", "7c"), ("claim", "9e"), ("restate", "5d"), ("top1", "9a"), ("posmatch", "9b"), ("category", "9c"), ("direction", "8b")):
        f = f"{d}/judged_{tag}_{task}.jsonl"
        if os.path.exists(f):
            from nlt.evals.judge_batch import summarize_records
            recs = [json.loads(l) for l in open(f) if l.strip()]; s = summarize_records(recs, task)
            v = next((vv for k, vv in s.items() if k.startswith("verdict")), None)
            if v: out["verdicts"][row] = v
            out["numbers"][f"{task}"] = s.get("accuracy", s.get("share_yes", s.get("mean")))
    c = _load(f"{d}/corr_{tag}.json")
    if c:
        out["numbers"]["rho_magnitude_vs_kl_skip"] = c.get("spearman_rho"); out["numbers"]["rho_ntokens_vs_kl_skip"] = c.get("rho_ntokens_vs_kl_skip")
        if "verdict_8a" in c: out["verdicts"]["8a"] = c["verdict_8a"]
    return out


def depth_gate(d):
    """4a/4b from infra's bits json(s): exact and proxy told-depth gain"""
    res = {}
    for f in sorted(glob.glob(f"{d}/../results/bits_*.json")) + sorted(glob.glob(f"{d}/bits_*.json")):
        b = _load(f)
        if not b: continue
        for name, c in b.get("critics", {}).items():
            if c.get("cond") == "depth" and c.get("exact_pmi_bits"):
                ex = c["exact_pmi_bits"]["mean"]; res[name] = {"exact_told_depth_gain_bits": ex, "proxy_told_depth_gain_bits": c.get("proxy_pmi_bits", {}).get("mean"), "file": f,
                                                                 "verdict_4a": "PASS" if ex <= 7 else ("WARN" if ex <= 20 else "FAIL")}
    return res


def to_markdown(table, gate):
    tags = [t["tag"] for t in table]; lines = ["| row | " + " | ".join(tags) + " |", "|---|" + "---|" * len(tags)]
    for r in ROWS:
        vals = [t["verdicts"].get(r, "") for t in table]
        if any(vals): lines.append(f"| {r} | " + " | ".join(vals) + " |")
    lines.append(""); lines.append("| number | " + " | ".join(tags) + " |"); lines.append("|---|" + "---|" * len(tags))
    keys = sorted({k for t in table for k in t["numbers"] if not isinstance(t["numbers"][k], (dict, list))})
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(("" if t["numbers"].get(k) is None else (f"{t['numbers'][k]:.3g}" if isinstance(t["numbers"][k], (int, float)) else str(t["numbers"][k]))) for t in table) + " |")
    if gate: lines.append(""); lines.append("told-depth gate (4a): " + json.dumps(gate))
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--tags", required=True); ap.add_argument("--out", required=True); ap.add_argument("--md")
    a = ap.parse_args(); table = [collect(a.dir, t) for t in a.tags.split(",")]; gate = depth_gate(a.dir)
    json.dump({"table": table, "depth_gate": gate}, open(a.out, "w"), indent=1, default=str); md = to_markdown(table, gate); print(md)
    if a.md: open(a.md, "w").write(md + "\n")
