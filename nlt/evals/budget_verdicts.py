"""Verdict tables from infra's bits jsons (nlt.eval_bits.run output: {"critics": {"<critic>@<set>": {...}}} or data/info_budget.json).
Per critic x set: z (exact PMI vs the blind prior), dm (depth-matched shuffle), rp (random pair), shuf_words, mask_next, and the redteam
decomposition  form = dm - shuf | depth-generic = dm - rp | content = z - dm , P(z > dm), rp-corrected ratio dm_c/orig_c, by band.
Verdicts (EVALS.md v1.3): 3a bits(rp)/bits(z) <= 0.10 ; 3b/4e on rp-corrected bits: dm_c/orig_c <= 0.25 PASS, <= 0.5 WARN ; content gate
P(z > dm) >= 0.75 PASS, >= 0.60 WARN ; form-vs-content P(z > shuf) >= 0.75 ; presence flag |rp| > 2 bits and > 2 sem.

  python -m nlt.evals.budget_verdicts --json /vol/results/bits_text_union_pooled_n.json [--json more.json ...] --out data/verdicts_<tag>.json [--md verdicts.md]
"""
from __future__ import annotations
import argparse, json, math, os


def _mean(c, k):
    v = c.get(k)
    return v["mean"] if isinstance(v, dict) and "mean" in v else (float(v) if isinstance(v, (int, float)) else float("nan"))


def _sem(c, k):
    v = c.get(k)
    return v.get("sem", float("nan")) if isinstance(v, dict) else float("nan")


def row_from_critic(name, c):
    z, zs = _mean(c, "exact_pmi_bits"), _sem(c, "exact_pmi_bits")
    if math.isnan(z): z, zs = c.get("exact_bits_mean", float("nan")), c.get("exact_bits_sem", float("nan"))          # info_budget.json schema
    dm = _mean(c, "shuffle_exact_pmi_bits")
    if math.isnan(dm): dm = c.get("dm_shuffle_exact_bits_mean") if c.get("dm_shuffle_exact_bits_mean") is not None else float("nan")
    rp = _mean(c, "rp_exact_pmi_bits"); sh = _mean(c, "shuf_words_exact_pmi_bits"); mk = _mean(c, "mask_next_exact_pmi_bits")
    p_dm = c.get("frac_z_beats_dm", float("nan")); p_sh = c.get("frac_z_beats_shuf_words", float("nan")); p_rp = c.get("frac_z_beats_rp", float("nan"))
    ntok = c.get("n_tokens_mean", float("nan")); n = c.get("n_rows", c.get("n"))
    r = {"critic": name.split("@")[0], "set": name.split("@")[-1], "n": n, "step": c.get("step"), "cond": c.get("cond"), "z": z, "z_sem": zs, "dm": dm, "rp": rp, "shuf_words": sh, "mask_next": mk,
         "form": dm - sh, "depth_generic": dm - rp, "content": z - dm, "p_z_gt_dm": p_dm, "p_z_gt_rp": p_rp, "p_z_gt_shuf": p_sh, "n_tokens": ntok,
         "bits_per_token": (z / ntok) if ntok and not math.isnan(ntok) and ntok > 0 else float("nan"),
         "content_per_token": ((z - dm) / ntok) if ntok and not math.isnan(ntok) and ntok > 0 else float("nan")}
    # rp-corrected ratio
    oc, dc = z - rp, dm - rp
    r["ratio_dm_over_z_raw"] = dm / z if z not in (0, float("nan")) and not math.isnan(z) and z != 0 else float("nan")
    r["ratio_dm_over_z_rp_corrected"] = dc / oc if oc and not math.isnan(oc) and oc != 0 else float("nan")
    r["presence_offset_flag"] = bool(not math.isnan(rp) and abs(rp) > 2.0 and (math.isnan(zs) or abs(rp) > 2 * zs))
    v = {}
    if not math.isnan(rp) and not math.isnan(z) and z != 0: rr = abs(rp / z); v["3a"] = "PASS" if rr <= 0.10 else ("WARN" if rr <= 0.25 else "FAIL")
    rc = r["ratio_dm_over_z_rp_corrected"]
    if not math.isnan(rc): v["3b_rp_corrected"] = "PASS" if rc <= 0.25 else ("WARN" if rc <= 0.50 else "FAIL"); ex = (1 - rc) / max(rc, 1e-9); v["4e_rp_corrected"] = "PASS" if ex >= 3 else ("WARN" if ex >= 1 else "FAIL")
    if not math.isnan(p_dm): v["content_P"] = "PASS" if p_dm >= 0.75 else ("WARN" if p_dm >= 0.60 else "FAIL")
    if not math.isnan(p_sh): v["form_vs_content"] = "PASS" if p_sh >= 0.75 else ("WARN" if p_sh >= 0.60 else "FAIL")
    r["verdicts"] = v
    # by band (content per band when available)
    bb = c.get("by_band")
    if isinstance(bb, dict):
        r["by_band"] = {}
        for band, val in bb.items():
            if isinstance(val, dict):
                r["by_band"][band] = {k: val.get(k) for k in ("mean", "sem", "n", "dm", "rp", "shuffle", "z_dm", "z_rp") if k in val}
    return r


def load_rows(paths):
    rows = []
    for p in paths:
        d = json.load(open(p)); crit = d.get("critics", {})
        for name, c in crit.items():
            if not isinstance(c, dict) or c.get("cond") not in ("text", None): continue
            if c.get("cond") is None and "exact_pmi_bits" not in c and "exact_bits_mean" not in c: continue
            r = row_from_critic(name, c); r["source_file"] = os.path.basename(p); rows.append(r)
    return rows


def to_markdown(rows):
    f = lambda x, p=2: ("" if x is None or (isinstance(x, float) and math.isnan(x)) else (f"{x:.{p}f}" if isinstance(x, (int, float)) else str(x)))
    lines = ["| critic | set | n | z | dm | rp | shuf | mask | form | depthG | content | P(z>dm) | P(z>shuf) | tok | content/tok | 3a | 3b_c | content_P | form_vs_content |", "|" + "---|" * 19]
    for r in rows:
        v = r["verdicts"]
        lines.append(f"| {r['critic']} | {r['set']} | {r['n']} | {f(r['z'])} | {f(r['dm'])} | {f(r['rp'])} | {f(r['shuf_words'])} | {f(r['mask_next'])} | {f(r['form'],1)} | {f(r['depth_generic'])} | {f(r['content'])} | {f(r['p_z_gt_dm'])} | {f(r['p_z_gt_shuf'])} | {f(r['n_tokens'],0)} | {f(r['content_per_token'],3)} | {v.get('3a','')} | {v.get('3b_rp_corrected','')} | {v.get('content_P','')} | {v.get('form_vs_content','')} |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--json", action="append", required=True); ap.add_argument("--out", required=True); ap.add_argument("--md")
    a = ap.parse_args(); rows = load_rows(a.json)
    json.dump({"rows": rows, "sources": a.json}, open(a.out, "w"), indent=1, default=str); md = to_markdown(rows); print(md)
    if a.md: open(a.md, "w").write(md + "\n")
