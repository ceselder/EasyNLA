"""One table across sources for a critic from the controls summaries (summarize_scored output = data/verdicts_<critic>_<src>_controls.json).
  python -m nlt.evals.controls_table --dir data --critic union_pooled_null --out data/verdicts_union_pooled_null_controls_table.json [--md ...]
"""
from __future__ import annotations
import argparse, glob, json, os, re


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--critic", required=True); ap.add_argument("--out", required=True); ap.add_argument("--md")
    a = ap.parse_args(); rows = []
    for f in sorted(glob.glob(os.path.join(a.dir, f"verdicts_{a.critic}_*_controls.json"))):
        src = re.sub(rf"^verdicts_{re.escape(a.critic)}_(.*)_controls\.json$", r"\1", os.path.basename(f)); s = json.load(open(f))
        g = lambda v, k="bits_mean": (s.get(v) or {}).get(k)
        r = {"source": src, "n": s["orig"]["n"], "orig": g("orig"), "orig_ci": s["orig"]["ci95"], "dm": g("dm"), "rp": g("rp"), "shuf_words": g("shuf_words"), "copy": g("copy"), "src_desc": g("src_desc"), "wrong_j": g("wrong_j"),
             "p_orig_gt_dm": g("dm", "p_orig_higher"), "p_orig_gt_rp": g("rp", "p_orig_higher"), "p_orig_gt_shuf": g("shuf_words", "p_orig_higher"), "p_orig_gt_src": g("src_desc", "p_orig_higher"), "p_orig_gt_wrong_j": g("wrong_j", "p_orig_higher"),
             "bits_per_token": s["orig"].get("bits_per_token_mean"), "tokens_median": s["orig"].get("tokens_median"), "share_nonpositive": s["orig"].get("share_nonpositive"),
             "content": (g("orig") or 0) - (g("dm") or 0), "form": ((g("dm") or 0) - g("shuf_words")) if g("shuf_words") is not None else None, "depth_generic": (g("dm") or 0) - (g("rp") or 0),
             "content_by_band": {b: round(s["by_band"]["orig"][b]["bits_mean"] - s["by_band"]["dm"][b]["bits_mean"], 2) for b in s.get("by_band", {}).get("orig", {}) if b in s["by_band"].get("dm", {})},
             "content_by_gap": {gp: round(s["by_gap_bin"]["orig"][gp]["bits_mean"] - s["by_gap_bin"]["dm"][gp]["bits_mean"], 2) for gp in s.get("by_gap_bin", {}).get("orig", {}) if gp in s["by_gap_bin"].get("dm", {})},
             "verdicts": {k.replace("verdict_", ""): v for k, v in s.items() if k.startswith("verdict_")}}
        rows.append(r)
    rows.sort(key=lambda r: -r["content"])
    f = lambda x, p=2: ("" if x is None else (f"{x:.{p}f}" if isinstance(x, (int, float)) else str(x)))
    md = [f"### {a.critic}: controls on the fixed eval set (exact bits vs the blind prior; content = orig − dm)", "",
          "| source | n | orig | dm | rp | shuf | copy | src | wrong_j | content | form | depthG | P(>dm) | P(>rp) | P(>shuf) | bits/tok | content pre/ws/motor |", "|" + "---|" * 17]
    for r in rows:
        cb = r["content_by_band"]; md.append(f"| {r['source']} | {r['n']} | {f(r['orig'])} | {f(r['dm'])} | {f(r['rp'])} | {f(r['shuf_words'])} | {f(r['copy'],1)} | {f(r['src_desc'])} | {f(r['wrong_j'],0)} | {f(r['content'])} | {f(r['form'],1)} | {f(r['depth_generic'])} | {f(r['p_orig_gt_dm'])} | {f(r['p_orig_gt_rp'])} | {f(r['p_orig_gt_shuf'])} | {f(r['bits_per_token'],3)} | {f(cb.get('pre'))}/{f(cb.get('workspace'))}/{f(cb.get('motor'))} |")
    json.dump({"critic": a.critic, "rows": rows}, open(a.out, "w"), indent=1, default=str); print("\n".join(md))
    if a.md: open(a.md, "w").write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
