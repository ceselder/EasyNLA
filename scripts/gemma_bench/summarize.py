"""Collect every benchmark result from the volume and print the comparison table (+ write data/results_table.json).

    python3 scripts/gemma_bench/summarize.py [--fetch] [names...]

Columns: plain prompts/s, decode tok/s, prefill tok/s (all prompt tokens), g2 stage-A positions/s + decode tok/s, stage-B renders/s,
parse-fail per set, engine start seconds; joined with the Sonnet judge summary (~/nla-exp-logs/gemma_engine/judge/summary.json) when present."""
import glob, json, os, subprocess, sys

RES = os.path.expanduser("~/nla-exp-logs/gemma_engine/results"); os.makedirs(RES, exist_ok=True)
JUDGE = os.path.expanduser("~/nla-exp-logs/gemma_engine/judge/summary.json")


def fetch(names=None):
    ls = subprocess.run(["modal", "volume", "ls", "nla-glp", "/scale/bench/results"], capture_output=True, text=True).stdout
    files = [l.strip().split()[-1] for l in ls.splitlines() if l.strip().endswith(".json")]
    for f in files:
        n = os.path.basename(f)[:-5]
        if names and n not in names: continue
        subprocess.run(["modal", "volume", "get", "nla-glp", f"/scale/bench/results/{os.path.basename(f)}", f"{RES}/{n}.json", "--force"], capture_output=True)


def g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d: return default
        d = d[k]
    return d


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--fetch" in sys.argv: fetch(args or None)
    J = json.load(open(JUDGE)) if os.path.exists(JUDGE) else {}
    rows = []
    for p in sorted(glob.glob(f"{RES}/*.json")):
        r = json.load(open(p)); n = os.path.basename(p)[:-5]
        if args and n not in args: continue
        S = r.get("sets", {}); j = J.get(n, {})
        rows.append({"name": n, "rc": r.get("rc"), "load_s": g(r, "load_seconds"),
                     "plain_prompts_s": g(S, "plain", "prompts_per_s"), "plain_decode_tok_s": g(S, "plain", "decode_tok_per_s"), "plain_prefill_tok_s": g(S, "plain", "prefill_tok_per_s"),
                     "plain_uncached_prefill_tok_s": g(S, "plain", "uncached_prefill_tok_per_s"), "plain_out_mean": g(S, "plain", "out_mean"), "plain_parse_fail": g(S, "plain", "parse_fail"),
                     "g2a_pos_s": g(S, "g2a", "prompts_per_s"), "g2a_decode_tok_s": g(S, "g2a", "decode_tok_per_s"), "g2a_out_mean": g(S, "g2a", "out_mean"), "g2a_parse_fail": g(S, "g2a", "parse_fail"),
                     "g2b_renders_s": g(S, "g2b", "prompts_per_s"), "g2b_decode_tok_s": g(S, "g2b", "decode_tok_per_s"), "g2b_qc_fail": g(S, "g2b", "parse_fail"),
                     "judge_parse_fail": g(S, "judge", "parse_fail"), "judge_claim_precision": j.get("claim_precision"), "judge_claims_per_expl": j.get("claims_per_expl"),
                     "judge_names_per_expl": j.get("names_per_expl"), "judge_numbers_per_expl": j.get("numbers_per_expl"), "judge_paired_dprec": j.get("paired_delta_precision_vs_bf16pilot"),
                     "profile_top": g(r, "profile", "top"), "profile_tok_s": g(r, "profile", "tok_per_s"), "profile_gpu_busy_frac": g(r, "profile", "gpu_busy_frac"), "error": r.get("error")})
    cols = [("name", 30, "s"), ("plain_prompts_s", 8, ".1f"), ("plain_decode_tok_s", 8, ".0f"), ("plain_prefill_tok_s", 8, ".0f"), ("g2a_pos_s", 7, ".1f"), ("g2a_decode_tok_s", 8, ".0f"),
            ("g2b_renders_s", 7, ".1f"), ("plain_parse_fail", 7, ".4f"), ("g2a_parse_fail", 7, ".4f"), ("g2b_qc_fail", 7, ".3f"), ("judge_claim_precision", 7, ".3f"), ("judge_claims_per_expl", 7, ".2f"), ("load_s", 6, ".0f")]
    hdr = "".join(("%-" if f == "s" else "%") + f"{w}s " % () for _, w, f in cols)
    print(" ".join(("%-*s" if f == "s" else "%*s") % (w, c[:w]) for c, w, f in cols))
    for r in rows:
        cells = []
        for c, w, f in cols:
            v = r.get(c)
            cells.append(("%-*s" % (w, str(v)[:w])) if f == "s" else ("%*s" % (w, ("%" + f) % v if isinstance(v, (int, float)) else "-")))
        print(" ".join(cells) + (f"  ERROR {r['error'][:80]}" if r.get("error") else ""))
    os.makedirs(os.path.expanduser("~/nla-exp-logs/gemma_engine/data"), exist_ok=True)
    json.dump(rows, open(os.path.expanduser("~/nla-exp-logs/gemma_engine/data/results_table.json"), "w"), indent=1)
    for r in rows:
        if r.get("profile_top"):
            print(f"\n== kernel breakdown {r['name']}: {r['profile_tok_s']:.0f} tok/s in the profiled window, GPU busy {r['profile_gpu_busy_frac']:.2f}")
            for k in r["profile_top"][:25]: print("   %5.1f%%  %8.1f ms  %6d calls  %s" % (100 * k["frac"], k["ms"], k["calls"], k["kernel"][:100]))


if __name__ == "__main__":
    main()
