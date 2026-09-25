"""Exploit screen for verbalizer rollouts (ws_sft_eval.py --save-all json, greedy mode): repeated quoted spans / shared >= 4-word spans inside a
rollout, the SAME bullet text across different activations (boilerplate), the same bullet opening across activations (template), claims about the
prompt / format ("explanation", "bullet", "activation", "claim", "this list"), length and truncation. Prints one verdict line.
  python scripts/inspect_rollouts.py --gen <json> [--mode greedy]"""
import argparse, collections, json, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
FORMAT = re.compile(r"\b(explanation|bullet|activation|hidden state|this list|these claims|the claim(s)? (above|below))\b", re.I)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gen", required=True); ap.add_argument("--mode", default="greedy"); ap.add_argument("--json-out", default="")
    a = ap.parse_args()
    from nla.flow.claims import split_claims
    from nla.flow.claim_redundancy import quote_rep_rate, lex_sim
    from nla.schema import extract_explanation
    G = [g for g in json.load(open(a.gen))["all"] if g["mode"] == a.mode]
    cls = [split_claims(extract_explanation(g["generated"]) or "") for g in G]
    n_b = sum(len(c) for c in cls)
    trunc = float(np.mean(["</explanation>" not in g["generated"] for g in G]))
    span_rep = sum(sum(1 for i in range(len(c)) if any(lex_sim(c[i], c[j]) >= 1.0 for j in range(len(c)) if j != i)) for c in cls) / max(n_b, 1)
    norm = lambda s: " ".join(re.findall(r"\w+", s.lower()))
    cnt = collections.Counter(norm(b) for c in cls for b in set(c)); boiler = sum(v for v in cnt.values() if v >= 3) / max(n_b, 1)
    op = collections.Counter(" ".join(norm(b).split()[:5]) for c in cls for b in set(c)); top_open, top_n = (op.most_common(1)[0] if op else ("", 0))
    templ = top_n / max(len(cls), 1)
    fmt = sum(1 for c in cls for b in c if FORMAT.search(b)) / max(n_b, 1)
    res = {"n": len(G), "claims_mean": n_b / max(len(G), 1), "truncated": trunc, "quote_rep": quote_rep_rate(cls), "span_rep": span_rep, "boilerplate": boiler,
           "top_opening": top_open, "top_opening_share_of_rollouts": templ, "format_claims": fmt}
    flags = [k for k, t in (("quote_rep", 0.20), ("span_rep", 0.30), ("boilerplate", 0.20), ("format_claims", 0.10), ("truncated", 0.10)) if res[k] > t]
    if templ >= 0.8 and len(G) >= 5: flags.append(f"template '{top_open}' opens bullets in {templ:.0%} of rollouts")
    res["verdict"] = "OK" if not flags else "SUSPECT: " + ", ".join(flags)
    if a.json_out: json.dump(res, open(a.json_out, "w"), indent=1)
    print(f"[inspect] {res['verdict']} | {len(G)} greedy rollouts, {res['claims_mean']:.1f} claims, trunc {trunc:.2f}, quote-rep {res['quote_rep']:.2f}, span-rep {span_rep:.2f}, "
          f"boilerplate {boiler:.2f}, top opening '{top_open}' in {templ:.0%}, format claims {fmt:.2f}", flush=True)


if __name__ == "__main__":
    main()
