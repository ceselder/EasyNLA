"""Score a dump of path-facts generations against the ground-truth facts: parse each categorical fact (kind / when / peak kind / peak when / route)
and the attention percentage from the text with the templates' own vocabulary, and report exact accuracy per fact against (a) chance,
(b) the majority class, (c) the GAP-INFORMED majority (the best a model that knows only j - i could do). Runs locally on parquets pulled
from the volume (small). Also accepts several dumps -> one JSON table.

  python -m nlt.path.score_facts --facts pathfacts_val.parquet --dump none=dump_none.parquet --dump count=dump_count.parquet --dump path=dump_path.parquet --out data/path_facts_scores.json
"""
from __future__ import annotations
import argparse, json, re
import numpy as np
import pandas as pd

RX = {
    "kind": [(r"mostly attention", "attention"), (r"mostly the MLPs?", "mlp"), (r"even mix", "mixed")],
    "when": [(r"early in the stretch", "early"), (r"around the middle of the stretch", "middle"), (r"late in the stretch", "late"), (r"single step", "single")],
    "peak_kind": [(r"biggest single push (?:came )?from attention", "attention"), (r"biggest single push (?:came )?from an MLP", "mlp")],
    "peak_when": [(r"near the start", "early"), (r"in the middle\b(?! of the stretch)", "middle"), (r"near the end", "late"), (r"in that one step", "single")],
    "route": [(r"a direct path", "direct"), (r"somewhat roundabout", "roundabout"), (r"very roundabout", "very_roundabout")],
}
RX_PCT = re.compile(r"(\d{1,3})% (?:of the shift came from )?attention")


def parse(text: str) -> dict:
    out = {}
    for fact, pats in RX.items():
        hits = [lab for rx, lab in pats if re.search(rx, text or "", re.I)]
        out[fact] = hits[0] if len(hits) == 1 else (hits[0] if hits else None)     # first match; ambiguous texts still count their first claim
        out[fact + "_n"] = len(hits)
    m = RX_PCT.search(text or ""); out["pct"] = int(m.group(1)) if m else None
    return out


def score(facts: pd.DataFrame, dump: pd.DataFrame, name: str) -> dict:
    d = dump.drop_duplicates("pair_id").merge(facts, on="pair_id", suffixes=("", "_true"), how="inner")
    P = pd.DataFrame([parse(t) for t in d["text"].values]); res = {"arm": name, "n": int(len(d))}
    multi = (d["when"].values != "single")                        # the peak facts are only stated for stretches of more than one step
    for fact in RX:
        true = d[fact].values; pred = P[fact].values; m = multi if fact.startswith("peak") else np.ones(len(d), bool)
        res[f"acc_{fact}"] = float(np.mean((pred == true)[m])); res[f"parsed_{fact}"] = float(np.mean([p is not None for p in pred[m]])); res[f"n_{fact}"] = int(m.sum())
    for cls in np.unique(d["kind"].values):                      # accuracy of the attention-vs-MLP fact by TRUE class (the rare attention class is the tell)
        mk = d["kind"].values == cls; res[f"acc_kind_true_{cls}"] = float(np.mean((P["kind"].values == cls)[mk])); res[f"n_kind_true_{cls}"] = int(mk.sum())
    if "attn_share" in d:
        tp = np.clip(d["attn_share"].values, 0, 1) * 100; pp = P["pct"].values.astype(float)
        ok = ~np.isnan(pp); res["pct_mae"] = float(np.mean(np.abs(pp[ok] - tp[ok]))) if ok.any() else None; res["pct_parsed"] = float(ok.mean())
    res["tokens_mean"] = float(d["n_tokens"].mean()) if "n_tokens" in d else None
    return res


def baselines(facts: pd.DataFrame) -> dict:
    out = {"arm": "baselines", "n": int(len(facts))}
    for fact in RX:
        F = facts[facts["when"] != "single"] if fact.startswith("peak") else facts
        g = F["j"].astype(int) - F["i"].astype(int)
        v = F[fact].values; classes, counts = np.unique(v, return_counts=True)
        out[f"chance_{fact}"] = float(1 / len(classes)); out[f"majority_{fact}"] = float(counts.max() / len(v))
        out[f"gapmajority_{fact}"] = float(sum(F[fact][g == gg].value_counts().max() for gg in np.unique(g)) / len(v)); out[f"n_{fact}"] = int(len(F))
    g = facts["j"].astype(int) - facts["i"].astype(int)
    tp = np.clip(facts["attn_share"].values, 0, 1) * 100; out["pct_mae_predict_mean"] = float(np.mean(np.abs(tp - tp.mean())))
    out["pct_mae_predict_gap_mean"] = float(np.mean([abs(x - tp[g == gg].mean()) for x, gg in zip(tp, g)]))
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("--facts", required=True); p.add_argument("--dump", action="append", default=[], help="name=parquet"); p.add_argument("--out", required=True)
    a = p.parse_args()
    facts = pd.read_parquet(a.facts); rows = [baselines(facts)]
    for spec in a.dump:
        name, path = spec.split("=", 1); rows.append(score(facts, pd.read_parquet(path), name))
    json.dump({"rows": rows, "facts": a.facts, "note": "exact accuracy of each categorical fact parsed from the generated text vs the fact computed from the writes; "
                                                "gapmajority = per-gap majority class = what a model knowing only j-i could reach; peak facts scored on stretches of > 1 step only"}, open(a.out, "w"), indent=1)
    cols = ["arm", "n"] + [f"acc_{f}" for f in RX] + ["pct_mae"]
    print("\t".join(cols))
    for r in rows:
        if r["arm"] == "baselines":
            for kind in ("chance", "majority", "gapmajority"): print("\t".join([kind, str(r["n"])] + [f"{r[f'{kind}_{f}']:.3f}" for f in RX] + [f"{r['pct_mae_predict_gap_mean'] if kind == 'gapmajority' else r['pct_mae_predict_mean']:.1f}" if kind != "chance" else "-"]))
        else: print("\t".join([r["arm"], str(r["n"])] + [f"{r[f'acc_{f}']:.3f}" for f in RX] + [f"{r['pct_mae']:.1f}" if r.get("pct_mae") is not None else "-"]))
    print("->", a.out)


if __name__ == "__main__":
    main()
