"""Exact-view RL curve (nlt-27b-olens): the policy's held-out greedy dumps at RL steps 0, 20, 40, ... scored with EXACT Heun-32 bits under the FROZEN judge and under an
OTHER-lineage critic, next to the crafted teacher on the same 128 held-out pairs. Orchestrator 2026-09-25 10:10 UTC: the exact view decides stop / keep and the RL headline; the FM
view (rl_verbalizer's own eval) stays as the cheap per-step monitor. Rule: if after ~40 steps the exact-view gain over step 0 is within one sem under BOTH judges -> stop the run
("FM-reward RL does not move exact content at this scale").
Reads data/rl_<tag>_exact_<frozen|other>_<step>.json -> data/rl_<tag>_exact.json (+ verdict) and fig_rl_<tag>_exact.{png,pdf}.
"""
import argparse, glob, json, os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REP = "/home/celeste/shared/reports/nlt-27b-olens"
C1, C2, C3, CG = "#2b6cb0", "#c05621", "#1a9c6e", "#888888"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="rl_v4b"); ap.add_argument("--min-steps", type=int, default=40); a = ap.parse_args()
    rows = {}
    for f in sorted(glob.glob(f"{REP}/data/rl_{a.tag}_exact_*_*.json")):
        m = re.search(r"_exact_(frozen|other)_(\d+)\.json$", f)
        if not m: continue
        d = json.load(open(f)); s = d["sets"]; j, st = m.group(1), int(m.group(2))
        rows.setdefault(st, {})[j] = {"policy": s["policy"]["content_bits"]["mean"], "policy_sem": s["policy"]["content_bits"]["sem"], "policy_p": s["policy"]["p_z_gt_dm"], "teacher": s["teacher"]["content_bits"]["mean"], "teacher_sem": s["teacher"]["content_bits"]["sem"], "teacher_p": s["teacher"]["p_z_gt_dm"],
                                       "policy_pmi": s["policy"]["pmi_bits"]["mean"] if isinstance(s["policy"]["pmi_bits"], dict) else s["policy"]["pmi_bits"], "tokens": s["policy"]["n_tokens_mean"], "judge": d["ckpt"], "n": s["policy"]["n"]}
    steps = sorted(rows)
    out = {"tag": a.tag, "steps": steps, "rows": {str(k): v for k, v in rows.items()}, "rule": "exact view decides; stop if after ~40 steps the gain over step 0 is within one sem under BOTH the frozen and the other-lineage judge"}
    verdict = "pending"
    if steps and 0 in rows:
        last = max(steps); gains = {}
        for j in ("frozen", "other"):
            if j in rows[0] and j in rows[last]:
                g = rows[last][j]["policy"] - rows[0][j]["policy"]; sem = (rows[last][j]["policy_sem"] ** 2 + rows[0][j]["policy_sem"] ** 2) ** 0.5; gains[j] = {"gain": g, "sem": sem, "significant": g > sem}
        out["gains_at_last"] = {"step": last, **gains}
        if last >= a.min_steps and len(gains) == 2:
            verdict = "MOVES EXACT CONTENT (gain > 1 sem under both judges)" if all(v["significant"] for v in gains.values()) else ("gain under one judge only" if any(v["significant"] for v in gains.values()) else "STOP: FM-reward RL does not move exact content at this scale (gain within one sem under both judges)")
    out["verdict"] = verdict; json.dump(out, open(f"{REP}/data/rl_{a.tag}_exact.json", "w"), indent=1)
    if not steps: print("no exact evals yet"); return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, j, title in ((axes[0], "frozen", "Frozen judge (critic v1b step 500), EXACT Heun-32 bits"), (axes[1], "other", "Other-lineage judge (critic v2 step 3000), EXACT bits")):
        xs = [s for s in steps if j in rows[s]]
        if not xs: continue
        ax.errorbar(xs, [rows[s][j]["policy"] for s in xs], yerr=[rows[s][j]["policy_sem"] for s in xs], fmt="o-", color=C1, lw=2, capsize=3, label="policy (held-out greedy dumps)")
        ax.errorbar(xs, [rows[s][j]["teacher"] for s in xs], yerr=[rows[s][j]["teacher_sem"] for s in xs], fmt="s--", color=C2, lw=1.5, capsize=3, label="crafted teacher, same pairs")
        ax.set_xlabel("RL step"); ax.set_ylabel("content bits (PMI(z) − PMI(z_dm))"); ax.set_title(title, fontsize=12); ax.legend(frameon=False, fontsize=9)
    fig.suptitle(f"Does RL move EXACT content? {a.tag} - {verdict}", fontsize=12, y=1.02); fig.tight_layout()
    fig.savefig(f"{REP}/fig_rl_{a.tag}_exact.png", dpi=150, bbox_inches="tight"); fig.savefig(f"{REP}/fig_rl_{a.tag}_exact.pdf", bbox_inches="tight")
    print("VERDICT", verdict, "|", json.dumps(out.get("gains_at_last")))


if __name__ == "__main__":
    main()
