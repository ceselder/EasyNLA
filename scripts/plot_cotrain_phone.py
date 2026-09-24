"""Phone-legible head-to-head of the critic co-training arms (rlQ36_co_*128) against the reference rlQ36_mse128 (critic SFT on the best
rollout per group); same RL recipe, only how the MSE critic co-trains differs (the fast-critic arm also halves the verbalizer lr).
Reads ~/shared/reports/nla-flow-prior/data/cotrain_evals.json (built by plot_twin_evals.py cotrain), data/judge_batch.json (per-row
claim-judge verdicts, for the paired per-document bootstrap vs the reference) and data/cotrain_tripwire.json (scripts/cotrain_tripwire.py).
Writes cotrain_phone.png/pdf + data/cotrain_phone.json: 3x2 grid (harness hallucination, fabricated claims, accurate claims, claim
precision, FVE under the frozen SFT critic, paraphrase FVE gap under each arm's live critic)."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.path.expanduser("~/shared/reports/nla-flow-prior"); D = f"{R}/data"
E = json.load(open(f"{D}/cotrain_evals.json")) if os.path.exists(f"{D}/cotrain_evals.json") else {}
J = json.load(open(f"{D}/judge_batch.json")) if os.path.exists(f"{D}/judge_batch.json") else {}
T = json.load(open(f"{D}/cotrain_tripwire.json")) if os.path.exists(f"{D}/cotrain_tripwire.json") else {}
REF = "rlQ36_mse128"
ARMS = [(REF, "reference: critic SFT on best rollout", "#2563eb", "--"), ("rlQ36_co_all128", "uniform (NLA paper)", "#64748b", "-"),
        ("rlQ36_co_lagav128", "fast critic + slow verbalizer", "#059669", "-"), ("rlQ36_co_lagar128", "lagged anchored critic", "#0e7490", "-"),
        ("rlQ36_co_para128", "paraphrase-augmented critic", "#c2410c", "-"), ("rlQ36_co_awr128", "advantage-weighted critic", "#7c3aed", "-"),
        ("rlQ36_co_dpo128", "pairwise-ranking critic", "#db2777", "-")]
PAN = [("hj_hallucination", "harness judge: hallucination (lower better)"), ("bad_per_expl", "fabricated specific claims / explanation"),
       ("supported_per_expl", "accurate specific claims / explanation"), ("claim_precision", "claim precision"),
       ("fve_frozen_sft_critic", "FVE % under the frozen SFT critic"), (None, "paraphrase FVE gap under the arm's own critic")]


def paired(arm, lo, hi):
    """per-document mean over matched steps in [lo, hi] of (arm - reference) for accurate/fabricated/precision/hallucination; bootstrap CI"""
    def per_doc(a):
        out = {}
        for s in range(lo, hi + 1, 20):
            for row, r in J.get(f"{a}:{s}", {}).get("per_row", {}).items():
                cl = r.get("claims") or []; sup = sum(c["verdict"] == "supported" for c in cl)
                out.setdefault(int(row), []).append([sup, len(cl) - sup, sup / len(cl) if cl else np.nan, r.get("h", np.nan)])
        return {k: np.nanmean(np.array(v, float), 0) for k, v in out.items() if len(v) >= max(1, (hi - lo) // 20)}
    A, M = per_doc(arm), per_doc(REF); docs = sorted(set(A) & set(M))
    if len(docs) < 50: return None
    X = np.array([A[d] - M[d] for d in docs]); rng = np.random.default_rng(0); res = {"n_docs": len(docs), "steps": [lo, hi]}
    for j, k in enumerate(["accurate", "fabricated", "precision", "hallucination"]):
        x = X[:, j][~np.isnan(X[:, j])]; bs = [rng.choice(x, len(x)).mean() for _ in range(2000)]
        res[k] = [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return res


plt.rcParams.update({"font.size": 12, "axes.titlesize": 13})
fig, ax = plt.subplots(3, 2, figsize=(11, 12)); ax = ax.ravel(); out = {"panels": {}, "paired_vs_ref": {}}
for i, (k, title) in enumerate(PAN):
    for arm, lab, col, ls in ARMS:
        if k is None:   # tripwire: live-critic paraphrase gap per step
            pts = [(r["step"], r["live_gap_para"]) for r in T.get(arm, []) if r.get("live_gap_para") is not None]
        else:
            pts = [tuple(p) for p in E.get(arm, {}).get(k, [])]
        if not pts: continue
        x, y = zip(*pts); ax[i].plot(x, y, ls=ls, color=col, lw=2.0, marker="o", ms=3, label=lab)
        out["panels"].setdefault(k or "tripwire_live_gap_para", {})[arm] = pts
    ax[i].set_title(title); ax[i].set_xlabel("RL step (1024 rollouts each)"); ax[i].grid(alpha=.3)
ax[0].legend(fontsize=9.5, loc="best")
steps = [s for s, _ in E.get(REF, {}).get("claim_precision", [])]
lo, hi = 100, 340
for arm, *_ in ARMS[1:]:
    have = [s for s, _ in E.get(arm, {}).get("claim_precision", [])]
    top = min(hi, max(have)) if have else 0
    if top >= lo: out["paired_vs_ref"][arm] = paired(arm, lo, top)
best = None
for arm, v in out["paired_vs_ref"].items():
    if v and (best is None or v["hallucination"][0] < out["paired_vs_ref"][best]["hallucination"][0]): best = arm
lab = dict((a, l) for a, l, *_ in ARMS)
if best:
    v = out["paired_vs_ref"][best]
    head = (f"Critic co-training variants vs the reference: best on judged hallucination is {lab[best]} "
            f"({v['hallucination'][0]:+.2f} [{v['hallucination'][1]:+.2f}, {v['hallucination'][2]:+.2f}] vs reference, steps {v['steps'][0]}–{v['steps'][1]})")
else:
    head = "Critic co-training variants vs the reference: too early for the paired comparison (needs judged checkpoints from step 100)"
fig.suptitle(head + "\nExperiment: identical RL recipe (128 × 8, CISPO, KL 0.01, lr 2e-5); only how the MSE critic co-trains differs; 736 held-out docs",
             fontsize=12, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.94))
for ext in ("png", "pdf"): fig.savefig(f"{R}/cotrain_phone.{ext}", dpi=150 if ext == "png" else None)
json.dump(out, open(f"{D}/cotrain_phone.json", "w"), indent=1); print("paired:", json.dumps(out["paired_vs_ref"])[:800])
