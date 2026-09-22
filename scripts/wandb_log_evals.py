"""Mirror the held-out eval series of every RL arm into wandb (one run per arm, name <arm>_evals, project nla-exp-qwen36_27b), so the
judge-based metrics live next to the training curves: suite (FVE under the frozen critic, NLA/harness judges), claim judge (fabricated /
accurate claims, precision) and position match. Idempotent: each arm's eval run is resumed by the id stored in data/wandb_eval_runs.json and
re-logged from scratch (define_metric step); safe to call from the refresh loops."""
import json, os, sys
import wandb
R = sys.argv[1] if len(sys.argv) > 1 else "/home/celeste/shared/reports/nla-flow-prior"
P = "octahedral-systems/nla-exp-qwen36_27b"
E = json.load(open(f"{R}/data/arm_evals.json")); J = json.load(open(f"{R}/data/judge_batch.json")) if os.path.exists(f"{R}/data/judge_batch.json") else {}
S = json.load(open(f"{R}/data/samedoc_match.json")) if os.path.exists(f"{R}/data/samedoc_match.json") else {}
ids_p = f"{R}/data/wandb_eval_runs.json"; ids = json.load(open(ids_p)) if os.path.exists(ids_p) else {}
SUITE = ["fve_frozen_sft_critic", "tj_hallucination", "hj_hallucination", "hj_informativeness", "tj_specificity", "tj_coherence", "tj_unique_info", "tj_writing_quality", "tj_repetitiveness"]
CLAIM = ["bad_per_expl", "supported_per_expl", "claim_precision", "contradicted_per_expl", "number_precision", "hallucination_1_10"]
arms = sorted({r["arm"] for r in E["rows"]} | {k.split(":")[0] for k in J} | {k.split(":")[0] for k in S})
only = set(sys.argv[2:]) if len(sys.argv) > 2 else None
for arm in arms:
    if only and arm not in only: continue
    rows = {}
    for r in E["rows"]:
        if r["arm"] == arm:
            for k in SUITE:
                if r.get(k) is not None: rows.setdefault(int(r["step"]), {})[f"evals/{k}"] = float(r[k])
    for k, v in J.items():
        a, st = k.split(":");
        if a == arm and v.get("n", 0) > 0:
            for kk in CLAIM:
                if v.get(kk) is not None: rows.setdefault(int(st), {})[f"evals/{kk}"] = float(v[kk])
    for k, v in S.items():
        a, st = k.split(":")
        if a == arm and v.get("n", 0) > 0 and v.get("acc") is not None: rows.setdefault(int(st), {})["evals/position_match_acc"] = float(v["acc"])
    if not rows: continue
    run = wandb.init(project=P.split("/")[1], entity=P.split("/")[0], name=f"{arm}_evals", id=ids.get(arm), resume="allow", reinit=True, tags=["evals", arm],
                     config={"arm": arm, "source": "report data/*.json (736 clean1 held-out docs)"})
    run.define_metric("evals/step"); run.define_metric("evals/*", step_metric="evals/step")
    for st in sorted(rows): run.log({"evals/step": st, **rows[st]})
    ids[arm] = run.id; run.finish(); print(f"{arm}_evals: {len(rows)} steps -> {run.url}")
json.dump(ids, open(ids_p, "w"), indent=1)
