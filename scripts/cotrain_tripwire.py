"""Co-adaptation tripwire aggregator for the critic co-training arms (rlQ36_co_*128).

The trainer (--tripwire-every N) dumps, at eval steps, the 128 held-out eval explanations and two meaning-preserving variants
(base-model paraphrase, bullet/sentence shuffle) to <save_dir>/tripwire/{orig,para,shuf}/step_*_r0.pt, with the LIVE co-trained
critic's rewards. This script scores each variant directory under the FROZEN SFT critic (nla.flow.score_dumps via the Modal
score_dumps task; incremental, already-scored steps are skipped), then writes per arm and step:
  live FVE (orig / para / shuf) and the gaps orig - para, orig - shuf  (from the dumps: the arm's current critic)
  frozen FVE (orig / para / shuf) and the same gaps                  (the fixed SFT critic)
A live gap that widens over training while the frozen gap stays flat = the co-trained critic has learned to read surface form.
usage: python scripts/cotrain_tripwire.py [--arms a,b,...] [--no-score]   -> ~/shared/reports/nla-flow-prior/data/cotrain_tripwire.json"""
import argparse, json, os, subprocess, sys
ARMS = ["rlQ36_co_all128", "rlQ36_co_lagav128", "rlQ36_co_para128", "rlQ36_co_awr128", "rlQ36_co_dpo128", "rlQ36_co_lagar128"]
V = ("orig", "para", "shuf")
C = "/vol/ckpts/qwen36_27b"; OUT = os.path.expanduser("~/shared/reports/nla-flow-prior/data/cotrain_tripwire.json")
REPO = os.path.expanduser("~/easynla-qwen36"); TMP = "/tmp/cotrain_tripwire"


def vol_ls(path):
    r = subprocess.run(["modal", "volume", "ls", "nla-exp", path], capture_output=True, text=True, timeout=120)
    return r.stdout.split() if r.returncode == 0 else []


def main():
    p = argparse.ArgumentParser(); p.add_argument("--arms", default=",".join(ARMS)); p.add_argument("--no-score", action="store_true")
    a = p.parse_args(); os.makedirs(TMP, exist_ok=True)
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    arms = [x for x in a.arms.split(",") if any("tripwire" in y for y in vol_ls(f"ckpts/qwen36_27b/{x}"))]
    if not arms:
        print("[tripwire] no arm has tripwire dumps yet", flush=True)
    if arms and not a.no_score:
        # ONE 1-GPU shell container scores every (arm, variant) dir under the frozen SFT critic only (score_dumps is incremental per step)
        loop = " ; ".join(f"[ -d {C}/{x}/tripwire/{v} ] && python -m nla.flow.score_dumps --dumps-dir {C}/{x}/tripwire/{v} --out {C}/{x}/tripwire_frozen_{v}.json "
                          f"--critic {C}/ar_sft_merged --sidecar /vol_q36/data/rl/rl_shuf.parquet" for x in arms for v in V)
        env = dict(os.environ, NLA_APP_NAME="nla-tripwire")
        r = subprocess.run(["modal", "run", "scripts/modal_nla_exp.py", "--task", "shell", "--gpus", "1", "--cmd", loop], cwd=REPO, env=env,
                           capture_output=True, text=True, timeout=7200)
        print(f"[tripwire] frozen scoring rc {r.returncode}; tail: {r.stdout[-400:]}", flush=True)
    for arm in arms:
        res = {}
        for v in V:
            remote = f"{C}/{arm}/tripwire_frozen_{v}.json"; loc = f"{TMP}/{arm}_{v}.json"
            subprocess.run(["modal", "volume", "get", "nla-exp", remote.replace("/vol/", ""), loc, "--force"], capture_output=True, timeout=300)
            if os.path.exists(loc): res[v] = json.load(open(loc))
        steps = sorted({int(s) for v in res for s in res[v] if s.isdigit()})
        rows = []
        for s in steps:
            g = lambda v, k: res.get(v, {}).get(str(s), {}).get(k)
            r = {"step": s}
            for v in V:
                r[f"live_{v}"] = g(v, "live_vector_fve"); r[f"frozen_{v}"] = g(v, "fve_frozen_sft_critic")
            for kind in ("live", "frozen"):
                for v in ("para", "shuf"):
                    o, x = r.get(f"{kind}_orig"), r.get(f"{kind}_{v}")
                    r[f"{kind}_gap_{v}"] = (o - x) if (o is not None and x is not None) else None
            rows.append(r)
        out[arm] = rows
        if rows:
            last = rows[-1]
            print(f"[tripwire] {arm} step {last['step']}: live gap para {last['live_gap_para']} shuf {last['live_gap_shuf']} | "
                  f"frozen gap para {last['frozen_gap_para']} shuf {last['frozen_gap_shuf']}", flush=True)
    json.dump(out, open(OUT, "w"), indent=1); print(f"[tripwire] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
