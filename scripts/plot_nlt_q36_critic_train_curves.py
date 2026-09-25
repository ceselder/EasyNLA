"""Training-time spot evals of the direction critics, aligned on TEXT steps (nlt-27b-olens).

Parses the Modal app logs of the listed critic runs ([eval@STEP ...] lines: craft_full proxy content / P(z>z_dm) / exact PMI / exact content,
describer P / content), aligns them on text steps (= step - uncond_steps) and writes
  data/critic_train_curves.json  +  fig_critic_uncond_ablation.{png,pdf}
Claim tested: does the 1500-step unconditional pretraining phase hurt the text path (v1 / v2 / v3: 1500 uncond steps; v1b / v3b: none)?
Spot evals = Heun 16, n 64 (the trainer's own monitor), NOT the Heun-64 bits tables.
"""
import argparse, json, os, re, subprocess
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REP = "/home/celeste/shared/reports/nlt-27b-olens"
RUNS = {  # tag: (app id, uncond steps, label)
    "v1": ("ap-xLqiQeU1iXNyuPWTaMI0tL", 1500, "critic v1: 1500 unconditional steps, then text"),
    "v2": ("ap-v6k1AKzbDY0s5VzKlxJXGH", 1500, "critic v2: 1500 unconditional steps, full text mix"),
    "v1b": ("ap-qQSm6zb1BbwIbCAiEdvnJ2", 0, "critic v1b: NO unconditional phase (uncond rows 10%)"),
    "v3b": (None, 0, "critic v3b: no unconditional phase + anchored contrast"),
}
COL = {"v1": "#2b6cb0", "v2": "#6b46c1", "v1b": "#c05621", "v3b": "#1a9c6e"}


def app_id(tag):
    if RUNS[tag][0]: return RUNS[tag][0]
    try:
        for l in open("/home/celeste/nlt-q36-logs/apps.txt"):
            m = re.match(rf"\[critic_{tag}\] https://modal.com/apps/\S+/(ap-[A-Za-z0-9]+)", l)
            if m: aid = m.group(1)
        return aid
    except Exception: return None


def parse(aid):
    env = dict(os.environ); env.pop("MODAL_TOKEN_ID", None); env.pop("MODAL_TOKEN_SECRET", None)
    try: out = subprocess.run(["timeout", "120", "modal", "app", "logs", aid], capture_output=True, text=True, env=env, cwd="/home/celeste/nlt").stdout
    except Exception: return []
    rows = []
    for l in out.splitlines():
        m = re.match(r"\[eval@(\d+) rows (\d+)\] (.*)", l)
        if not m: continue
        f = dict(re.findall(r"([a-z_A-Z0-9]+/[a-z_]+)=([-0-9.]+)", m.group(3)))
        g = lambda k: float(f[k]) if k in f else None
        rows.append({"step": int(m.group(1)), "craft_proxy_content": g("craft_full/proxy_content_bits"), "craft_p": g("craft_full/proxy_p_z_gt_dm"), "craft_pmi": g("craft_full/exact_pmi_bits"),
                     "craft_content": g("craft_full/exact_content_bits"), "craft_cos_c": g("craft_full/cos_mean_c"), "craft_cos_u": g("craft_full/cos_mean_u"),
                     "desc_p": g("describer/proxy_p_z_gt_dm") if "describer/proxy_p_z_gt_dm" in f else g("describer_A/proxy_p_z_gt_dm"),
                     "desc_pmi": g("describer/exact_pmi_bits") if "describer/exact_pmi_bits" in f else g("describer_A/exact_pmi_bits"),
                     "desc_content": g("describer/exact_content_bits") if "describer/exact_content_bits" in f else g("describer_A/exact_content_bits")})
    return sorted({r["step"]: r for r in rows}.values(), key=lambda r: r["step"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", default="v1,v2,v1b,v3b"); a = ap.parse_args()
    data = {}
    for t in a.tags.split(","):
        aid = app_id(t)
        if not aid: continue
        rows = parse(aid)
        for r in rows: r["text_step"] = r["step"] - RUNS[t][1]
        data[t] = {"app": aid, "uncond_steps": RUNS[t][1], "label": RUNS[t][2], "evals": [r for r in rows if r["text_step"] >= 0], "note": "trainer spot evals: Heun 16, n 64 held-out pairs"}
    os.makedirs(f"{REP}/data", exist_ok=True); json.dump(data, open(f"{REP}/data/critic_train_curves.json", "w"), indent=1)
    have = [t for t in data if data[t]["evals"]]
    if not have: print("no evals"); return
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    def series(t, k): return [r["text_step"] for r in data[t]["evals"] if r.get(k) is not None], [r[k] for r in data[t]["evals"] if r.get(k) is not None]
    for ax, k, title, yl in ((axes[0, 0], "craft_content", "Paired content of the crafted text: ~3x without the unconditional phase", "exact content bits (Heun 16, n 64)"),
                             (axes[0, 1], "craft_p", "Own text vs depth-matched wrong text", "P(z > z_dm)"),
                             (axes[1, 0], "craft_pmi", "Presence penalty: exact PMI(z) vs no text", "exact PMI bits"),
                             (axes[1, 1], "desc_content", "Sonnet trace, same critics", "exact content bits")):
        for t in have:
            x, y = series(t, k)
            if x: ax.plot(x, y, "o-", color=COL[t], lw=2, label=data[t]["label"])
        ax.axhline(0, color="k", lw=0.8) if k in ("craft_pmi",) else None
        if k == "craft_p": ax.axhline(0.5, color="k", lw=0.8); ax.set_ylim(0.4, 1.0)
        ax.set_xlabel("text-training steps (uncond phase removed)"); ax.set_ylabel(yl); ax.set_title(title, fontsize=12); ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Unconditional pretraining hurt the judge: it absorbed what the text path should learn (trainer spot evals, Heun 16, n 64)", fontsize=13, y=1.0); fig.tight_layout()
    fig.savefig(f"{REP}/fig_critic_uncond_ablation.png", dpi=150, bbox_inches="tight"); fig.savefig(f"{REP}/fig_critic_uncond_ablation.pdf", bbox_inches="tight"); print("saved fig_critic_uncond_ablation")
    for t in have:
        for r in data[t]["evals"]: print(t, "text step", r["text_step"], "content", r["craft_content"], "P", r["craft_p"], "PMI", r["craft_pmi"])


if __name__ == "__main__":
    main()
