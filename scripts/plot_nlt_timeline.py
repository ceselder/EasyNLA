"""The night at a glance, from the team board (board/posts.jsonl): every post per agent over time, the crash window shaded, and the
cumulative post count. Every number plotted goes to data/night_timeline.json.

  python scripts/plot_nlt_timeline.py --report ~/shared/reports/natural-language-transcoder
"""
from __future__ import annotations
import argparse, json, os, textwrap
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e4de"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
START_H, END_H = 9.0, 29.5                     # 09:00 UTC start, 05:30 UTC next day = hour 29.5
CRASH = (10.15, 19.483)                         # 10:09 -> 19:29 UTC
AGENTS = ["orchestrator", "infra", "lens", "redteam", "proposer", "designer-oracle", "rl", "designer-bootstrap", "trunk", "reporter"]
LABEL = {"orchestrator": "orchestrator", "infra": "infra (data, critic, exact bits)", "lens": "lens (lenses, lens texts, prior doctor)", "redteam": "redteam (evals, gates)", "proposer": "proposer (teacher / oracle pools)",
         "designer-oracle": "designer-oracle (round 1)", "rl": "rl (verbalizer, RL)", "designer-bootstrap": "designer-bootstrap (round 1)", "trunk": "trunk (Bet B critic)", "reporter": "reporter (this report)"}
KIND_COL = {"result": CAT[0], "decision": CAT[1], "critique": CAT[7], "other": "#b3b1a8"}
MILESTONES = [(9.35, "DECISIONS v1"), (9.8, "data ready"), (10.15, "box OOM"), (19.67, "restart"), (20.68, "D3 gate FAIL"), (20.78, "pooled fix (v1.7)"), (20.9, "readers 73%"), (21.67, "critic v3 (v1.12)")]


def hours(ts):
    h, m, s = ts.split()[0].split(":"); return int(h) + int(m) / 60 + int(s) / 3600


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--report", default=os.path.expanduser("~/shared/reports/natural-language-transcoder")); ap.add_argument("--stem", default="night_timeline")
    a = ap.parse_args(); posts = [json.loads(l) for l in open(os.path.join(a.report, "board", "posts.jsonl"))]
    t = []; last = 0.0
    for p in posts:                                # append-only log; a drop of > 12 h in the clock means we passed midnight
        h = hours(p["ts"]); h = h + 24 if h + 12 < last else h; last = max(last, h); t.append(h)
    agents = [ag for ag in AGENTS if any(p["from"] == ag for p in posts)] + sorted({p["from"] for p in posts} - set(AGENTS))
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "legend.fontsize": 10.5, "xtick.labelsize": 11, "ytick.labelsize": 10.5,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 9.5), dpi=150, gridspec_kw={"height_ratios": [2.4, 1], "hspace": 0.38}, sharex=True)
    y_of = {ag: i for i, ag in enumerate(agents[::-1])}
    for kind, col in KIND_COL.items():
        sel = [(t[i], y_of[p["from"]]) for i, p in enumerate(posts) if (p["kind"] if p["kind"] in KIND_COL else "other") == kind]
        if sel: ax1.scatter([s[0] for s in sel], [s[1] + np.random.default_rng(len(sel)).uniform(-0.18, 0.18, len(sel))[k] for k, s in enumerate(sel)], s=22, color=col, alpha=0.85, label=f"{kind} ({len(sel)})", zorder=3)
    for ax in (ax1, ax2):
        ax.axvspan(*CRASH, color="#f7e1da", alpha=0.9, lw=0, zorder=0); ax.grid(axis="y", visible=False)
    ax1.text(np.mean(CRASH), len(agents) - 0.55, "box out of memory:\nnothing ran for 9 h 20 min\n(10:09 → 19:29 UTC)", ha="center", va="top", fontsize=11.5, color="#9A3B22")
    ax1.set_yticks(range(len(agents))); ax1.set_yticklabels([LABEL.get(ag, ag) for ag in agents[::-1]]); ax1.set_ylim(-0.6, len(agents) - 0.4)
    ax1.legend(frameon=False, loc="upper right", title="board post kind", title_fontsize=10)
    ax1.set_title("Every message on the team board, by agent (one dot per post)", loc="left", fontsize=12.5)
    cum = np.arange(1, len(t) + 1); ax2.step(t, cum, where="post", color=CAT[0], lw=2); ax2.set_ylabel("posts so far"); ax2.set_xlabel("UTC time (2026-09-23 → 09-24)")
    shown = [m for m in MILESTONES if m[0] < t[-1] + 0.1]
    for k, (h, lab) in enumerate(shown):
        ax2.axvline(h, color=INK2, lw=0.7, ls=(0, (2, 2)))
        ax2.text(h, cum[-1] * (1.12 if k % 2 == 0 else 1.30), str(k + 1), fontsize=9.5, color=SURFACE, ha="center", va="center", bbox={"boxstyle": "circle,pad=0.25", "fc": INK2, "ec": "none"})
    key = "\n".join(f"{k + 1}  {lab}  ({int(h) % 24:02d}:{int(round((h % 1) * 60)):02d})" for k, (h, lab) in enumerate(shown))
    ax2.text(23.3, cum[-1] * 1.45, key, fontsize=9.5, color=INK2, ha="left", va="top", linespacing=1.35)
    ax2.set_ylim(0, cum[-1] * 1.5); ax2.set_title("Cumulative posts, with the milestones that shaped the night", loc="left", fontsize=12.5)
    ax2.axvline(END_H, color=INK, lw=1); ax2.text(END_H, cum[-1] * 0.1, " hard end 05:30", fontsize=9.5, color=INK, ha="left")
    ticks = np.arange(START_H, END_H + 0.1, 2); ax2.set_xticks(ticks); ax2.set_xticklabels([f"{int(h) % 24:02d}:00" for h in ticks]); ax2.set_xlim(START_H - 0.3, END_H + 0.6)
    n_res = sum(1 for p in posts if p["kind"] == "result"); n_dec = sum(1 for p in posts if p["kind"] == "decision")
    fig.suptitle("\n".join(textwrap.wrap(f"The night at a glance: {len(posts)} board posts by {len(agents)} agents ({n_res} results, {n_dec} decisions) in {t[-1] - START_H - (CRASH[1] - CRASH[0]):.1f} working hours, "
                                          f"and a 9 h 20 min hole where the shared box ran out of memory and killed every agent", 112)), fontsize=13.5, x=0.01, y=0.995, ha="left", va="top")
    fig.text(0.01, 0.005, "Source: board/posts.jsonl (append-only team message board). Post kinds other than result / decision / critique (status, proposal, question, answer, idea) are grey.", fontsize=9.5, color=INK2, ha="left", va="bottom")
    fig.subplots_adjust(left=0.27, right=0.985, top=0.88, bottom=0.08)
    for ext in ("png", "pdf"): fig.savefig(os.path.join(a.report, f"{a.stem}.{ext}"), facecolor=SURFACE, bbox_inches="tight")
    per_agent = {ag: {"posts": sum(1 for p in posts if p["from"] == ag), "first_utc": next(p["ts"] for p in posts if p["from"] == ag), "last_utc": [p["ts"] for p in posts if p["from"] == ag][-1],
                      "kinds": dict(Counter(p["kind"] for p in posts if p["from"] == ag))} for ag in agents}
    per_hour = dict(sorted(Counter(int(h) % 24 for h in t).items()))
    json.dump({"n_posts": len(posts), "n_agents": len(agents), "per_agent": per_agent, "per_kind": dict(Counter(p["kind"] for p in posts)), "posts_per_utc_hour": per_hour,
               "crash_window_utc": ["10:09", "19:29"], "crash_hours": round(CRASH[1] - CRASH[0], 2), "start_utc": "09:00", "hard_end_utc": "05:30 (+1 day)", "last_post_utc": posts[-1]["ts"], "milestones": [{"hour": h, "label": l} for h, l in MILESTONES]},
              open(os.path.join(a.report, "data", f"{a.stem}.json"), "w"), indent=1)
    print("saved", os.path.join(a.report, f"{a.stem}.png"), len(posts), "posts")


if __name__ == "__main__":
    main()
