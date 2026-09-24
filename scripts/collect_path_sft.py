"""Parse the PATH SFT logs (+ V0b's own SFT log) into data/path_sft.json: per arm the val-loss curve (step, val), the final val loss and
the per-source held-out losses (from the 'done -> ... final eval {...}' line or an EVAL-ONLY line). Then plot curves + bars (PNG + PDF).

  python scripts/collect_path_sft.py --out ~/shared/reports/natural-language-transcoder/data/path_sft.json \
      --arm v0b=logs/rl/v0b_chain.log --arm v0b_path_d=logs/path/sft_v0b_path_d.log --arm v0b_path=logs/path/sft_v0b_path.log ...
"""
from __future__ import annotations
import argparse, json, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {"v0b": "V0b: h_i, h_j (2 markers)", "v0b_evalonly": "V0b re-evaluated (same code path)", "v0b_path_d": "path-delta: h_i, d_k = h_k - h_{k-1}, h_j",
          "v0b_path": "path: h_i, a_k, m_k (attn + MLP writes), h_j", "v0b_path_c": "count control: h_i, (j-i) empty markers, h_j",
          "v0b_path_f": "path, fixed 50 markers (count carries no gap)"}
SERIES = ["#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
RX_STEP = re.compile(r"^(?:path-)?sft\s+(\d+)/(\d+) \| loss/tok ([\d.]+) \| val ([\d.]+)")
RX_FINAL = re.compile(r"final eval (\{.*\})")
RX_EVALONLY = re.compile(r"EVAL-ONLY (\{.*\})")


def parse(path):
    curve, final, n_steps = [], None, None
    for line in open(path, errors="replace"):
        m = RX_STEP.match(line.strip())
        if m: curve.append((int(m.group(1)), float(m.group(4)))); n_steps = int(m.group(2))
        m = RX_FINAL.search(line) or RX_EVALONLY.search(line)
        if m:
            try: final = json.loads(m.group(1))
            except Exception: pass
    out = {"curve": curve, "n_steps": n_steps}
    if final:
        out["val_loss"] = final.get("val/loss_per_token")
        for k, v in final.items():
            if k.startswith("val/loss_per_token_"): out["val_loss_" + k[len("val/loss_per_token_"):]] = v
    elif curve: out["val_loss"] = curve[-1][1]
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--arm", action="append", default=[], help="tag=logpath"); p.add_argument("--plot", default=None, help="output stem for the figure")
    a = p.parse_args(); arms = []
    for spec in a.arm:
        tag, path = spec.split("=", 1)
        if not os.path.exists(path): print(f"[collect] missing {path}"); continue
        d = parse(path); d["tag"] = tag; d["label"] = LABELS.get(tag, tag); d["log"] = path; arms.append(d)
        print(f"[collect] {tag}: {len(d['curve'])} evals, final val {d.get('val_loss')}, per-source {[ (k, round(v, 4)) for k, v in d.items() if k.startswith('val_loss_')]}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"arms": arms, "note": "held-out CE in nats per response token on the first 768 rows of /vol/z/v0b_mix/val/rows.parquet (same rows, same filters for every arm); "
                                     "all arms = V0 LoRA init + 1 epoch on V0b's 20,126 rows, lr 3e-5, batch 32, warmup 20, cosine, r64 a16 rsLoRA; only the input differs"}, open(a.out, "w"), indent=1)
    print("->", a.out)
    if a.plot:
        plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
        ax = axes[0]
        for k, d in enumerate(arms):
            if len(d["curve"]) < 2: continue
            xs, ys = zip(*d["curve"]); ax.plot(xs, ys, color=SERIES[k % len(SERIES)], lw=2, label=d["label"], marker="o", ms=3.5)
        ax.set_xlabel("SFT step (batch 32)"); ax.set_ylabel("held-out CE, nats / token"); ax.set_ylim(1.9, 3.0); ax.grid(color="#e6e5e1", lw=0.8); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=9.5, loc="upper right"); ax.set_title("Held-out loss during the 1-epoch SFT", loc="left", fontsize=12, color=INK2)
        ax = axes[1]; src = [("val_loss_lenslist-v0b", "J-lens list-sentences"), ("val_loss_teacher-sonnet-v1", "teacher prose"), ("val_loss", "all rows")]
        fin = [d for d in arms if d.get("val_loss") is not None and d["tag"] != "v0b_evalonly" or (d["tag"] == "v0b_evalonly")]
        w = 0.8 / max(1, len(fin))
        for k, d in enumerate(fin):
            vals = [d.get(kk, float("nan")) for kk, _ in src]
            xs = [q + (k - (len(fin) - 1) / 2) * w for q in range(len(src))]
            ax.bar(xs, vals, width=w * 0.92, color=SERIES[arms.index(d) % len(SERIES)], edgecolor=SURFACE, linewidth=1.5, label=d["label"])
            for x_, v in zip(xs, vals):
                if v == v: ax.text(x_, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK, rotation=90)
        ax.set_xticks(range(len(src))); ax.set_xticklabels([lab for _, lab in src]); ax.set_ylabel("final held-out CE, nats / token")
        allv = [d.get(kk) for d in fin for kk, _ in src if d.get(kk) is not None]
        if allv: ax.set_ylim(max(0, min(allv) - 0.4), max(allv) + 0.45)
        ax.grid(axis="y", color="#e6e5e1", lw=0.8); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False); ax.set_title("Final loss by SFT source", loc="left", fontsize=12, color=INK2)
        fig.suptitle("Do the intermediate attention / MLP writes help the verbalizer? Held-out SFT loss, same rows, same init, same hparams", fontsize=13, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(a.plot + ".png", dpi=150); fig.savefig(a.plot + ".pdf"); print("->", a.plot + ".png")


if __name__ == "__main__":
    main()
