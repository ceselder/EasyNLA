"""Where the bits are: the blind prior's denoiser gap vs the told-depth oracle vs text content, on one (log) scale.

Every number is read from the report's data/*.json (trunk_null_checks, board_numbers, acceptance_*, verdicts_*) and
written back to data/bits_scale.json; the two Heun-32 told-depth numbers come from infra's re-denomination (board #546).
"""
import argparse, json, os, textwrap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e4de"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--report", default=os.path.expanduser("~/shared/reports/natural-language-transcoder")); ap.add_argument("--stem", default="bits_scale")
    a = ap.parse_args(); D = os.path.join(a.report, "data")
    J = lambda n: json.load(open(os.path.join(D, n)))
    tnc = J("trunk_null_checks.json"); bn = J("board_numbers.json"); e2 = J("acceptance_enc_e2_plain.json")["sources"]; big = J("acceptance_union_pooled_big.json")["sources"]
    ct = {r["source"]: r for r in J("verdicts_union_pooled_null_controls_table.json")["rows"]}; ref = J("headline_reader_vs_critic.json")
    gain = tnc["val"]["gain_bits_trunk_null_minus_prior"]["mean"]; shuf = tnc["checks"]["shuffled_pairs"]["mean"]
    g = bn["d3_gate"]
    rows = [  # (label, bits, group, provenance)
        ("a Qwen3-8B-sized denoiser vs the 1.89B blind prior\n(the trunk's empty-prefix path, all 4096 fixed pairs)", gain, "prior", "data/trunk_null_checks.json val.mean"),
        ("… of which needs the matching h_i\n(gain minus the shuffled-pair gain)", gain - shuf, "prior", "trunk_null_checks: val.mean − checks.shuffled_pairs.mean"),
        ("told-depth oracle over the blind prior, Heun 32\n(over-reads the pooled space ~2×; board #546)", 57.5, "depth", "board #546: +57.5 ± 1.7 on rows 0–1023"),
        ("told-depth oracle over the blind prior, exact Heun 64\n(the reported D3(b) value)", g["told_depth_gain_pooled_full_bits"], "depth", "board_numbers.d3_gate.told_depth_gain_pooled_full_bits"),
        ("ideal mixture bound for depth (D3(b) target)", 6.6, "depth", "DECISIONS v1.9"),
        ("raw J-lens top-20 lists as TEXT (T2), paired content", ref["reference_raw_jlens_list_as_text"]["content_bits"], "text", "data/headline_reader_vs_critic.json reference_raw_jlens_list_as_text.content_bits"),
        ("accepted 8B-encoder critic: J-lens sentence, content\n(held-out manifests, n = 1001)", next(v for k, v in e2.items() if k.endswith("lensdiff_jlens_L1"))["content_all"], "text", "acceptance_enc_e2_plain L1 content_all"),
        ("wider adapter: J-lens sentence, content", big["lensdiff_jlens_L1"]["content_all"], "text", "acceptance_union_pooled_big L1 content_all"),
        ("headline critic: the VERBALIZER's sentence, content", ct["v0_ao_tsv1"]["content"], "text", "verdicts_union_pooled_null_controls_table v0_ao_tsv1 content"),
    ]
    col = {"prior": CAT[6], "depth": CAT[3], "text": CAT[0]}
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID})
    fig, ax = plt.subplots(figsize=(12.5, 7.6), dpi=150)
    y = np.arange(len(rows))[::-1]; vals = [max(r[1], 0.3) for r in rows]
    ax.barh(y, vals, color=[col[r[2]] for r in rows], height=0.66)
    for yy, r in zip(y, rows): ax.text(max(r[1], 0.3) * 1.08, yy, f"{r[1]:,.1f} bits", va="center", fontsize=11.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=10.5); ax.set_xscale("log"); ax.set_xlim(0.5, 4000)
    ax.set_xlabel("exact bits per (h_i, h_j) pair, log scale"); ax.grid(axis="x", color=GRID); ax.grid(axis="y", visible=False)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=col["prior"], label="prior quality (denoiser gap)"), Patch(color=col["depth"], label="depth: what a told-depth oracle adds"), Patch(color=col["text"], label="text: paired content = bits(z) − bits(z_dm)")], frameon=False, loc="lower right", fontsize=10.5)
    fig.suptitle("\n".join(textwrap.wrap("Where the bits are: the blind prior's denoiser gap (~880 bits per pair) dwarfs what a told-depth oracle adds (30–58) and what any text adds (1–14) — every PMI in this report sits on that denominator (Qwen3-8B, layers 9–34, exact ODE log-likelihood, fixed held-out pairs)", 100)), fontsize=13.5, x=0.01, y=0.995, ha="left", va="top")
    fig.text(0.01, 0.005, "Denoiser gap: trunk critic's empty-prefix path (Qwen3-8B + LoRA + 300M readout on (x_t, t, h_i)) vs the frozen 1.89B pooled prior, 4096 pairs, Heun 32, paired probes; the shuffled-pair check (h_i of row k, h_j of row k+1) keeps 692 of the 880 bits. "
             "Told-depth: same rows, Heun 32 (board #546) and Heun 64 (bits_pooled_mix.json). Text: redteam's held-out control manifests (~1000 pairs) except the raw-list reference (infra's fixed set, n = 512). PRELIMINARY: no blind prior has passed the D3 gate.", fontsize=9, color=INK2, ha="left", va="bottom", wrap=True)
    fig.subplots_adjust(left=0.36, right=0.97, top=0.86, bottom=0.14)
    for ext in ("png", "pdf"): fig.savefig(os.path.join(a.report, f"{a.stem}.{ext}"), facecolor=SURFACE, bbox_inches="tight")
    json.dump({"rows": [{"label": r[0].replace("\n", " "), "bits": r[1], "group": r[2], "provenance": r[3]} for r in rows], "denoiser_gap_all4096": gain, "shuffled_pair_gain": shuf}, open(os.path.join(D, f"{a.stem}.json"), "w"), indent=1)
    print("saved", os.path.join(a.report, f"{a.stem}.png"))


if __name__ == "__main__":
    main()
