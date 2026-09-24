"""data/trunk_results.json: EVERY trunk-critic arm run tonight -- exact bits (from /vol/results/bits_trunk_*.json, fetched locally) + the
in-training held-out proxy trajectories (wandb octahedral-systems/nlt-qwen3-8b runs trunk_*) + the run configs.

  python scripts/nlt_trunk_results.py --bits ~/nlt-trunk-results/bits_trunk_*.json --out ~/shared/reports/natural-language-transcoder/data/trunk_results.json

Schema:
  {generated_utc, design (one paragraph), verdict (one paragraph),
   arms{run_name: {config{groups, batch, readout_rank, text_pool, lr_lora, null_reg, null_dm, contrast, curriculum_j_min, text_synth, prior, steps_planned},
                   status, rows_per_step, fm_samples_per_step, wandb_id,
                   proxy_evals[{step, rows_seen, fm_samples_seen, sets{label: {pmi_proxy_bits, dm_proxy_bits, content_proxy_bits, p_z_beats_dm, null_vs_prior_bits, workspace_content, workspace_p_beats_dm}}}],
                   exact_bits{tag: {step, n_per_set, ode_steps, ms_per_row_exact, sets{label: {n, n_tokens_mean, frac_z_beats_dm, frac_z_beats_rp,
                        bands{all|pre<=13|workspace14-32|motor>=33: {bits (PMI vs the trunk's OWN empty-prefix null), z_dm, z_rp, shuf_words, mask_next, content (paired z - z_dm), content_sem, p_z_beats_dm, null_vs_prior (exact, when computed), n}}}}}}}}
All bits are exact-ODE bits (Heun 32, paired probes) unless the key says proxy. content = paired PMI(z) - PMI(z_dm) is the comparable quantity across critics.
"""
import argparse, glob, json, os, time, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nlt_trunk_bits_json import band_entry

DESIGN = ("TRUNK critic (Bet B, DECISIONS v1.11): Qwen3-8B layers 0-23 with LoRA r64/a16 (rsLoRA) is the text-conditional denoiser on top of a frozen blind "
          "PairDenoiser prior (none_v1_pooled, pooled affine + delta target). Input [z tokens | source token(h_i) | K=4 activation tokens(x_t)+t]; fresh "
          "bidirectional attention over the activation tokens only, so the text prefix has an exact KV cache and each ODE step re-runs 5 tokens (~100 ms/row/solve "
          "at Heun 32, batch 128). Null path = the same trunk with an empty prefix. Arms varied: noise groups per text row (1 vs 8, block-diagonal mask), readout "
          "(full 20480->4096 vs low-rank 256), a pooled direct text read, LoRA lr, a motor-band curriculum, null-reg / null-dm terms, and a depth-tag-only diagnostic (T1).")
VERDICT = ("Negative for tonight: with <= ~130k distinct (pair, text) rows (<= 1M FM samples) no arm made the velocity depend on the text -- paired content z - z_dm "
           "within +-0.15 bits of 0 and P(z > z_dm) 0.48-0.60 on every set, including a categorical depth tag (T1: +0.9 vs +0.8 for the same-(i,j) control). "
           "The same networks learned the h_i pathway quickly: the empty-prefix null path beats the frozen prior by +33..+374 proxy bits by step 500-1000 with 8 noise "
           "groups (and -5 with 1 group), i.e. the trunk reads the source activation token but information in the text tokens does not reach the velocity readout at "
           "this sample budget. The trunk is ~100x costlier per row than the MLP cross-read adapter (56 rows/s on a B200), so the row budget that gives the adapters "
           "their 5-12 content bits (1.8-2M rows) was out of reach. The depth-tag diagnostic (T1b) settles that it is not only the budget: at 768k rows "
           "(1.5x the rows at which the 0.6B cross-read adapter recovered +38 bits from the same tag) the trunk still does not read the tag (+1.2 vs +0.8 control). "
           "Positive by-product: the trunk's empty-prefix null path is a much better BLIND density than the 1.89B pooled prior it sits on -- +885 +- 44 exact "
           "bits/pair at 128k rows (NLL 0.403 vs 0.742 bits/dim on the same rows; pre-workspace +1784, workspace +753, motor +442; gap 1-3 +1417, 11-25 +218), "
           "on all 4096 rows +880 +- 11. Checks (data/trunk_null_checks.json): train rows +859 (no memorisation, docs disjoint), N(0,I) targets -299 (not a probe/divergence "
           "artefact), SHUFFLED pairs +692 -> most of the gain is a generically better denoiser (the readout also sees x_t and t through the activation tokens); only "
           "~190 bits/pair need the matching h_i. So the 1.89B MLP prior is ~900 bits/pair short of the density a stronger denoiser reaches in 128k rows, i.e. prior capacity/steps.")


def wandb_runs():
    import wandb
    api = wandb.Api(timeout=90); out = {}
    for r in api.runs("octahedral-systems/nlt-qwen3-8b", filters={"display_name": {"$regex": "^trunk_"}}, order="-created_at"):
        if r.name in out: continue                          # newest run per name (relaunches share names)
        cfg = r.config; h = r.history(samples=5000, pandas=False)
        ev = []
        for x in h:
            keys = [k for k in x if k.startswith("eval_") and k.endswith("/pmi_proxy_bits")]
            if not keys: continue
            sets = {}
            for k in keys:
                lab = k.split("/")[0][5:]; g = lambda s: x.get(f"eval_{lab}/{s}")
                sets[lab] = {"pmi_proxy_bits": g("pmi_proxy_bits"), "dm_proxy_bits": g("dm_proxy_bits"), "content_proxy_bits": g("content_proxy_bits"), "p_z_beats_dm": g("p_z_beats_dm"),
                             "null_vs_prior_bits": g("null_vs_prior_bits"), "workspace_content": g("content_workspace"), "workspace_p_beats_dm": g("pbeat_workspace")}
            step = int(x.get("_step", 0)) + 1; B = int(cfg.get("batch", 0) or 0); G = int(cfg.get("groups", 1) or 1)
            ev.append({"step": step, "rows_seen": step * B, "fm_samples_seen": step * B * G, "sets": sets})
        out[r.name] = {"config": {k: cfg.get(k) for k in ("groups", "batch", "readout_rank", "text_pool", "lr", "lr_lora", "null_reg", "null_dm", "contrast", "curriculum_j_min", "curriculum_steps", "text_synth", "prior", "steps", "n_layers", "n_act_tokens", "max_len", "pool_weights")},
                       "status": r.state, "rows_per_step": int(cfg.get("batch", 0) or 0), "fm_samples_per_step": int(cfg.get("batch", 0) or 0) * int(cfg.get("groups", 1) or 1), "wandb_id": r.id, "proxy_evals": ev, "exact_bits": {}}
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("--bits", nargs="*", default=[]); p.add_argument("--out", required=True); p.add_argument("--no-wandb", action="store_true")
    a = p.parse_args()
    arms = {} if a.no_wandb else wandb_runs()
    for pat in a.bits:
        for f in sorted(glob.glob(os.path.expanduser(pat))):
            r = json.load(open(f)); tag = os.path.basename(f)[len("bits_"):-len(".json")]
            run = tag.split("_s")[0] if "_s" in tag and tag.rsplit("_s", 1)[1].isdigit() else tag
            arms.setdefault(run, {"config": r.get("config"), "status": "?", "proxy_evals": [], "exact_bits": {}})
            E = {"step": r.get("step"), "n_per_set": r.get("n_per_set"), "ode_steps": r.get("ode_steps"), "ckpt": r.get("ckpt"), "sets": {}}
            for name, res in r["critics"].items():
                label = name.split("@", 1)[1] if "@" in name else name; E["ms_per_row_exact"] = res.get("ms_per_row_exact")
                S = {"n": res["n_rows"], "n_tokens_mean": res.get("n_tokens_mean"), "frac_z_beats_dm": res.get("frac_z_beats_dm"), "frac_z_beats_rp": res.get("frac_z_beats_rp"), "frac_z_beats_shuf_words": res.get("frac_z_beats_shuf_words"),
                     "bands": {b: band_entry(res, b) for b in ("all", "pre<=13", "workspace14-32", "motor>=33")}}
                if "null_vs_prior_bits" in res:
                    for b in S["bands"]: S["bands"][b]["null_vs_prior"] = res["null_vs_prior_bits"]["mean"] if b == "all" else (res["null_vs_prior_bits"].get("by_band", {}).get(b) or {}).get("mean")
                    S["exact_pmi_vs_prior_bits"] = res["exact_pmi_vs_prior_bits"]["mean"]
                E["sets"][label] = S
            arms[run]["exact_bits"][tag] = E
    out = {"generated_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "design": DESIGN, "verdict": VERDICT, "arms": arms}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
    for run, A in sorted(arms.items()):
        last = A["proxy_evals"][-1] if A["proxy_evals"] else None
        cfg = A.get("config") or {}
        line = f"{run:>20} groups={cfg.get('groups')} rank={cfg.get('readout_rank')} pool={cfg.get('text_pool')} synth={cfg.get('text_synth')} status={A.get('status')}"
        if last:
            ws = [(l, v.get('content_proxy_bits'), v.get('p_z_beats_dm'), v.get('null_vs_prior_bits')) for l, v in last["sets"].items()]
            line += f" | last proxy eval step {last['step']} ({last['rows_seen']} rows): " + "; ".join(f"{l} content {c:+.2f} P {pz:.2f} null-vs-prior {nv:+.1f}" if c is not None and nv is not None else f"{l} content {c}" for l, c, pz, nv in ws)
        for tag, E in A["exact_bits"].items():
            line += f"\n{'':>20} exact {tag} (step {E.get('step')}, n={E.get('n_per_set')}): " + "; ".join(f"{l} content {S['bands']['all']['content']:+.2f}+-{S['bands']['all']['content_sem']:.2f} P {S['frac_z_beats_dm']:.2f}" for l, S in E["sets"].items())
        print(line)
    print("->", a.out)


if __name__ == "__main__":
    main()
