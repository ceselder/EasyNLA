"""Collect the TRUNK critic's bits jsons (/vol/results/bits_trunk_*.json, fetched to a local dir) into ONE compact report json.

  python scripts/nlt_trunk_bits_json.py --bits ~/nlt-trunk-results/bits_trunk_*.json --out ~/shared/reports/natural-language-transcoder/data/trunk_bits.json

Schema: {generated_utc, arms{arm: {ckpt, step, space, config, ode_steps, n_per_set, ms_per_row_exact,
          sets{label: {n, n_tokens_mean, bits_per_token, frac_z_beats_dm, frac_z_beats_rp, frac_z_beats_shuf_words, text_presence_offset_flag,
                       form_bits (dm - shuf_words), depth_generic_bits (dm - rp),
                       bands{all | pre<=13 | workspace14-32 | motor>=33: {bits, sem, n, z_dm, z_rp, shuf_words, mask_next, content, content_sem, p_z_beats_dm, vs_mix}}}}}}}
All numbers are exact-ODE bits (paired probes); content = paired z - z_dm.
"""
import argparse, glob, json, os, time


def band_entry(res, band):
    def g(key, field="mean"):
        d = res.get(key)
        if not d: return None
        if band == "all": return d.get(field)
        return (d.get("by_band", {}).get(band) or {}).get(field)
    e = {"bits": g("exact_pmi_bits"), "sem": g("exact_pmi_bits", "sem"), "n": g("exact_pmi_bits", "n"), "z_dm": g("shuffle_exact_pmi_bits"), "z_rp": g("rp_exact_pmi_bits"),
         "shuf_words": g("shuf_words_exact_pmi_bits"), "mask_next": g("mask_next_exact_pmi_bits"), "content": g("content_exact_bits"), "content_sem": g("content_exact_bits", "sem"),
         "vs_mix": g("exact_pmi_vs_mix_bits"), "blind_vs_mix": g("blind_vs_mix_bits")}
    e["p_z_beats_dm"] = res.get("frac_z_beats_dm") if band == "all" else (res.get("p_z_beats_dm_by_band") or {}).get(band)
    return e


def main():
    p = argparse.ArgumentParser(); p.add_argument("--bits", nargs="+", required=True); p.add_argument("--out", required=True)
    a = p.parse_args()
    out = {"generated_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "arms": {}}
    for pat in a.bits:
        for f in sorted(glob.glob(os.path.expanduser(pat))):
            r = json.load(open(f)); arm = os.path.basename(f)[len("bits_"):-len(".json")]
            A = {"ckpt": r.get("ckpt"), "step": r.get("step"), "space": r.get("space"), "config": r.get("config"), "ode_steps": r.get("ode_steps"), "n_per_set": r.get("n_per_set"), "sets": {}}
            for name, res in r["critics"].items():
                label = name.split("@", 1)[1] if "@" in name else name
                A["ms_per_row_exact"] = res.get("ms_per_row_exact")
                A["sets"][label] = {"n": res["n_rows"], "n_tokens_mean": res.get("n_tokens_mean"), "bits_per_token": res.get("exact_bits_per_token"), "frac_z_beats_dm": res.get("frac_z_beats_dm"), "frac_z_beats_rp": res.get("frac_z_beats_rp"),
                                    "frac_z_beats_shuf_words": res.get("frac_z_beats_shuf_words"), "text_presence_offset_flag": res.get("text_presence_offset_flag"), "form_bits": res.get("form_bits_dm_minus_shufwords"),
                                    "depth_generic_bits": res.get("depth_generic_bits_dm_minus_rp"), "mask_next_drop_by_band": res.get("mask_next_drop_bits_by_band"),
                                    "bands": {b: band_entry(res, b) for b in ("all", "pre<=13", "workspace14-32", "motor>=33")}}
            out["arms"][arm] = A
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
    for arm, A in out["arms"].items():
        for label, S in A["sets"].items():
            ws = S["bands"]["workspace14-32"]; al = S["bands"]["all"]
            print(f"{arm:>18} {label:>12}: all bits {al['bits']:+7.2f} dm {al['z_dm']:+7.2f} rp {al['z_rp']:+7.2f} content {al['content']:+6.2f}+-{al['content_sem']:.2f} P {S['frac_z_beats_dm']:.2f} | workspace content {ws['content'] if ws['content'] is None else round(ws['content'], 2)} P {ws['p_z_beats_dm']}")
    print("->", a.out)


if __name__ == "__main__":
    main()
