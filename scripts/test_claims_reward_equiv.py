"""GPU equivalence test: the RL reward (FlowCritic.score_claims_composed, --claim-reward singles_red) must equal the composition eval's singles_red
(scripts/claims_compose_variants.py: sum of single-claim PMIs - text-LM redundancy) on identical claims, noise and critic.

Rows of the 120-row stage-0 benchmark; claims = the row's true claims, given to the RL path as a bullet list (so the claims parser is in the loop);
noise = the eval's per-row draws (row_setup: D eps x the 5-point t grid), handed to the RL path through eps_fn; both paths load the same adapter.
Eval path on cuda:0 (FlowBundle + claims_compose_variants.claim_deltas / single_pmis + ClaimLM), RL path on cuda:1 (FlowCritic), one shared ClaimLM.
Pass = the parsed claims equal the eval's claims on every row and |reward_RL - singles_red_eval| <= atol + rtol * |singles_red_eval|.
--mode fp32 (default, the pass/fail test): ONE FlowCritic's weights serve both paths (the eval functions get a FlowBundle-shaped shim), the denoiser
runs in fp32 without autocast, eval deltas are kept in fp32 and every claim is encoded alone in both paths, so only the code paths differ.
--mode bf16: production numerics (two model copies on two GPUs, bf16 autocast, fp16 eval deltas): reports the numerical noise level, not a test.
  python scripts/test_claims_reward_equiv.py --adapter /vol_glp/cond/c1_synth_p2/adapter_latest.pt --rows 50"""
import argparse, json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from claims_compose_variants import TS, OUT, row_setup, claim_deltas, single_pmis


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--adapter", required=True); ap.add_argument("--rows", type=int, default=50); ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--atol", type=float, default=None); ap.add_argument("--rtol", type=float, default=None); ap.add_argument("--lm", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--mode", choices=["fp32", "bf16"], default="fp32")
    a = ap.parse_args(); d0, d1 = "cuda:0", ("cuda:0" if ap.parse_args().mode == "fp32" else "cuda:1")
    if a.atol is None: a.atol = 0.05 if a.mode == "fp32" else 0.5
    if a.rtol is None: a.rtol = 1e-4 if a.mode == "fp32" else 0.005
    import claims_compose_variants as ccv
    import pyarrow.parquet as pq
    from nla.flow.scoring import FlowBundle
    from nla.flow.rl_critic import FlowCritic
    from nla.flow.claims import format_claims, split_claims
    from nla.flow.claim_lm import ClaimLM
    C = json.load(open(f"{OUT}/claims.json"))["items"][: a.rows]
    t = pq.read_table("/vol_q36/data/sft/av_sft_val_clean1.parquet", columns=["activation_vector"]); N = t.num_rows
    acts = torch.tensor(np.asarray(t.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(N, -1))
    aa = torch.load(a.adapter, map_location="cpu")["args"]; pov = os.path.join(os.path.dirname(a.adapter), "prior_cotrained_latest.pt"); pov = pov if os.path.exists(pov) else None
    fb = None
    if a.mode == "bf16":
        fb = FlowBundle(aa["prior"], a.adapter, aa["stats"], d0, base="Qwen/Qwen3.6-27B", enc_layer=aa.get("enc_layer", 42), ar_ckpt=aa.get("ar_ckpt", "/vol/ckpts/qwen36_27b/ar_sft_merged"), prior_override=pov)
        fb.model.eval()
    fc = FlowCritic(aa["prior"], a.adapter, aa["stats"], None, None, torch.device(d1), enc_layer=aa.get("enc_layer", 42), t_grid=tuple(TS), eps_per_t=a.D, train_adapter=False,
                    prior_override=pov, ar_ckpt=aa.get("ar_ckpt", "/vol/ckpts/qwen36_27b/ar_sft_merged"), enc_device=torch.device(d1))
    if a.mode == "fp32":   # same weights for both paths, fp32 denoiser, per-claim encoding, fp32 eval deltas
        fc.model.float(); fc.amp = False; ccv.PREC.update(amp=False, delta=torch.float32)
        _tok = fc._tok_states; fc._tok_states = lambda texts, chunk=1: (lambda em: (em[0].float(), em[1]))(_tok(texts, chunk=1))
        class _Shim:
            norm, model = fc.norm, fc.model
            @staticmethod
            def cond(texts): e, m = fc._tok_states(texts); return e, m, None
        fb = _Shim()
    lm = ClaimLM(a.lm, d0); bullet = aa.get("claim_subsets", 0) > 0
    res = []
    for it in C:
        row = it["row"]; tc = [c["claim"] for c in it["true_claims"]]
        x0, E, xt, tv, tgt, v0, Lu = row_setup(fb, acts[row], row, a.D, d0)
        single = single_pmis(claim_deltas(fb, tc, xt, tv, v0, bullet, 8), v0, tgt, Lu)
        ev = sum(single) - lm.redundancy(tc)                                                       # the eval's singles_red (no claim cost)
        out = fc.score_claims_composed([format_claims(tc)], [acts[row]], [row], cost=0.0, claim_max=64, reward="singles_red", lm=lm, eps_fn=lambda g, k: E[k])
        parsed = out["claims"][0]; rl = out["reward"][0]
        res.append({"row": row, "n": len(tc), "parse_ok": parsed == [c.rstrip(";").strip() for c in tc], "eval": ev, "rl": rl, "diff": None if rl is None else rl - ev,
                    "singles_eval": single, "singles_rl": out["singles"].get(0)})
        print(f"[equiv] row {row}: {len(tc)} claims, parse {'ok' if res[-1]['parse_ok'] else 'MISMATCH'}, eval {ev:.3f} rl {rl:.3f} diff {rl - ev:+.4f}", flush=True)
    ok = [r["parse_ok"] and r["rl"] is not None and abs(r["diff"]) <= a.atol + a.rtol * abs(r["eval"]) for r in res]
    dd = np.array([abs(r["diff"]) for r in res if r["diff"] is not None]); rel = np.array([abs(r["diff"]) / max(abs(r["eval"]), 1e-9) for r in res if r["diff"] is not None])
    summ = {"adapter": a.adapter, "mode": a.mode, "rows": len(res), "pass": bool(all(ok)), "n_pass": int(sum(ok)), "max_abs_diff": float(dd.max()), "mean_abs_diff": float(dd.mean()),
            "max_rel_diff": float(rel.max()), "parse_all_ok": bool(all(r["parse_ok"] for r in res)), "atol": a.atol, "rtol": a.rtol, "D": a.D, "t_grid": TS}
    json.dump({"summary": summ, "rows": res}, open(f"{OUT}/test_claims_reward_equiv_{a.mode}.json", "w"), indent=1)
    print(f"[equiv] {'PASS' if summ['pass'] else 'FAIL'}: {summ['n_pass']}/{len(res)} rows within {a.atol} + {a.rtol}*|eval|; max |diff| {summ['max_abs_diff']:.4f} nats, "
          f"mean {summ['mean_abs_diff']:.4f}, max rel {summ['max_rel_diff']:.2e}; parser exact on all rows: {summ['parse_all_ok']}", flush=True)
    sys.exit(0 if summ["pass"] else 1)


if __name__ == "__main__":
    main()
