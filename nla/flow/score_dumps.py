"""Offline frozen-critic scoring of the eval rollouts dumped by train_rl_vllm (save_dir/eval_rollouts/step_*_r*.pt).

Every eval the trainer writes the 128 held-out explanations it generated plus their gold activations. Scoring them here with the
FROZEN SFT MSE critic gives a fixed-scorer FVE curve every eval step on identical generations, for any reward mode (MSE-critic arms,
flow-critic arms), independent of whatever critic co-trained inside the run. Same conventions as the trainer's eval: critic prompt
template from the NLA sidecar config, unit-L2 normalisation, predict-the-mean baseline of the scored rows, failures excluded."""
from __future__ import annotations
import argparse, glob, json, math, os, re
import torch
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dumps-dir", required=True); p.add_argument("--critic", required=True); p.add_argument("--sidecar", required=True)
    p.add_argument("--out", required=True); p.add_argument("--batch", type=int, default=32)
    a = p.parse_args()
    from nla.models import NLACriticModel
    from nla.utils import critic_predict
    from nla.config import load_nla_config
    from nla.schema import resolve_target_scale, normalize_activation, compute_predict_mean_baselines
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.critic)
    cfg = load_nla_config(a.sidecar, tok); template = cfg.critic_prompt_template; msf = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    critic = NLACriticModel.from_pretrained(a.critic, torch_dtype=torch.bfloat16).to(dev).eval(); critic.requires_grad_(False)
    prev = json.load(open(a.out)) if os.path.exists(a.out) else {}
    files = sorted(glob.glob(os.path.join(a.dumps_dir, "step_*_r*.pt")))
    by_step = {}
    for f in files:
        m = re.search(r"step_(\d+)_r(\d+)\.pt$", f)
        if m: by_step.setdefault(int(m.group(1)), []).append(f)
    out = dict(prev); baseline = None; pad = tok.eos_token_id
    for step in sorted(by_step):
        if str(step) in out and out[str(step)].get("n_files") == len(by_step[step]): continue
        mses, n_rows, live, flow = [], 0, [], []
        for f in by_step[step]:
            d = torch.load(f, map_location="cpu"); acts = d["activations"].float(); expls = d["explanations"]; n_rows += len(expls)
            live += [r for r in d.get("vector_rewards", []) if r is not None and r > -2.0]; flow += [r for r in d.get("flow_rewards", []) if r is not None]
            if baseline is None: _, baseline = compute_predict_mean_baselines(acts, msf)
            idx = [i for i, z in enumerate(expls) if z is not None]
            for cs in range(0, len(idx), a.batch):
                ch = idx[cs: cs + a.batch]
                ids_l = [tok.encode(template.format(explanation=expls[i]), add_special_tokens=False)[:1024] for i in ch]
                T = max(len(x) for x in ids_l); bx = torch.full((len(ch), T), pad, dtype=torch.long, device=dev); am = torch.zeros((len(ch), T), dtype=torch.long, device=dev)
                for r, x in enumerate(ids_l): bx[r, : len(x)] = torch.tensor(x, device=dev); am[r, : len(x)] = 1
                with torch.no_grad(): pred = critic_predict(critic, bx, am, msf)
                gold = acts[ch].to(dev)
                mse = ((normalize_activation(pred.float(), msf) - normalize_activation(gold, msf)) ** 2).mean(1)
                mses += [v for v in mse.tolist() if math.isfinite(v)]
        fve = 100 * (1 - sum(mses) / len(mses) / baseline) if mses else float("nan")
        out[str(step)] = {"step": step, "fve_frozen_sft_critic": fve, "n_valid": len(mses), "n_rows": n_rows, "n_files": len(by_step[step]),
                          "live_vector_fve": (100 * (1 + sum(live) / len(live) / baseline)) if live else None,
                          "flow_reward_mean": (sum(flow) / len(flow)) if flow else None, "baseline_mse": baseline}
        print(f"[score_dumps] step {step}: frozen-critic FVE {fve:.1f}% ({len(mses)}/{n_rows} valid) | live {out[str(step)]['live_vector_fve']} | flow reward {out[str(step)]['flow_reward_mean']}", flush=True)
        json.dump(out, open(a.out, "w"), indent=1)
    print(f"[score_dumps] wrote {a.out} ({len(out)} steps)", flush=True)


if __name__ == "__main__":
    main()
