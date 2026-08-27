"""Tests for CriticEMA (--critic-ema-decay): the EMA'd AR critic scorer.

One test per property the design depends on:
  1  shadow initialised to live weights (so d=0 and "no EMA" are one run)
  2  recurrence ema = d*ema + (1-d)*live holds exactly over many steps
  3  only TRAINABLE params are tracked (frozen backbone excluded)
  4  shadow is NOT a registered buffer (cannot leak into module.state_dict())
  5  swapped() puts EMA weights in the live slots, and restores bit-exactly
  6  swapped() restores even when the body RAISES  (the finally landmine)
  7  update() refuses to run while weights are swapped
  8  assert_live() refuses an optimizer step while weights are swapped
  9  d=0.0 is the control: disabled, swap is a no-op, shadow tracks live
  10 state_dict / load_state_dict round-trip (resume keeps the average)
  11 NoEMA has the same surface (frozen-critic path stays branch-free)

Run: python tests/test_critic_ema.py   (CPU-only, no model download.)
"""

import sys

import torch
import torch.nn as nn

sys.path.insert(0, ".")
from nla.critic_ema import CriticEMA, NoEMA

FAILS = []


def check(cond, label):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        FAILS.append(label)


class FakeCritic(nn.Module):
    """Mimics the real critic's param layout: frozen backbone + LoRA A/B + value_head."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(8, 8, bias=False)
        self.lora_A = nn.Linear(8, 4, bias=False)
        self.lora_B = nn.Linear(4, 8, bias=False)
        self.value_head = nn.Linear(8, 3, bias=False)
        for p in self.parameters():
            p.requires_grad_(False)
        for m in (self.lora_A, self.lora_B, self.value_head):
            for p in m.parameters():
                p.requires_grad_(True)


def jitter(critic, scale=0.1):
    """Stand in for an optimizer step."""
    with torch.no_grad():
        for p in critic.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * scale)


def main():
    torch.manual_seed(0)

    # 1 — shadow == live at construction
    c = FakeCritic()
    ema = CriticEMA(c, 0.98)
    check(all(torch.equal(ema._shadow[n], p.detach().float())
              for n, p in c.named_parameters() if p.requires_grad),
          "1  shadow initialised to live weights")

    # 2 — recurrence holds over 25 steps, checked against an independent copy
    c = FakeCritic()
    ema = CriticEMA(c, 0.98)
    ref = {n: p.detach().clone().float() for n, p in c.named_parameters() if p.requires_grad}
    for _ in range(25):
        jitter(c)
        ema.update()
        for n, p in c.named_parameters():
            if p.requires_grad:
                ref[n] = 0.98 * ref[n] + 0.02 * p.detach().float()
    check(all(torch.allclose(ema._shadow[n], ref[n], atol=1e-6) for n in ref),
          "2  ema = d*ema + (1-d)*live holds over 25 steps")

    # 3 — frozen backbone excluded
    check("backbone.weight" not in ema._shadow
          and {"lora_A.weight", "lora_B.weight", "value_head.weight"} == set(ema._shadow),
          "3  only trainable params tracked (backbone excluded)")

    # 4 — shadow is not a buffer / not in the module's state_dict
    check(not any("shadow" in k or "ema" in k for k in c.state_dict()),
          "4  shadow absent from module.state_dict() (not a buffer)")

    # 5 — swap installs EMA weights, restore is bit-exact
    c = FakeCritic()
    ema = CriticEMA(c, 0.9)
    for _ in range(5):
        jitter(c)
        ema.update()
    live_before = {n: p.detach().clone() for n, p in c.named_parameters()}
    with ema.swapped() as active:
        swapped_in = all(
            torch.allclose(p.detach(), ema._shadow[n].to(p.dtype), atol=1e-6)
            for n, p in c.named_parameters() if p.requires_grad)
        differs = not torch.equal(c.lora_A.weight.detach(), live_before["lora_A.weight"])
    restored = all(torch.equal(p.detach(), live_before[n]) for n, p in c.named_parameters())
    check(active and swapped_in and differs and restored,
          "5  swapped() installs EMA weights and restores bit-exactly")

    # 6 — restore happens even on an exception inside the body
    c = FakeCritic()
    ema = CriticEMA(c, 0.9)
    for _ in range(3):
        jitter(c)
        ema.update()
    live_before = {n: p.detach().clone() for n, p in c.named_parameters()}
    try:
        with ema.swapped():
            raise RuntimeError("simulated scoring crash")
    except RuntimeError:
        pass
    check(all(torch.equal(p.detach(), live_before[n]) for n, p in c.named_parameters())
          and not ema._swapped,
          "6  live weights restored after an exception mid-scoring")

    # 7 / 8 — no update() and no optimizer step under swapped weights
    got_update, got_step = False, False
    with ema.swapped():
        try:
            ema.update()
        except RuntimeError:
            got_update = True
        try:
            ema.assert_live("(test)")
        except RuntimeError:
            got_step = True
    check(got_update, "7  update() refuses to run while swapped")
    check(got_step, "8  assert_live() blocks an optimizer step while swapped")

    # 9 — d=0 control arm
    c = FakeCritic()
    ema0 = CriticEMA(c, 0.0)
    jitter(c)
    ema0.update()
    live_before = {n: p.detach().clone() for n, p in c.named_parameters()}
    with ema0.swapped() as active:
        untouched = all(torch.equal(p.detach(), live_before[n]) for n, p in c.named_parameters())
    check(not ema0.enabled and active is False and untouched
          and torch.allclose(ema0._shadow["lora_A.weight"],
                             c.lora_A.weight.detach().float(), atol=1e-6),
          "9  d=0.0 is the control: disabled, swap is a no-op, shadow == live")

    # 10 — state_dict round-trip
    c = FakeCritic()
    ema = CriticEMA(c, 0.995)
    for _ in range(7):
        jitter(c)
        ema.update()
    sd = ema.state_dict()
    fresh = CriticEMA(FakeCritic(), 0.995)
    fresh.load_state_dict(sd)
    check(all(torch.allclose(fresh._shadow[n], ema._shadow[n], atol=1e-6) for n in ema._shadow)
          and fresh.decay == 0.995,
          "10  state_dict/load_state_dict round-trips the shadow")

    # 11 — NoEMA surface
    n = NoEMA()
    with n.swapped() as active:
        pass
    n.update(); n.assert_live()
    check(n.enabled is False and active is False and n.n_params() == 0,
          "11  NoEMA matches the CriticEMA surface")

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILURE(S): " + "; ".join(FAILS))
        return 1
    print("all critic-EMA properties hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
