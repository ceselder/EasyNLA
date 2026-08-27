"""EMA over the co-trained AR critic's trainable parameters (easyNLA RL loop).

Design (per the DSv4 handoff):
  * EMA covers the critic's TRAINABLE params only -- attention/shared LoRA A/B,
    value_head, dense-adapter A/B. The frozen backbone and the actor are excluded.
  * shadow initialised to live weights at step 0.
  * after every critic optimizer step:  ema = d*ema + (1-d)*live
  * applied at REWARD SCORING via shadow-swap; the critic's supervised co-train
    loss and optimizer always run on LIVE weights.
  * sweep  --critic-ema-decay  in {0.0 = control, 0.98, 0.995}

Landmines this encodes:
  1. shadow lives in a plain dict, NOT registered buffers -- buffers ride along
     in state_dict()/save and DDP broadcasts and would silently contaminate
     checkpoints.
  2. restore happens in `finally`, so a crash mid-scoring cannot leave EMA
     weights sitting in the live slots (the next step would then train the EMA).
  3. update() and the optimizer refuse to run while weights are swapped in.
  4. the actor is never touched -- only params handed in via `critic`.
"""
from __future__ import annotations

import contextlib

import torch


class CriticEMA:
    def __init__(self, critic: torch.nn.Module, decay: float):
        self.decay = float(decay)
        # requires_grad is the selector: picks up LoRA A/B + value_head +
        # adapter A/B and excludes the frozen backbone by construction.
        self._params = {n: p for n, p in critic.named_parameters() if p.requires_grad}
        if not self._params:
            raise ValueError("CriticEMA: critic has no trainable params")
        # float32 shadow so the average stays sane under a bf16 critic.
        self._shadow = {n: p.detach().clone().float() for n, p in self._params.items()}
        self._stash: dict[str, torch.Tensor] | None = None
        self._swapped = False

    @property
    def enabled(self) -> bool:
        return self.decay > 0.0

    def n_params(self) -> int:
        return sum(p.numel() for p in self._params.values())

    @torch.no_grad()
    def update(self) -> None:
        """Call immediately AFTER critic_optim.step()."""
        if self._swapped:
            raise RuntimeError("CriticEMA.update() while EMA weights are swapped in")
        d = self.decay
        for n, p in self._params.items():
            s, live = self._shadow[n], p.detach().float()
            s.copy_(live) if d == 0.0 else s.mul_(d).add_(live, alpha=1.0 - d)

    @contextlib.contextmanager
    def swapped(self):
        """Score rollouts under the EMA critic; live weights always restored.

        Yields True if EMA weights are actually in place, False in the d=0
        control arm (where scoring under live weights *is* the control).
        """
        if not self.enabled:
            yield False
            return
        if self._swapped:
            raise RuntimeError("CriticEMA.swapped() is not re-entrant")
        with torch.no_grad():
            self._stash = {n: p.detach().clone() for n, p in self._params.items()}
            for n, p in self._params.items():
                p.copy_(self._shadow[n].to(p.dtype))
        self._swapped = True
        try:
            yield True
        finally:
            with torch.no_grad():
                for n, p in self._params.items():
                    p.copy_(self._stash[n])
            self._stash = None
            self._swapped = False

    def assert_live(self, where: str = "") -> None:
        """Guard to drop in front of critic_optim.step()."""
        if self._swapped:
            raise RuntimeError(f"optimizer step under swapped EMA weights {where}".strip())

    def state_dict(self) -> dict:
        return {"decay": self.decay,
                "shadow": {n: t.clone() for n, t in self._shadow.items()}}

    def load_state_dict(self, sd: dict) -> None:
        self.decay = float(sd["decay"])
        for n, t in sd["shadow"].items():
            if n in self._shadow:
                self._shadow[n].copy_(t.to(self._shadow[n].dtype))


class NoEMA:
    """Null object for the frozen-critic path (`--no-train-critic`).

    Same surface as CriticEMA so the call sites stay branch-free: `swapped()`
    yields False, everything else is a no-op.
    """

    enabled = False
    decay = 0.0

    def n_params(self) -> int:
        return 0

    def update(self) -> None:
        pass

    def assert_live(self, where: str = "") -> None:
        pass

    @contextlib.contextmanager
    def swapped(self):
        yield False

    def state_dict(self) -> dict:
        return {"decay": 0.0, "shadow": {}}

    def load_state_dict(self, sd: dict) -> None:
        pass
