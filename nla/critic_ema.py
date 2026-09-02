"""EMA over the co-trained AR critic's trainable parameters (easyNLA RL loop).

Design (per the DSv4 handoff):
  * EMA covers the critic's TRAINABLE params only, selected by requires_grad.
    Under --ar-lora that is LoRA A/B + value_head (~340M on Qwen3.6 27B). With
    a FULL fine-tuned critic (train_rl_vllm's default: --ar-lora off,
    --train-critic on) requires_grad selects the WHOLE backbone, so the fp32
    shadow is ~4 bytes x every critic param -- ~22GB for a 5.5B critic, plus a
    same-size transient during swapped() and again during state_dict(). The
    constructor prints the tracked count; check it before assuming this is
    cheap. The actor is never included: only `critic` is passed in.
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
    def __init__(self, critic: torch.nn.Module, decay: float, lag_steps: int = 0):
        """decay>0: exponential moving average. lag_steps>0: HARD lag (target
        network): the scoring shadow is a snapshot of the live weights refreshed
        every `lag_steps` critic updates (decay is ignored). Both 0: control."""
        self.decay = float(decay)
        self.lag_steps = int(lag_steps)
        self._n_updates = 0
        # A negative decay would silently read as "disabled" (control arm) while
        # still paying the shadow cost; decay >= 1 freezes the shadow at step 0
        # so every rollout is scored by the SFT-init critic. Both are typos, not
        # intentions.
        if self.lag_steps < 0:
            raise ValueError(f"critic lag steps must be >= 0, got {self.lag_steps}")
        if self.lag_steps > 0 and self.decay > 0.0:
            raise ValueError("pass EITHER --critic-ema-decay or --critic-lag-steps, not both")
        if not (0.0 <= self.decay < 1.0):
            raise ValueError(
                f"critic EMA decay must be in [0.0, 1.0), got {self.decay!r}. "
                f"0.0 = control (no EMA); >=1.0 would freeze the shadow forever."
            )
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
        return self.decay > 0.0 or self.lag_steps > 0

    def n_params(self) -> int:
        return sum(p.numel() for p in self._params.values())

    @torch.no_grad()
    def update(self) -> None:
        """Call immediately AFTER critic_optim.step()."""
        if self._swapped:
            raise RuntimeError("CriticEMA.update() while EMA weights are swapped in")
        self._n_updates += 1
        if self.lag_steps > 0:
            # hard lag: refresh the snapshot only every lag_steps updates
            if self._n_updates % self.lag_steps != 0:
                return
            for n, p in self._params.items():
                self._shadow[n].copy_(p.detach().float())
            return
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
        # _swapped is set BEFORE any mutation and the copy-in is inside the try:
        # an exception midway through the copy loop (e.g. OOM on the stash
        # clones) previously escaped before the try, leaving live weights as a
        # MIX of EMA and live with _swapped still False -- so assert_live()
        # passed and the next optimizer step would have trained the EMA copy,
        # the exact landmine this class exists to close.
        self._swapped = True
        try:
            with torch.no_grad():
                self._stash = {n: p.detach().clone()
                               for n, p in self._params.items()}
                for n, p in self._params.items():
                    p.copy_(self._shadow[n].to(p.dtype))
            yield True
        finally:
            # Restore whatever was stashed. On a partial stash (exception during
            # the clone loop) only those entries were ever overwritten, so
            # restoring exactly the stashed keys is correct and complete.
            with torch.no_grad():
                for n, p in self._params.items():
                    if self._stash is not None and n in self._stash:
                        p.copy_(self._stash[n])
            self._stash = None
            self._swapped = False

    def assert_live(self, where: str = "") -> None:
        """Guard to drop in front of critic_optim.step()."""
        if self._swapped:
            raise RuntimeError(f"optimizer step under swapped EMA weights {where}".strip())

    def state_dict(self) -> dict:
        return {"decay": self.decay, "lag_steps": self.lag_steps,
                "n_updates": self._n_updates,
                "shadow": {n: t.clone() for n, t in self._shadow.items()}}

    def load_state_dict(self, sd: dict, strict: bool = True) -> None:
        """Restore the shadow. Does NOT adopt the checkpoint's decay.

        Overwriting self.decay from the checkpoint would let a resume silently
        change sweep arm — resuming a d=0 control checkpoint with
        --critic-ema-decay 0.98 would set decay back to 0.0, flip `enabled` to
        False after the caller's gate had already passed, and quietly turn the
        treatment arm back into a control.
        """
        ckpt_decay = float(sd.get("decay", self.decay))
        if abs(ckpt_decay - self.decay) > 1e-12:
            print(f"[critic-ema] checkpoint decay {ckpt_decay} != requested "
                  f"{self.decay}; KEEPING the requested value (the CLI wins).",
                  flush=True)
        self._n_updates = int(sd.get("n_updates", 0))
        shadow = sd["shadow"]
        missing = [n for n in self._shadow if n not in shadow]
        unexpected = [n for n in shadow if n not in self._shadow]
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"critic EMA shadow does not match this critic: "
                f"{len(missing)} missing, {len(unexpected)} unexpected "
                f"(e.g. missing={missing[:3]}, unexpected={unexpected[:3]}). "
                f"Silently keeping a fresh shadow would reset the EMA horizon "
                f"for this arm only. Pass strict=False to accept a partial load."
            )
        if missing or unexpected:
            print(f"[critic-ema] WARN partial shadow restore: {len(missing)} "
                  f"missing, {len(unexpected)} unexpected", flush=True)
        for n, t in shadow.items():
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
