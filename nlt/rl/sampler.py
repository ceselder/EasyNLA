"""Stratified pair sampler for referential co-training (DECISIONS v1.13 item 1).

Each RL step draws n_classes distinct (i, j) layer pairs (j ~ U{10..34}, i ~ U{9..j-1}) and per_class DISTINCT positions for each, so every
pair has per_class - 1 depth-matched distractor pairs inside the batch: the same (i, j), another position. Distractors for the reward and
the listener are drawn from that in-class set.
"""
from __future__ import annotations
import torch
from nlt.data.extract import K_LO
from nlt.data.finalize import J_LO, J_HI


class StratifiedSampler:
    def __init__(self, store, n_classes: int = 16, per_class: int = 8, j_lo: int = J_LO, j_hi: int = J_HI, i_lo: int = K_LO):
        self.store, self.n_classes, self.per_class, self.j_lo, self.j_hi, self.i_lo = store, n_classes, per_class, j_lo, j_hi, i_lo

    def sample(self, gen: torch.Generator | None = None):
        """-> rows [P], I [P], J [P], cls [P] with P = n_classes * per_class; positions distinct within a class; classes distinct."""
        seen = set(); classes = []
        while len(classes) < self.n_classes:
            j = int(torch.randint(self.j_lo, self.j_hi + 1, (1,), generator=gen)); i = int(torch.randint(self.i_lo, j, (1,), generator=gen))
            if (i, j) in seen and len(seen) < (self.j_hi - self.j_lo + 1) * 10: continue
            seen.add((i, j)); classes.append((i, j))
        rows, I, J, cls = [], [], [], []
        for c, (i, j) in enumerate(classes):
            r = torch.randperm(self.store.N, generator=gen)[: self.per_class]
            rows.append(r); I += [i] * self.per_class; J += [j] * self.per_class; cls += [c] * self.per_class
        return torch.cat(rows), torch.tensor(I), torch.tensor(J), torch.tensor(cls)

    @staticmethod
    def distractors(cls: torch.Tensor, n_dist: int, gen: torch.Generator | None = None) -> torch.Tensor:
        """[P, n_dist] indices of OTHER pairs in the same class (without replacement within a row; n_dist <= per_class - 1)."""
        P = cls.numel(); out = torch.zeros(P, n_dist, dtype=torch.long)
        for p in range(P):
            others = torch.nonzero((cls == cls[p]) & (torch.arange(P) != p)).flatten()
            assert others.numel() >= n_dist, f"class of pair {p} has only {others.numel()} other members (< n_dist={n_dist})"
            out[p] = others[torch.randperm(others.numel(), generator=gen)[:n_dist]]
        return out

    def same_class_partner(self, i: int, j: int, exclude_row: int | None, gen: torch.Generator | None = None):
        """a random store row != exclude_row, used as a depth-matched distractor for replay-pool rows (same (i, j), other position)."""
        while True:
            r = int(torch.randint(0, self.store.N, (1,), generator=gen))
            if r != exclude_row: return r
