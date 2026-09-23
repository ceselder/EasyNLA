"""Lens-diff describer: automatic text about what the forward pass changed between h_i and h_j.

Given two residual activations (and the layer indices, used ONLY to pick the per-layer lens map), the
describer reads both through a lens (J-lens by default), compares the two vocabulary readouts and writes
natural-sounding text at several verbosity levels:

  L0  one phrase  (<= ~8 tokens)          "from 'the' to 'Paris'"
  L1  one sentence                         top-1 movement + confidence + 2-3 risers / fallers
  L2  ~three sentences                     + emerging concepts, fading concepts, how much the representation moved
  L3  lists                                top-20 rising / falling tokens, top-10 now vs before, numbers

Hard rules (PLAN.md): no layer tags or depth words (regex-enforced, see FORBIDDEN), no verbatim restatement
of the preceding text (only single lens tokens appear; n-gram copy rate is measured by `copy_stats`).
Implicit depth information (confidence words, entropy numbers, cosine buckets) IS present and documented
in notes/STATE_lens.md; probes.py measures how much.
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field

import torch

from .common import D_MODEL, entropy_rows

FORBIDDEN_WORDS = ("layer", "layers", "depth", "depths", "deep", "deeper", "deepest", "shallow", "shallower", "early", "earlier",
                   "earliest", "late", "later", "latest", "block", "blocks", "stage", "stages", "final", "middle", "mid", "halfway",
                   "penultimate", "upstream", "downstream", "residual", "hidden", "sublayer", "transformer", "forward")
FORBIDDEN = re.compile(r"\b(" + "|".join(FORBIDDEN_WORDS) + r")\b|\bL\d+\b|\b\d+\s*(st|nd|rd|th)\b", re.IGNORECASE)

SPECIAL_NAMES = {"\n": "a line break", "\n\n": "a paragraph break", ".": "a period", ",": "a comma", ":": "a colon", ";": "a semicolon",
                 "?": "a question mark", "!": "an exclamation mark", "(": "an opening parenthesis", ")": "a closing parenthesis",
                 '"': "a quotation mark", "'": "an apostrophe", "-": "a hyphen", "--": "a dash", "/": "a slash", "=": "an equals sign",
                 "*": "an asterisk", "#": "a hash sign", "$": "a dollar sign", "%": "a percent sign", "&": "an ampersand", "[": "an opening bracket",
                 "]": "a closing bracket", "{": "an opening brace", "}": "a closing brace", "<": "a less-than sign", ">": "a greater-than sign",
                 " ": "a space", "  ": "a double space", "\t": "a tab", "_": "an underscore", "+": "a plus sign", "@": "an at sign", "...": "an ellipsis"}
WORD_RE = re.compile(r"^[A-Za-z][A-Za-z'\-]*$")
NUM_RE = re.compile(r"^\d{1,4}$")


def build_vocab_view(tok, V: int | None = None):
    """Per-token display strings and a readability mask over the whole (padded) vocabulary of size V."""
    n_tok = len(tok)
    V = V or n_tok
    display, readable, lower = [], torch.zeros(V, dtype=torch.bool), []
    for t in range(V):
        if t >= n_tok:                      # padding rows of the unembedding
            display.append(""); lower.append(""); continue
        try:
            s = tok.decode([t])
        except Exception:
            s = ""
        core = s.strip()
        ok = False
        word_start = s.startswith(" ") or s.startswith("\n")
        if core in SPECIAL_NAMES and s == core or s in SPECIAL_NAMES:
            core = SPECIAL_NAMES.get(s, SPECIAL_NAMES.get(core)); ok = True
        elif WORD_RE.match(core) and (len(core) > 1 or core in ("a", "I")) and not FORBIDDEN.search(core):
            # word-start tokens always; tokens without a leading space only if they look like a whole word (Capitalised / ALLCAPS / long)
            ok = word_start or core[0].isupper() or len(core) >= 6
        elif NUM_RE.match(core):
            ok = True
        display.append(core if ok else s)
        readable[t] = ok
        lower.append(core.lower() if ok else "")
    return display, readable, lower


def _bucket(x, edges, names):
    for e, n in zip(edges, names):
        if x < e:
            return n
    return names[-1]


CONF_NAMES = ("very unsure", "unsure", "leaning one way", "fairly confident", "nearly certain")
CONF_EDGES = (0.1, 0.3, 0.6, 0.9)


@dataclass
class PairFeatures:
    top1_i: str; top1_j: str; p1_i: float; p1_j: float; ent_i: float; ent_j: float; cos: float
    top1_i_id: int = -1; top1_j_id: int = -1
    risers: list = field(default_factory=list)      # (display, delta_logp, token_id)
    fallers: list = field(default_factory=list)
    top_j: list = field(default_factory=list)       # top readable tokens after
    top_i: list = field(default_factory=list)       # top readable tokens before
    emerging: list = field(default_factory=list)    # in top_j, not in top-50 of i
    fading: list = field(default_factory=list)      # in top_i, not in top-50 of j


class LensDiffDescriber:
    def __init__(self, tok, bank, seed: int = 0, k_list: int = 20, pool: int = 300, k_top: int = 10):
        self.tok, self.bank = tok, bank
        self.V = int(bank.W_U.shape[0])
        self.display, self.readable, self.lower = build_vocab_view(tok, self.V)
        self.readable_dev = None
        self.k_list, self.pool, self.k_top = k_list, pool, k_top
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------ numeric features (batched, GPU)
    @torch.no_grad()
    def features(self, h_i: torch.Tensor, i, h_j: torch.Tensor, j) -> list[PairFeatures]:
        """h_i, h_j: [B, d]; i, j: ints or [B] int tensors (per-layer lens maps only)."""
        B = h_i.shape[0]
        dev = h_i.device
        if self.readable_dev is None or self.readable_dev.device != dev:
            self.readable_dev = self.readable.to(dev)
        i_t = torch.full((B,), int(i), device=dev) if isinstance(i, int) else torch.as_tensor(i, device=dev)
        j_t = torch.full((B,), int(j), device=dev) if isinstance(j, int) else torch.as_tensor(j, device=dev)
        lp_i = torch.empty(B, self.V, device=dev); lp_j = torch.empty_like(lp_i)
        for k in i_t.unique().tolist():
            m = i_t == k; lp_i[m] = self.bank.log_probs(h_i[m], k)
        for k in j_t.unique().tolist():
            m = j_t == k; lp_j[m] = self.bank.log_probs(h_j[m], k)
        ent_i, ent_j = entropy_rows(lp_i), entropy_rows(lp_j)
        p1_i, a_i = lp_i.max(-1); p1_j, a_j = lp_j.max(-1)
        cos = torch.nn.functional.cosine_similarity(h_i.float(), h_j.float(), dim=-1)
        delta = lp_j - lp_i
        neg = torch.finfo(lp_i.dtype).min
        # candidate pools: readable tokens in the top-`pool` of the destination (risers) / source (fallers)
        lp_j_r = lp_j.masked_fill(~self.readable_dev, neg); lp_i_r = lp_i.masked_fill(~self.readable_dev, neg)
        pool_j = lp_j_r.topk(self.pool, dim=-1).indices; pool_i = lp_i_r.topk(self.pool, dim=-1).indices
        d_j = delta.gather(1, pool_j); d_i = delta.gather(1, pool_i)
        ris_order = d_j.argsort(dim=-1, descending=True); fal_order = d_i.argsort(dim=-1)
        risers = pool_j.gather(1, ris_order); risers_d = d_j.gather(1, ris_order)
        fallers = pool_i.gather(1, fal_order); fallers_d = d_i.gather(1, fal_order)
        top50_i = lp_i_r.topk(50, dim=-1).indices; top50_j = lp_j_r.topk(50, dim=-1).indices
        out = []
        cpu = lambda t: t.cpu()
        risers, risers_d, fallers, fallers_d = map(cpu, (risers, risers_d, fallers, fallers_d))
        pool_j, pool_i, top50_i, top50_j = map(cpu, (pool_j, pool_i, top50_i, top50_j))
        a_i, a_j, p1_i, p1_j, ent_i, ent_j, cos = map(cpu, (a_i, a_j, p1_i.exp(), p1_j.exp(), ent_i, ent_j, cos))
        for b in range(B):
            f = PairFeatures(top1_i=self._disp(int(a_i[b])), top1_j=self._disp(int(a_j[b])), p1_i=float(p1_i[b]), p1_j=float(p1_j[b]),
                             ent_i=float(ent_i[b]), ent_j=float(ent_j[b]), cos=float(cos[b]), top1_i_id=int(a_i[b]), top1_j_id=int(a_j[b]))
            skip = {self.lower[int(a_i[b])], self.lower[int(a_j[b])]} - {""}
            f.risers = self._dedupe(risers[b].tolist(), risers_d[b].tolist(), self.k_list, min_delta=0.0, skip=skip)
            f.fallers = self._dedupe(fallers[b].tolist(), fallers_d[b].tolist(), self.k_list, max_delta=0.0, skip=skip)
            f.top_j = self._dedupe(pool_j[b, :40].tolist(), None, self.k_top)
            f.top_i = self._dedupe(pool_i[b, :40].tolist(), None, self.k_top)
            s50_i = {self.lower[t] for t in top50_i[b].tolist()}; s50_j = {self.lower[t] for t in top50_j[b].tolist()}
            f.emerging = [(d, 0.0, t) for d, _, t in f.top_j if d.lower() not in s50_i]
            f.fading = [(d, 0.0, t) for d, _, t in f.top_i if d.lower() not in s50_j]
            out.append(f)
        return out

    def _disp(self, t: int) -> str:
        if self.readable[t]:
            return self.display[t]
        raw = self.tok.decode([t]).strip().replace("'", "")
        return raw if raw else "a whitespace token"

    def _dedupe(self, ids, deltas, k, min_delta=None, max_delta=None, skip=()):
        seen, out = set(skip), []
        for n, t in enumerate(ids):
            if not self.readable[t]:
                continue
            key = self.lower[t]
            if key in seen:
                continue
            d = deltas[n] if deltas is not None else 0.0
            if min_delta is not None and d <= min_delta:
                break
            if max_delta is not None and d >= max_delta:
                break
            seen.add(key); out.append((self.display[t], d, t))
            if len(out) >= k:
                break
        return out

    # ------------------------------------------------------------------ text
    @staticmethod
    def _q(s: str) -> str:
        return s if (s.startswith(("a ", "an ")) and " " in s) else f"'{s}'"

    def _join(self, items, n):
        words = [self._q(d) for d, *_ in items[:n]]
        if not words:
            return ""
        if len(words) == 1:
            return words[0]
        return ", ".join(words[:-1]) + " and " + words[-1]

    def _conf_change(self, f: PairFeatures) -> str:
        dp = f.p1_j - f.p1_i
        if dp > 0.25: return self.rng.choice(["becomes far more confident", "sharpens a lot", "locks in with much more confidence"])
        if dp > 0.07: return self.rng.choice(["grows more confident", "sharpens", "firms up"])
        if dp < -0.25: return self.rng.choice(["loses most of its confidence", "spreads out a lot", "becomes far less sure"])
        if dp < -0.07: return self.rng.choice(["becomes less confident", "spreads out", "loosens"])
        return self.rng.choice(["keeps a similar confidence", "stays about as sure", "holds its confidence"])

    def _move(self, f: PairFeatures) -> str:
        if f.top1_i.lower() != f.top1_j.lower():
            return self.rng.choice([f"the prediction moves from {self._q(f.top1_i)} toward {self._q(f.top1_j)}",
                                    f"{self._q(f.top1_j)} takes over from {self._q(f.top1_i)} as the favourite",
                                    f"the favoured continuation shifts from {self._q(f.top1_i)} to {self._q(f.top1_j)}"])
        return self.rng.choice([f"the prediction stays on {self._q(f.top1_j)}", f"{self._q(f.top1_j)} remains the favourite",
                                f"the guess holds at {self._q(f.top1_j)}"])

    def _cos_words(self, f: PairFeatures) -> str:
        return _bucket(f.cos, (0.5, 0.75, 0.9), ("changes substantially", "shifts moderately", "shifts a little", "barely moves"))

    def text(self, f: PairFeatures, level: int) -> str:
        if level == 0:
            if f.top1_i.lower() != f.top1_j.lower():
                s = self.rng.choice([f"from {self._q(f.top1_i)} to {self._q(f.top1_j)}", f"{self._q(f.top1_j)} replaces {self._q(f.top1_i)}",
                                     f"toward {self._q(f.top1_j)}, away from {self._q(f.top1_i)}"])
            else:
                extra = self._join(f.risers, 1)
                s = f"{self._q(f.top1_j)} holds" + (f"; {extra} gains" if extra else "")
        elif level == 1:
            r, fl = self._join(f.risers, 3), self._join(f.fallers, 2)
            s = f"{self._move(f)} and {self._conf_change(f)}"
            if r: s += f"; {r} {'gains' if len(f.risers) == 1 else 'gain'} ground"
            if fl: s += f" while {fl} {'fades' if len(f.fallers) == 1 else 'fade'}"
            s = s[0].upper() + s[1:] + "."
        elif level == 2:
            b_i, b_j = _bucket(f.p1_i, CONF_EDGES, CONF_NAMES), _bucket(f.p1_j, CONF_EDGES, CONF_NAMES)
            conf = f"going from {b_i} to {b_j}" if b_i != b_j else f"staying {b_j}"
            s1 = f"{self._move(f)}; it {self._conf_change(f)}, {conf}."
            s1 = s1[0].upper() + s1[1:]
            em = self._join(f.emerging, 5) or self._join(f.risers, 5)
            s2 = self.rng.choice([f"Gaining ground: {em}.", f"Newly prominent are {em}.", f"Coming to the fore: {em}."]) if em else "Little new comes to the fore."
            fa = self._join(f.fading, 4) or self._join(f.fallers, 4)
            s3 = (self.rng.choice([f"Fading: {fa}.", f"Receding are {fa}.", f"Losing ground: {fa}."]) if fa else "Nothing notable recedes.")
            s3 += f" Overall the representation {self._cos_words(f)}."
            s = " ".join([s1, s2, s3])
        else:
            ris = ", ".join(d for d, *_ in f.risers[:20]) or "nothing besides the top choice"; fal = ", ".join(d for d, *_ in f.fallers[:20]) or "nothing notable"
            now = ", ".join(d for d, *_ in f.top_j[:10]) or "(none)"; before = ", ".join(d for d, *_ in f.top_i[:10]) or "(none)"
            s = (f"Rising: {ris}. Falling: {fal}. Now favoured: {now}. Previously favoured: {before}. "
                 f"Top choice {f.top1_i} -> {f.top1_j}; confidence {f.p1_i:.2f} -> {f.p1_j:.2f}; entropy {f.ent_i:.1f} -> {f.ent_j:.1f} nats; cosine {f.cos:.2f}.")
        if FORBIDDEN.search(s):
            s = FORBIDDEN.sub("[...]", s)      # belt and braces: the vocab filter already drops these words
        return s

    def describe(self, h_i, i, h_j, j, levels=(0, 1, 2, 3)):
        feats = self.features(h_i, i, h_j, j)
        return [{lvl: self.text(f, lvl) for lvl in levels} for f in feats], feats


# ---------------------------------------------------------------------- copy / leak checks
def copy_stats(text: str, context_ids, tok, n: int = 3) -> dict:
    """Fraction of text word-tokens present in the context and count of shared n-grams (word level)."""
    ctx = tok.decode([t for t in context_ids if t >= 0]) if not isinstance(context_ids, str) else context_ids
    cw = re.findall(r"[A-Za-z']+", ctx.lower()); tw = re.findall(r"[A-Za-z']+", text.lower())
    cset = set(cw)
    grams_c = {tuple(cw[k:k + n]) for k in range(len(cw) - n + 1)}
    grams_t = [tuple(tw[k:k + n]) for k in range(len(tw) - n + 1)]
    shared = sum(g in grams_c for g in grams_t)
    return {"word_in_ctx_frac": (sum(w in cset for w in tw) / max(1, len(tw))), "shared_ngrams": shared, "n_words": len(tw)}


def leak_check(text: str) -> bool:
    return FORBIDDEN.search(text) is None
