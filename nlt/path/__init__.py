"""PATH verbalizer (DECISIONS v1.26, user suggestion): the verbalizer sees h_i, then every write between i and j (attention output a_k and
MLP output m_k for k = i+1..j, or the per-layer deltas d_k = h_k - h_{k-1} as the extraction-free fallback), then h_j -- one norm-matched
' ?' marker per vector at the output of block 1 (the activation-oracle recipe generalised to N markers). No layer labels anywhere.

  prompt.py    build_path_prompt(tok, n_mid)      ' ? \\n' x (2 + n_mid) + question
  inject.py    MultiMarkerInjector                 ref[0] = (vecs [B, Nmax, d], pos [B, Nmax], -1 = pad)
  vectors.py   path_inputs(...)                    [h_i, writes..., h_j] per row from the ActStore (+ PathStore for attn/mlp)
  extract.py   re-forward the stored docs -> a_k, m_k at the stored positions (Modal app nlt-path) + the h_i + sum = h_j check
  sft.py       controlled SFT (same rows / init / hparams as V0b; only the input differs) + eval-only mode
  dump.py      HF-generate rollouts on the fixed val pairs in the board #31 text format
"""
