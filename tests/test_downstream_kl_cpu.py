"""CPU smoke tests for the downstream-KL machinery and the hard-lag critic.

Tiny random Qwen3 (4 layers, d=64, vocab 512) so the whole thing runs in seconds:
  * build_kl_gold_cache: keys per prompt group, [N,k] tensors
  * downstream_kl_reward: patching the GOLD vector back in gives KL ~ 0; a noisy
    reconstruction gives a strictly more negative reward
  * downstream_kl_critic_loss: finite loss, gradient reaches the critic's value head
  * train_sft.ar_kl_loss_batched: same, through the SFT path
  * CriticEMA(lag_steps=2): shadow refreshes only every 2nd update
"""
import os
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

from nla.critic_ema import CriticEMA  # noqa: E402

torch.manual_seed(0)
D, V, L = 64, 512, 4
KL_LAYER = 1


class FakeTok:
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [1 + (ord(c) * 7 + i) % (V - 1) for i, c in enumerate(text)]


def _tiny_model():
    cfg = Qwen3Config(hidden_size=D, intermediate_size=128, num_hidden_layers=L,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=V,
                      max_position_embeddings=512, head_dim=16, tie_word_embeddings=False)
    m = Qwen3ForCausalLM(cfg).float().eval()
    return m


@pytest.fixture(scope="module")
def actor():
    m = _tiny_model()
    peft = get_peft_model(m, LoraConfig(r=4, lora_alpha=4, target_modules=["q_proj", "v_proj"]))
    peft.eval()
    return peft


@pytest.fixture(scope="module")
def critic_and_base(tmp_path_factory):
    from nla.train_sft import init_critic_from_base
    d = tmp_path_factory.mktemp("tiny")
    base = _tiny_model()
    base.save_pretrained(d)
    critic = init_critic_from_base(str(d), KL_LAYER + 1, torch.float32, None,
                                   device_map=None, max_memory=None, strip_final_norm=True)
    return critic, base


def _batch(n=4):
    sources = ["the quick brown fox jumps over the lazy dog " * 3,
               "a b c d e f g h i j k l m n o p q r s t u v w x y z " * 2] * (n // 2)
    golds = [torch.randn(D) * 3 for _ in range(n)]
    pg = [0, 0, 1, 1][:n]
    return sources, golds, pg


def test_gold_cache_and_reward(actor):
    from nla.train_rl_vllm import build_kl_gold_cache, downstream_kl_reward
    tok = FakeTok()
    sources, golds, pg = _batch()
    vref = [None]
    cache = build_kl_gold_cache(actor, tok, vref, sources, golds, pg, KL_LAYER, "cpu",
                                n_future=3, top_k=8, ctx_tokens=32, batch_size=2)
    assert set(cache) == {0, 1}
    ctx, cont, tki, tlp = cache[0]
    assert len(cont) == 3 and tki.shape == (3, 8) and tlp.shape == (3, 8)
    assert vref[0] is None
    # identical reconstruction -> KL ~ 0
    r_same, cache2 = downstream_kl_reward(actor, tok, vref, sources, golds, golds, pg, KL_LAYER, "cpu",
                                          n_future=3, top_k=8, ctx_tokens=32, decay=0.9,
                                          gold_cache=cache)
    assert cache2 is cache
    # ~0 up to padding/kernel numerics (different batch compositions in the clean
    # pass vs the reward pass); the pre-fix offset was ~-3.8 here
    assert all(r is not None and abs(r) < 5e-3 for r in r_same), r_same
    noisy = [g + torch.randn(D) * 3 for g in golds]
    r_noisy, _ = downstream_kl_reward(actor, tok, vref, sources, noisy, golds, pg, KL_LAYER, "cpu",
                                      n_future=3, top_k=8, ctx_tokens=32, gold_cache=cache)
    rand = [torch.randn(D) * 3 for _ in golds]
    r_rand, _ = downstream_kl_reward(actor, tok, vref, sources, rand, golds, pg, KL_LAYER, "cpu",
                                     n_future=3, top_k=8, ctx_tokens=32, gold_cache=cache)
    # monotone on average: gold (≈0) > noisy > random
    assert np.mean(r_noisy) < np.mean(r_same) and np.mean(r_rand) < np.mean(r_noisy), (r_same, r_noisy, r_rand)
    # None preds are skipped, not crashed
    r_part, _ = downstream_kl_reward(actor, tok, vref, sources, [None] + noisy[1:], golds, pg,
                                     KL_LAYER, "cpu", n_future=3, top_k=8, ctx_tokens=32,
                                     gold_cache=cache)
    assert r_part[0] is None and r_part[1] is not None


def test_critic_kl_loss_grad(actor, critic_and_base):
    from nla.train_rl_vllm import build_kl_gold_cache, downstream_kl_critic_loss
    critic, _ = critic_and_base
    tok = FakeTok()
    sources, golds, pg = _batch()
    vref = [None]
    cache = build_kl_gold_cache(actor, tok, vref, sources, golds, pg, KL_LAYER, "cpu",
                                n_future=3, top_k=8, ctx_tokens=32)
    for p in critic.parameters():
        p.requires_grad_(True)
    critic.zero_grad()
    expl_ids = [torch.randint(1, V, (12,)) for _ in range(4)]
    lv, n_used = downstream_kl_critic_loss(critic, actor, tok, vref, expl_ids, pg, cache, KL_LAYER,
                                           mse_scale_f=float(np.sqrt(D)), device="cpu",
                                           n_future=3, top_k=8, decay=0.9, ctx_tokens=32, scale=0.5)
    assert n_used == 4 and np.isfinite(lv) and lv >= 0
    g = critic.value_head.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_sft_ar_kl_loss_batched(critic_and_base):
    from nla.train_sft import ar_kl_loss_batched
    from nla.utils.arch_adapters import resolve_decoder_layers
    critic, base = critic_and_base
    tok = FakeTok()
    for p in critic.parameters():
        p.requires_grad_(True)
    critic.zero_grad()
    rows = [{"prompt": "Summary of the following text: <text>hello world</text> <summary>",
             "detokenized_text_truncated": "the quick brown fox " * 4,
             "activation_vector": np.random.randn(D).astype(np.float32) * 3} for _ in range(4)]
    kl_layer = resolve_decoder_layers(base)[KL_LAYER]
    W = base.get_output_embeddings().weight
    mean_kl, n_used, diag = ar_kl_loss_batched(critic, base, kl_layer, W, rows, tok, float(np.sqrt(D)),
                                               "cpu", KL_LAYER, n_future=3, top_k=8, decay=0.9,
                                               ctx_tokens=32, gen_bs=2, scale=1.0)
    assert n_used == 4 and np.isfinite(mean_kl)
    assert set(diag) >= {"mse", "cos_pred_gold", "pred_selfcos"}
    g = critic.value_head.weight.grad
    assert g is not None and g.abs().sum() > 0


def test_critic_hard_lag():
    lin = torch.nn.Linear(4, 4)
    ema = CriticEMA(lin, 0.0, lag_steps=2)
    assert ema.enabled
    w0 = lin.weight.detach().clone()
    with torch.no_grad():
        lin.weight.add_(1.0)
    ema.update()                       # update 1: no refresh
    with ema.swapped() as on:
        assert on and torch.allclose(lin.weight, w0)
    assert torch.allclose(lin.weight, w0 + 1.0)   # restored
    with torch.no_grad():
        lin.weight.add_(1.0)
    ema.update()                       # update 2: refresh -> snapshot == live
    with ema.swapped():
        assert torch.allclose(lin.weight, w0 + 2.0)
    with pytest.raises(ValueError):
        CriticEMA(lin, 0.98, lag_steps=2)
    sd = ema.state_dict()
    assert sd["lag_steps"] == 2 and sd["n_updates"] == 2
