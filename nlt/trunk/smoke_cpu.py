"""CPU plumbing test of the TRUNK critic with a TINY random Qwen3 trunk + tiny prior (no data, no GPU): checks that
 (1) the cached (TextKV) forward equals the full-sequence (TextIDs) forward,  (2) a row with all text keys masked equals the null path,
 (3) the prefix cache is restored after every call, (4) exact_logp / pair_fm_loss run and are finite, (5) state() round-trips.
  python -m nlt.trunk.smoke_cpu
"""
import os, tempfile, torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from nlt.critic.model import PairDenoiser, pair_fm_loss
from nlt.eval_bits.exact import exact_logp, make_probe_bank
from nlt.trunk.model import TrunkCritic, TextIDs


def main():
    torch.manual_seed(0); dev = "cpu"; d = 32
    tmp = tempfile.mkdtemp()
    cfg = Qwen3Config(vocab_size=1000, hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512, tie_word_embeddings=False)
    lm = Qwen3ForCausalLM(cfg); lm.save_pretrained(tmp)
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers
    t = Tokenizer(models.WordLevel({w: i for i, w in enumerate(["[PAD]", "[UNK]"] + [f"w{i}" for i in range(998)])}, unk_token="[UNK]")); t.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=t, pad_token="[PAD]", unk_token="[UNK]")
    tok.save_pretrained(tmp)
    prior = PairDenoiser(d=d, d_model=64, d_mlp=128, n_layers=2, cond="none", target="delta").to(dev).eval()
    m = TrunkCritic(prior, tmp, n_layers=4, lora_r=4, lora_alpha=8, n_act_tokens=2, fresh_every=2, fresh_heads=2, fresh_dhead=8, grad_ckpt=False, device=dev, space={"target": "delta", "src_rms": False, "squash": 1.0}, dtype=torch.float32)
    # make the readout / fresh blocks non-trivial so the test is not vacuous
    with torch.no_grad():
        m.readout.weight.normal_(0, 0.05); m.readout.bias.normal_(0, 0.05)
        for blk in m.fresh: blk.out.weight.normal_(0, 0.05)
    B = 5; x_t = torch.randn(B, d); t = torch.rand(B); h_i = torch.randn(B, d) * 3
    texts = ["w1 w2 w3 w4", "w7 w8", "", "w9 w1 w2 w3 w4 w5 w6", "w3"]
    ids, mask = m.tokenize(texts)
    m.eval()
    with torch.no_grad():
        v_full = m(x_t, t, h_i, enc=TextIDs(ids), enc_mask=mask)
        kv, mask2 = m.encode(texts); T0 = kv.cache.layers[0].keys.shape[-2]
        v_kv = m(x_t, t, h_i, enc=kv, enc_mask=mask2)
        assert kv.cache.layers[0].keys.shape[-2] == T0, "cache not restored"
        v_kv2 = m(x_t, t, h_i, enc=kv, enc_mask=mask2)
        v_null = m(x_t, t, h_i)
        v_prior = prior(x_t, t, h_i)
    print("full vs cached max|diff|", (v_full - v_kv).abs().max().item(), " repeat", (v_kv - v_kv2).abs().max().item())
    print("empty-text row vs null path", (v_full[2] - v_null[2]).abs().max().item(), " (text row vs null, should be > 0):", (v_full[0] - v_null[0]).abs().max().item())
    print("readout delta magnitude", (v_full - v_prior).abs().mean().item())
    assert (v_full - v_kv).abs().max() < 1e-4, "cached != full"
    assert (v_full[2] - v_null[2]).abs().max() < 1e-4, "empty text != null"
    # dropped row via enc_mask (all False) must equal the null path in both modes
    mask_drop = mask.clone(); mask_drop[0] = False
    with torch.no_grad():
        v_d_full = m(x_t, t, h_i, enc=TextIDs(ids), enc_mask=mask_drop); v_d_kv = m(x_t, t, h_i, enc=kv, enc_mask=mask_drop)
    print("dropped row full vs null", (v_d_full[0] - v_null[0]).abs().max().item(), " cached vs null", (v_d_kv[0] - v_null[0]).abs().max().item())
    assert (v_d_full[0] - v_null[0]).abs().max() < 1e-4 and (v_d_kv[0] - v_null[0]).abs().max() < 1e-4
    # multi-group forward == single forwards (fp32), dropped group == null path
    G = 3; tg = torch.rand(B, G); xg = torch.randn(B, G, d); keep = torch.ones(B, G, dtype=torch.bool); keep[1, 1] = False; keep[2, :] = False
    with torch.no_grad():
        vg = m.multi_forward(xg, tg, h_i, ids, mask, keep)
        for g in range(G):
            mk = mask & keep[:, g][:, None]
            vs = m(xg[:, g], tg[:, g], h_i, enc=TextIDs(ids), enc_mask=mk)
            print(f"group {g}: multi vs single max|diff| {(vg[:, g] - vs).abs().max().item():.2e}"); assert (vg[:, g] - vs).abs().max() < 1e-4
        vn = m(xg[1, 1][None], tg[1, 1][None], h_i[1][None])
        print("dropped group vs null", (vg[1, 1] - vn[0]).abs().max().item()); assert (vg[1, 1] - vn[0]).abs().max() < 1e-4
    # gradient checkpointing: multi-group forward + a single forward, then ONE backward (recompute must match) -- this is the training-step pattern
    mc = TrunkCritic(prior, tmp, n_layers=4, lora_r=4, lora_alpha=8, n_act_tokens=2, fresh_every=2, fresh_heads=2, fresh_dhead=8, grad_ckpt=True, device=dev, space=m.space, dtype=torch.float32)
    mc.load_state(m.state()); mc.train()
    v_multi = mc.multi_forward(xg, tg, h_i, ids, mask, keep); v_single = mc(x_t, t, h_i, enc=TextIDs(ids), enc_mask=mask)
    (v_multi.pow(2).mean() + v_single.pow(2).mean()).backward()
    assert all(p.grad is not None for p in mc.lora_parameters()), "no LoRA grads under checkpointing"
    with torch.no_grad(): assert (v_multi - vg).abs().max() < 1e-4 and (v_single - v_full).abs().max() < 1e-4
    print("grad-ckpt multi+single backward OK")
    # training loss + grads
    m.train(); x0 = torch.randn(B, d)
    loss, tt, kept = pair_fm_loss(m, x0, h_i, enc=TextIDs(ids), enc_mask=mask, p_uncond=0.5)
    loss.mean().backward()
    g_lora = sum(float(p.grad.abs().sum()) for p in m.lora_parameters() if p.grad is not None); g_ad = sum(float(p.grad.abs().sum()) for p in m.adapter_parameters() if p.grad is not None)
    print("fm loss", loss.mean().item(), "kept", kept.tolist(), "grad lora", g_lora, "grad adapters", g_ad); assert g_lora > 0 and g_ad > 0
    m.eval()
    pb = make_probe_bank(4, 1, d, torch.Generator().manual_seed(1))
    lp_c = exact_logp(m, x0, h_i, enc=kv, enc_mask=mask2, n_steps=4, probe_bank=pb); lp_u = exact_logp(m, x0, h_i, n_steps=4, probe_bank=pb)
    print("exact logp cond", lp_c.tolist(), "uncond", lp_u.tolist()); assert torch.isfinite(lp_c).all() and torch.isfinite(lp_u).all()
    assert abs(float(lp_c[2] - lp_u[2])) < 1e-3, "empty text must give the unconditional log p"
    m3 = TrunkCritic(prior, tmp, n_layers=4, lora_r=4, lora_alpha=8, n_act_tokens=2, fresh_every=2, fresh_heads=2, fresh_dhead=8, grad_ckpt=False, device=dev, space=m.space, dtype=torch.float32, readout_rank=16)
    with torch.no_grad(): v3 = m3(x_t, t, h_i, enc=TextIDs(ids), enc_mask=mask); assert (v3 - prior(x_t, t, h_i)).abs().max() < 1e-6, "low-rank readout must start as the prior"
    m3.load_state(m3.state()); print("low-rank readout ok")
    # text-pool variant: cached == full == multi, empty == null, nonzero text dependence
    mp = TrunkCritic(prior, tmp, n_layers=4, lora_r=4, lora_alpha=8, n_act_tokens=2, fresh_every=2, fresh_heads=2, fresh_dhead=8, grad_ckpt=False, device=dev, space=m.space, dtype=torch.float32, readout_rank=16, text_pool=True)
    with torch.no_grad():
        mp.readout[1].weight.normal_(0, 0.05)
        vpf = mp(x_t, t, h_i, enc=TextIDs(ids), enc_mask=mask); kvp, maskp = mp.encode(texts); vpk = mp(x_t, t, h_i, enc=kvp, enc_mask=maskp); vpn = mp(x_t, t, h_i)
        vpm = mp.multi_forward(xg, tg, h_i, ids, mask, keep)
        for g in range(G):
            vs = mp(xg[:, g], tg[:, g], h_i, enc=TextIDs(ids), enc_mask=mask & keep[:, g][:, None]); assert (vpm[:, g] - vs).abs().max() < 1e-4, "pool multi != single"
    print("text-pool: cached vs full", (vpf - vpk).abs().max().item(), "empty vs null", (vpf[2] - vpn[2]).abs().max().item(), "text vs null", (vpf[0] - vpn[0]).abs().max().item())
    assert (vpf - vpk).abs().max() < 1e-4 and (vpf[2] - vpn[2]).abs().max() < 1e-5 and (vpf[0] - vpn[0]).abs().max() > 1e-3
    mp.load_state(mp.state()); print("text-pool ok")
    st = m.state(); m2 = TrunkCritic(prior, tmp, n_layers=4, lora_r=4, lora_alpha=8, n_act_tokens=2, fresh_every=2, fresh_heads=2, fresh_dhead=8, grad_ckpt=False, device=dev, space=m.space, dtype=torch.float32); m2.load_state(st); m2.eval()
    with torch.no_grad(): v2 = m2(x_t, t, h_i, enc=TextIDs(ids), enc_mask=mask)
    print("state round-trip max|diff|", (v2 - v_full).abs().max().item()); assert (v2 - v_full).abs().max() < 1e-5
    print("SMOKE OK")


if __name__ == "__main__":
    main()
