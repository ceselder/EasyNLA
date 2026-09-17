"""CPU test of the RL flow critic: shared-noise group rewards, FVE readout, co-training grads, save/load. Fake tiny actor."""
import contextlib, os, tempfile, torch, torch.nn as nn
from nla.flow.model import Denoiser
from nla.flow.cond_model import CondDenoiser

D, T = 16, 7

class _Inner(nn.Module):
    def __init__(self):
        super().__init__(); self.emb = nn.Embedding(100, D); self.layers = nn.ModuleList([nn.Linear(D, D) for _ in range(3)])
    def forward(self, input_ids, attention_mask=None, use_cache=False):
        h = self.emb(input_ids)
        for l in self.layers: h = l(h)
        return h

class _Actor(nn.Module):
    def __init__(self):
        super().__init__(); self.model = _Inner()
    @contextlib.contextmanager
    def disable_adapter(self):
        yield
    def get_base_model(self): return self

class _Tok:
    pad_token_id = 0; eos_token_id = 0
    def encode(self, s, add_special_tokens=False): return [1 + (ord(c) % 90) for c in s][:64]

def test_flow_critic():
    from nla.flow.rl_critic import FlowCritic
    torch.manual_seed(0); tmp = tempfile.mkdtemp()
    prior = Denoiser(D, 32, 64, 2)
    torch.save({"args": {"d_input": D, "d_model": 32, "d_mlp": 64, "n_layers": 2}, "model": prior.state_dict()}, f"{tmp}/model.pt")
    cm = CondDenoiser(prior, D, n_slots=4, n_heads=2, d_head=8, gate_rank=4)
    with torch.no_grad():
        for blk in cm.blocks: blk.read.out.weight.normal_(0, 0.05)   # non-trivial adapter so conditioning matters
    torch.save({"adapter": {k: v for k, v in cm.state_dict().items() if ".read." in k or ".gate_mod." in k}, "args": {"n_slots": 4, "n_heads": 2, "d_head": 8, "gate_rank": 4}, "step": 3}, f"{tmp}/adapter.pt")
    torch.save({"mean": torch.zeros(D), "var": torch.ones(D), "n": 10}, f"{tmp}/stats.pt")
    actor, tok = _Actor(), _Tok()
    fc = FlowCritic(tmp, f"{tmp}/adapter.pt", f"{tmp}/stats.pt", actor, tok, torch.device("cpu"), enc_layer=1, micro_batch=3, t_grid=(0.3, 0.7))
    acts = [torch.randn(D) for _ in range(3)]
    expl = ["cat", "cat", "dog", None, "bird", "fish"]; groups = [0, 0, 0, 1, 1, 2]; A = [acts[g] for g in groups]
    fr, vr, preds = fc.score(expl, A, groups, seed=5)
    assert fr[3] is None and vr[3] is None and preds[3] is None
    assert abs(fr[0] - fr[1]) < 1e-5, "same explanation + same group noise -> same reward"
    assert abs(fr[0] - fr[2]) > 1e-6, "different explanation -> different reward"
    fr2, _, _ = fc.score(expl, A, groups, seed=5); assert all((a is None and b is None) or abs(a - b) < 1e-5 for a, b in zip(fr, fr2)), "deterministic in seed"
    fr3, _, _ = fc.score(expl, A, groups, seed=6); assert abs(fr3[0] - fr[0]) > 1e-7, "seed changes the shared noise"
    assert all(-3 < r < 0 for r in fr if r is not None) and all(-3 < r < 0 for r in vr if r is not None)
    # co-training: grads land on the adapter only
    fc.optim.zero_grad(); l = fc.train_backward(expl, A, accum=1); assert l == l and l > 0
    assert all(p.grad is not None for p in fc.trainable) and all(p.grad is None for p in fc.model.prior.parameters())
    fc.optim.step(); fc.save(f"{tmp}/flow_latest", 7); st = fc.load(f"{tmp}/flow_latest/adapter_latest.pt"); assert st == 7
    print("flow RL critic CPU test OK")

if __name__ == "__main__":
    test_flow_critic()
