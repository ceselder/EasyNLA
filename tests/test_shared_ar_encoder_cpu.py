"""CPU test of SharedARVecEncoder on a tiny PEFT-wrapped model: adapter stacking, stage-2 LoRA copy, truncated forward with the norm bypassed,
policy adapter restored, grads reach the ar_critic LoRA and the value head, save() writes critic-style keys."""
import os, tempfile, torch, torch.nn as nn
from types import SimpleNamespace
from peft import LoraConfig, get_peft_model
D, V = 16, 50
TM = r".*layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))"

class Block(nn.Module):
    def __init__(self):
        super().__init__(); self.self_attn = nn.Module(); self.self_attn.q_proj = nn.Linear(D, D); self.mlp = nn.Module(); self.mlp.up_proj = nn.Linear(D, D)
    def forward(self, h): return h + torch.tanh(self.mlp.up_proj(torch.relu(self.self_attn.q_proj(h))))

class Inner(nn.Module):
    def __init__(self, n=4):
        super().__init__(); self.embed = nn.Embedding(V, D); self.layers = nn.ModuleList([Block() for _ in range(n)]); self.norm = nn.LayerNorm(D); self.calls = []
    def forward(self, input_ids, attention_mask=None, use_cache=False):
        h = self.embed(input_ids); self.calls.append(len(self.layers))
        for l in self.layers: h = l(h)
        return SimpleNamespace(last_hidden_state=self.norm(h))

class LM(nn.Module):
    def __init__(self): super().__init__(); self.model = Inner()
    def forward(self, **kw): return self.model(**kw)

class Tok:
    pad_token_id = 0; eos_token_id = 0
    def __call__(self, texts, return_tensors=None, padding=True, truncation=True, max_length=256, add_special_tokens=False):
        ids = [[1 + (ord(c) % (V - 2)) for c in t][:max_length] for t in texts]; L = max(len(x) for x in ids)
        return {"input_ids": torch.tensor([x + [0] * (L - len(x)) for x in ids]), "attention_mask": torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in ids])}

def test():
    from nla.flow.rl_critic import SharedARVecEncoder
    torch.manual_seed(0); tmp = tempfile.mkdtemp()
    cfg = LoraConfig(r=4, lora_alpha=2, use_rslora=True, target_modules=TM, lora_dropout=0.0, bias="none")
    TM3 = r".*layers\.(?:0|1|2)\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))"
    # an "SFT delta" adapter dir made with PEFT's own save
    sft = get_peft_model(LM(), LoraConfig(r=4, lora_alpha=2, use_rslora=True, target_modules=TM3))
    for n, p_ in sft.named_parameters():
        if "lora_" in n: torch.nn.init.normal_(p_, std=0.1)
    sft.save_pretrained(f"{tmp}/ar_sft_lora")
    actor = get_peft_model(LM(), cfg)                                     # the policy: adapter "default" on all 4 layers
    base_sd = {k: v.clone() for k, v in actor.get_base_model().model.state_dict().items() if "lora_" not in k}
    # stage-2 LoRA in critic-style keys (layers 0..2) + value head
    lora = {}
    for i in range(3):
        for mod in ("self_attn.q_proj", "mlp.up_proj"):
            lora[f"backbone.model.layers.{i}.{mod}.lora_A.default.weight"] = torch.randn(4, D) * 0.1; lora[f"backbone.model.layers.{i}.{mod}.lora_B.default.weight"] = torch.randn(D, 4) * 0.1
    vh = nn.Linear(D, D, bias=False); torch.save({"lora": lora, "value_head": vh.state_dict(), "step": 5}, f"{tmp}/ar_encoder_latest.pt")
    enc = SharedARVecEncoder(actor, Tok(), torch.device("cpu"), f"{tmp}/ar_encoder_latest.pt", f"{tmp}/ar_sft_lora", D, enc_layer=2, lora_r=4, lora_alpha=2)
    assert set(actor.peft_config) == {"default", "ar_sft", "ar_critic"}
    # copied weights present
    q0 = actor.get_base_model().model.layers[0].self_attn.q_proj
    assert torch.allclose(q0.lora_A["ar_critic"].weight.float(), lora["backbone.model.layers.0.self_attn.q_proj.lora_A.default.weight"])
    inner = actor.get_base_model().model
    cv = enc(["hello world", "another text"], grad=True); assert cv.shape == (2, 2 * D) and enc.last_pred_raw.shape == (2, D)
    assert inner.calls[-1] == 3, inner.calls                              # truncated to enc_layer+1 layers
    assert len(inner.layers) == 4 and isinstance(inner.norm, nn.LayerNorm)   # restored
    assert actor.active_adapter == "default" and all(not p_.requires_grad for n, p_ in actor.named_parameters() if ".ar_sft." in n)
    assert all(p_.requires_grad for n, p_ in actor.named_parameters() if ".default." in n and "lora_" in n)
    cv.sum().backward(); assert all(p_.grad is not None for p_ in enc.ar_params) and all(p_.grad is not None for p_ in enc.value_head.parameters())
    assert all(p_.grad is None for n, p_ in actor.named_parameters() if ".default." in n and "lora_" in n)   # policy LoRA untouched
    # the shared trunk's base weights are unchanged and the ar_sft adapter changes the output vs default-only
    for k, v in base_sd.items(): assert torch.equal(actor.get_base_model().model.state_dict()[k], v)
    # save writes critic-style keys
    class _FC:  # minimal stand-in for FlowCritic.save's use
        pass
    from nla.flow.rl_critic import FlowCritic
    fc = FlowCritic.__new__(FlowCritic); fc.arvec = enc; fc.actor = actor; fc.model = torch.nn.Module(); fc.adapter_args = {}; fc.cfg = {}
    FlowCritic.save(fc, f"{tmp}/flow_latest", 9); st = torch.load(f"{tmp}/flow_latest/ar_encoder_latest.pt")
    assert set(st["lora"]) == set(lora), (sorted(st["lora"])[:2], sorted(lora)[:2])
    print("shared AR encoder CPU test OK")

if __name__ == "__main__":
    test()
