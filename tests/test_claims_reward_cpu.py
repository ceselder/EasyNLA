"""--claim-reward singles_red in the RL path = sum_i single PMI(c_i) - R_LM(C) - cost*|C| with the shared text-LM redundancy (nla.flow.claim_lm),
LOO credits consistent with it, min_composed = the old definition. Tiny CPU critic + a deterministic stub LM."""
import math, tempfile, torch
from test_flow_rl_critic_cpu import _Actor, _Tok, D
from nla.flow.model import Denoiser
from nla.flow.cond_model import CondDenoiser
from nla.flow.claim_lm import ClaimLM


class _StubLM(ClaimLM):
    def __init__(self):
        self.cache = {}
    def logp(self, sets):   # deterministic, order-dependent, sub-additive enough to make R non-trivial
        return [-0.3 * sum(len(c) for c in s) - 1.7 * len(s) + 0.9 * sum(1 for a, b in zip(s, s[1:]) if a[0] == b[0]) + 0.1 * (len(s[0]) if s else 0) for s in sets]


def _critic():
    torch.manual_seed(0); tmp = tempfile.mkdtemp()
    prior = Denoiser(D, 32, 64, 2)
    torch.save({"args": {"d_input": D, "d_model": 32, "d_mlp": 64, "n_layers": 2}, "model": prior.state_dict()}, f"{tmp}/model.pt")
    cm = CondDenoiser(prior, D, n_slots=4, n_heads=2, d_head=8, gate_rank=4)
    with torch.no_grad():
        for blk in cm.blocks: blk.read.out.weight.normal_(0, 0.05)
    torch.save({"adapter": {k: v for k, v in cm.state_dict().items() if ".read." in k or ".gate_mod." in k}, "args": {"n_slots": 4, "n_heads": 2, "d_head": 8, "gate_rank": 4}, "step": 3}, f"{tmp}/adapter.pt")
    torch.save({"mean": torch.zeros(D), "var": torch.ones(D), "n": 10}, f"{tmp}/stats.pt")
    from nla.flow.rl_critic import FlowCritic
    return FlowCritic(tmp, f"{tmp}/adapter.pt", f"{tmp}/stats.pt", _Actor(), _Tok(), torch.device("cpu"), enc_layer=1, micro_batch=3, t_grid=(0.3, 0.7), eps_per_t=2)


def test_singles_red_matches_eval_definition():
    fc = _critic(); lm = _StubLM(); cost = 5.0
    acts = [torch.randn(D) for _ in range(3)]
    expl = ["• the cat sat down\n• a dog barked loudly\n• cats like milk", "• only one claim here", None, "• birds fly south\n• bees make honey"]
    groups = [0, 0, 1, 2]; A = [acts[g] for g in groups]
    E = {(g, k): torch.randn(D, generator=torch.Generator().manual_seed(100 + 10 * g + k)) for g in range(3) for k in range(2)}
    ef = lambda g, k: E[(g, k)]
    r = fc.score_claims_composed(expl, A, groups, cost=cost, loo_rows=[0, 1, 3], reward="singles_red", lm=lm, eps_fn=ef)
    assert r["reward"][2] is None
    # independent single-claim PMI with the same eps: (d/2) mean_{t,k}[ |v0 - tgt|^2 - |v_c - tgt|^2 ] / d
    from nla.flow.claims import split_claims
    for i in (0, 1, 3):
        cl = split_claims(expl[i]); x0 = fc.norm.normalize(A[i][None].float()); ref = []
        enc, mk = fc._tok_states(cl)
        for j in range(len(cl)):
            acc = 0.0
            for k in range(2):
                eps = E[(groups[i], k)][None]; tgt = eps - x0
                for tv in (0.3, 0.7):
                    t = torch.full((1,), tv); xt = (1 - tv) * x0 + tv * eps
                    with torch.no_grad(), fc._ac():
                        v0 = fc.model(xt, t).float(); vc = fc.model(xt, t, enc[j:j + 1], mk[j:j + 1]).float()
                    acc += (((v0 - tgt) ** 2).mean() - ((vc - tgt) ** 2).mean()).item()
            ref.append(0.5 * D * acc / 4)
        sg = r["singles"][i]
        assert all(abs(a - b) < 2e-3 + 0.02 * abs(a) for a, b in zip(ref, sg)), (ref, sg)          # same definition (bf16 autocast, different batching)
        R = lm.redundancy(cl)
        if len(cl) > 1: assert abs(R - (0.5 * (lm.logp([tuple(cl)])[0] + lm.logp([tuple(cl[::-1])])[0]) - sum(lm.logp([(c,) for c in cl])))) < 1e-9
        assert abs(r["reward"][i] - (sum(sg) - R - cost * len(cl))) < 1e-9                     # exact formula
        loo = [R - lm.redundancy(cl[:j] + cl[j + 1:]) for j in range(len(cl))]
        assert all(abs(c - (s_ - l)) < 1e-9 for c, s_, l in zip(r["credits"][i], sg, loo))
    # the old definition only behind min_composed
    r2 = fc.score_claims_composed(expl, A, groups, cost=cost, reward="min_composed", eps_fn=ef)
    for i in (0, 1, 3):
        assert abs(r2["reward"][i] - (min(sum(r2["singles"][i]), r2["pmi"][i]) - cost * len(split_claims(expl[i])))) < 1e-5
    try: fc.score_claims_composed(expl, A, groups, reward="singles_red", eps_fn=ef); assert False, "singles_red without an LM must fail"
    except AssertionError as e: assert "needs the text LM" in str(e)


class _StubSim:
    """deterministic similarity: 0.9 for claims sharing their first word, else 0.1"""
    def matrix(self, claims):
        m = len(claims); S = torch.zeros(m, m)
        for i in range(m):
            for j in range(m): S[i, j] = 1.0 if i == j else (0.9 if claims[i].split()[0] == claims[j].split()[0] else 0.1)
        return S


def test_redundancy_modes():
    from nla.flow.claim_redundancy import ClaimRedundancy, semdup_score
    fc = _critic(); lm = _StubLM(); sim = _StubSim(); cost = 3.0
    acts = [torch.randn(D) for _ in range(2)]; expl = ["• cats sit on mats\n• cats sit on rugs\n• dogs bark at night", "• birds fly south"]; groups = [0, 1]; A = [acts[g] for g in groups]
    ef = lambda g, k: torch.randn(D, generator=torch.Generator().manual_seed(7 + 3 * g + k))
    from nla.flow.claims import split_claims
    for red in (ClaimRedundancy("lm", lm=lm, alpha=2.0), ClaimRedundancy("semdup", sim=sim), ClaimRedundancy("semdup", sim=sim, floor=0.5)):
        r = fc.score_claims_composed(expl, A, groups, cost=cost, loo_rows=[0, 1], reward="singles_red", redundancy=red, eps_fn=ef)
        for i in (0, 1):
            cl = split_claims(expl[i]); v = r["singles"][i]
            want = (sum(v) - 2.0 * lm.redundancy(cl)) if red.mode == "lm" else semdup_score(cl, v, sim, red.floor)
            assert abs(r["reward"][i] - (want - cost * len(cl))) < 1e-9
            loo_want = [want - ((sum(v[:j] + v[j + 1:]) - 2.0 * lm.redundancy(cl[:j] + cl[j + 1:])) if red.mode == "lm" else semdup_score(cl[:j] + cl[j + 1:], v[:j] + v[j + 1:], sim, red.floor)) for j in range(len(cl))]
            assert all(abs(x - y) < 1e-9 for x, y in zip(r["credits"][i], loo_want))
    # semdup discounts the near-duplicate (second "cats" claim) by 0.9 and leaves the distinct one alone
    v = [10.0, 10.0, 10.0]; assert abs(semdup_score(["cats a b c", "cats d e f", "dogs g h i"], v, sim) - (10 + 1 + 9)) < 1e-6
