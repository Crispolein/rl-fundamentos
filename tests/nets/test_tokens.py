import numpy as np
import pytest
import torch

from rlf.core import VecEnv, to_torch, TorchObs
from rlf.envs.cartas import CartasEnv
from rlf.envs.despacho import DespachoEnv
from rlf.nets.tokens import MiniPTCGNet

CPU = torch.device("cpu")
MEDIANO = lambda: DespachoEnv.from_size("mediano")


def sample_obs(make, n_envs=50, steps=10, seed=0, n_batches=1):
    vec = VecEnv(make, n_envs, seed=seed)
    obs = vec.reset()
    rng = np.random.default_rng(seed)
    out = []
    for i in range(steps * n_batches):
        obs = vec.step(rng.integers(0, obs.n_options)).obs
        if i % steps == steps - 1:
            out.append(to_torch(obs, CPU))
    return out


def permute_entities(t: TorchObs, perm: torch.Tensor) -> TorchObs:
    """Reorder entity rows by `perm` [B, E] (new row i = old row perm[i]) and remap option_refs accordingly."""
    B, E = perm.shape
    inv = torch.argsort(perm, dim=1)                                    # old index j -> new position inv[j]
    ent = t.entities.gather(1, perm[:, :, None].expand(-1, -1, t.entities.shape[2]))
    mask = t.entity_mask.gather(1, perm)
    refs = t.option_refs.clone()
    for k in range(2):
        old = t.option_refs[:, :, k]
        new = inv.gather(1, old.clamp(min=0))
        refs[:, :, k] = torch.where(old >= 0, new, old)
    return TorchObs(state=t.state, options=t.options, option_mask=t.option_mask, option_ids=t.option_ids,
                    option_refs=refs, entities=ent, entity_mask=mask, player=t.player)


def finite_part(logits, mask):
    return logits.masked_fill(~mask, 0.0)


def test_shapes_masking_and_no_illegal_mass():
    for make in (MEDIANO, lambda: CartasEnv()):
        torch.manual_seed(0)
        net = MiniPTCGNet(make().spec)
        for t in sample_obs(make, 40, 8, n_batches=3):
            logits, value = net(t)
            assert logits.shape == t.option_mask.shape and value.shape == (40,)
            probs = torch.softmax(logits, -1)
            assert (probs[~t.option_mask] == 0).all() and torch.allclose(probs.sum(-1), torch.ones(40))
            a = torch.multinomial(probs, 1).squeeze(1)
            assert t.option_mask.gather(1, a[:, None]).all()


def test_permutation_invariance_1000_observations():
    worst = 0.0
    n = 0
    for make in (MEDIANO, lambda: CartasEnv()):
        torch.manual_seed(1)
        net = MiniPTCGNet(make().spec).eval()
        for t in sample_obs(make, 50, 5, seed=3, n_batches=10):
            B, E = t.entity_mask.shape
            perm = torch.stack([torch.randperm(E) for _ in range(B)])
            with torch.no_grad():
                a, va = net(t)
                b, vb = net(permute_entities(t, perm))
            worst = max(worst, float((finite_part(a, t.option_mask) - finite_part(b, t.option_mask)).abs().max()),
                        float((va - vb).abs().max()))
            n += B
    assert n == 1000 and worst <= 1e-5, worst


def test_padding_rows_are_invisible():
    torch.manual_seed(2)
    net = MiniPTCGNet(MEDIANO().spec).eval()
    t = sample_obs(MEDIANO, 30, 6)[0]
    with torch.no_grad():
        a, va = net(t)
        junk_e = torch.where(t.entity_mask[:, :, None], t.entities, torch.randn_like(t.entities) * 5)
        junk_o = torch.where(t.option_mask[:, :, None], t.options, torch.randn_like(t.options) * 5)
        t2 = TorchObs(**{**t.__dict__, "entities": junk_e, "options": junk_o})
        b, vb = net(t2)
    assert (~t.entity_mask).any() and (~t.option_mask).any()
    assert torch.allclose(finite_part(a, t.option_mask), finite_part(b, t.option_mask), atol=1e-6)
    assert torch.allclose(va, vb, atol=1e-6)


def test_option_refs_matter():
    torch.manual_seed(3)
    net = MiniPTCGNet(MEDIANO().spec).eval()
    t = sample_obs(MEDIANO, 30, 6)[0]
    no_refs = TorchObs(**{**t.__dict__, "option_refs": -torch.ones_like(t.option_refs)})
    with torch.no_grad():
        a, _ = net(t)
        b, _ = net(no_refs)
    assert not torch.allclose(finite_part(a, t.option_mask), finite_part(b, t.option_mask), atol=1e-4)


def test_parameter_count_within_budget_for_every_env():
    from rlf.envs.bandit import Bandit
    for make in (lambda: DespachoEnv.from_size("diminuto"), MEDIANO, lambda: DespachoEnv.from_size("grande"),
                 lambda: CartasEnv()):
        n = sum(p.numel() for p in MiniPTCGNet(make().spec).parameters())
        assert 20_000 < n <= 150_000, n


def test_same_weights_run_on_a_bigger_env_without_retraining():
    torch.manual_seed(0)
    net = MiniPTCGNet(DespachoEnv.from_size("mediano").spec).eval()
    t = sample_obs(lambda: DespachoEnv.from_size("grande"), 8, 5)[0]
    with torch.no_grad():
        logits, value = net(t)
    assert logits.shape == (8, 73) and torch.isfinite(logits[t.option_mask]).all()
