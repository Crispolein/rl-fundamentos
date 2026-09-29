import numpy as np
import pytest
import torch

from rlf.core import VecEnv, to_torch
from rlf.envs.cartas import CartasEnv
from rlf.envs.despacho import DespachoEnv
from rlf.nets.pointer import PointerActorCritic

CPU = torch.device("cpu")
SIZES = ("diminuto", "mediano", "grande")


def rollout_batches(make, n_envs=32, steps=25, seed=0):
    vec = VecEnv(make, n_envs, seed=seed)
    obs = vec.reset()
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        yield obs
        obs = vec.step(rng.integers(0, obs.n_options)).obs


@pytest.mark.parametrize("make", [lambda: DespachoEnv.from_size("mediano"), lambda: CartasEnv()])
def test_untrained_pointer_never_puts_mass_on_illegal_options(make):
    spec = make().spec
    torch.manual_seed(0)
    net = PointerActorCritic(spec, d_model=32)
    n_obs = violations = 0
    for obs in rollout_batches(make, 40, 30):
        t = to_torch(obs, CPU)
        logits, value = net(t)
        assert logits.shape == (40, spec.max_options) and value.shape == (40,)
        probs = torch.softmax(logits, -1)
        assert (probs[~t.option_mask] == 0).all()
        assert torch.allclose(probs.sum(-1), torch.ones(40))
        a = torch.distributions.Categorical(logits=logits).sample()
        violations += int((a.numpy() >= obs.n_options).sum())
        n_obs += 40
    assert n_obs >= 1000 and violations == 0


def test_same_network_runs_on_every_despacho_size():
    torch.manual_seed(0)
    spec_small = DespachoEnv.from_size("diminuto").spec
    net = PointerActorCritic(spec_small, d_model=32)
    n_params = sum(p.numel() for p in net.parameters())
    for size in SIZES:
        make = lambda size=size: DespachoEnv.from_size(size)
        obs = next(rollout_batches(make, 8, 1))
        logits, value = net(to_torch(obs, CPU))
        assert logits.shape == (8, make().spec.max_options)
        assert torch.isfinite(logits[to_torch(obs, CPU).option_mask]).all()
    assert n_params == sum(p.numel() for p in net.parameters())


def test_option_order_equivariance():
    make = lambda: DespachoEnv.from_size("mediano")
    torch.manual_seed(1)
    net = PointerActorCritic(make().spec, d_model=32)
    obs = list(rollout_batches(make, 16, 12))[-1]
    t = to_torch(obs, CPU)
    logits, _ = net(t)
    perm = torch.stack([torch.randperm(t.options.shape[1]) for _ in range(len(obs))])
    idx = perm[:, :, None]
    t2 = type(t)(state=t.state, options=t.options.gather(1, idx.expand(-1, -1, t.options.shape[2])),
                 option_mask=t.option_mask.gather(1, perm), option_ids=t.option_ids.gather(1, perm),
                 option_refs=t.option_refs, entities=t.entities, entity_mask=t.entity_mask, player=t.player)
    logits2, _ = net(t2)
    fix = lambda x: torch.nan_to_num(x, neginf=-1e9)
    assert torch.allclose(fix(logits.gather(1, perm)), fix(logits2), atol=1e-5)


def test_bandit_without_entities_works():
    from rlf.envs.bandit import Bandit
    make = lambda: Bandit(k=5, horizon=10)
    net = PointerActorCritic(make().spec, d_model=16)
    obs = next(rollout_batches(make, 4, 1))
    logits, value = net(to_torch(obs, CPU))
    assert logits.shape == (4, 5) and torch.isfinite(value).all()


def test_value_depends_on_state():
    make = lambda: DespachoEnv.from_size("mediano")
    torch.manual_seed(0)
    net = PointerActorCritic(make().spec, d_model=32)
    obs = list(rollout_batches(make, 8, 6))[-1]
    t = to_torch(obs, CPU)
    _, v1 = net(t)
    t2 = type(t)(**{**t.__dict__, "state": t.state + 1.0})
    _, v2 = net(t2)
    assert not torch.allclose(v1, v2)
