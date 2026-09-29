import numpy as np
import pytest
import torch

from rlf.core import VecEnv, to_torch
from rlf.envs.cartas import CartasEnv
from rlf.envs.despacho import DespachoEnv
from rlf.nets.mlp import MLP, MLPActorCritic, option_logits_from_ids

CPU = torch.device("cpu")


def batch(make, n=16, steps=5, seed=0):
    vec = VecEnv(make, n, seed=seed)
    obs = vec.reset()
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        obs = vec.step(rng.integers(0, obs.n_options)).obs
    return vec, obs


def test_mlp_shapes_and_depth():
    net = MLP(5, (8, 8), 3)
    assert net(torch.zeros(4, 5)).shape == (4, 3)
    assert sum(isinstance(m, torch.nn.Linear) for m in net.modules()) == 3


def test_option_logits_from_ids_gathers_and_masks():
    vec, obs = batch(lambda: DespachoEnv.from_size("mediano"))
    t = to_torch(obs, CPU)
    full = torch.arange(vec.spec.n_actions, dtype=torch.float32).repeat(len(obs), 1)      # logit of id a is a
    out = option_logits_from_ids(full, t)
    assert out.shape == (len(obs), vec.spec.max_options)
    legal = t.option_mask
    assert torch.equal(out[legal], t.option_ids[legal].float())
    assert torch.isinf(out[~legal]).all() and (out[~legal] < 0).all()


@pytest.mark.parametrize("make", [lambda: DespachoEnv.from_size("mediano"), lambda: CartasEnv()])
def test_untrained_actor_critic_never_puts_mass_on_illegal_options(make):
    vec, obs = batch(make, n=64, steps=3)
    net = MLPActorCritic(vec.spec, hidden=(32, 32)).to(CPU)
    violations = 0
    for step in range(30):
        t = to_torch(obs, CPU)
        logits, value = net(t)
        assert logits.shape == (64, vec.spec.max_options) and value.shape == (64,)
        probs = torch.softmax(logits, dim=-1)
        assert (probs[~t.option_mask] == 0).all()                                  # exactly 0
        assert torch.allclose(probs.sum(-1), torch.ones(64))
        a = torch.distributions.Categorical(logits=logits).sample()
        violations += int((~t.option_mask.gather(1, a[:, None])).sum())
        assert (a.numpy() < obs.n_options).all()
        obs = vec.step(a.numpy()).obs
    assert violations == 0


def test_actor_critic_uses_flattened_state_and_entities():
    spec = DespachoEnv.from_size("diminuto").spec
    net = MLPActorCritic(spec, hidden=(16,))
    first = next(m for m in net.modules() if isinstance(m, torch.nn.Linear))
    assert first.in_features == spec.state_dim + spec.max_entities * spec.entity_dim
