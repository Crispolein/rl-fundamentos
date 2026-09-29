import numpy as np

from rlf.agents.baselines import RandomPolicy
from rlf.core import VecEnv
from tests.toy_env import ToyEnv


def batch(n=200, seed=0):
    vec = VecEnv(lambda: ToyEnv(), n, seed=seed)
    return vec, vec.reset()


def test_random_policy_only_legal_indices():
    vec, obs = batch()
    pol = RandomPolicy(0)
    for _ in range(20):
        a = pol.act(obs)
        assert a.dtype == np.int64 and a.shape == (200,)
        assert (a >= 0).all() and (a < obs.n_options).all()
        obs = vec.step(a).obs


def test_random_policy_deterministic_by_seed():
    _, obs = batch()
    assert np.array_equal(RandomPolicy(5).act(obs), RandomPolicy(5).act(obs))
    assert not np.array_equal(RandomPolicy(5).act(obs), RandomPolicy(6).act(obs))


def test_random_policy_is_uniform():
    _, obs = batch(1)
    n = int(obs.n_options[0])
    big = obs.select(np.zeros(20000, dtype=np.int64))
    a = RandomPolicy(1).act(big)
    freq = np.bincount(a, minlength=n) / len(a)
    assert np.allclose(freq, 1 / n, atol=0.02)
