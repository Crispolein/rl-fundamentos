import numpy as np
import pytest

from rlf.envs.bandit import Bandit


def test_B1_means_are_standard_normal_and_fixed_in_episode():
    all_mu = []
    for seed in range(2000):
        env = Bandit(k=10)
        env.reset(seed)
        all_mu.append(env.mu.copy())
    all_mu = np.array(all_mu)
    assert all_mu.shape == (2000, 10)
    assert abs(all_mu.mean()) < 0.03 and abs(all_mu.std() - 1) < 0.03
    assert abs(np.corrcoef(all_mu[:, 0], all_mu[:, 1])[0, 1]) < 0.06      # independent arms
    env = Bandit(k=10, horizon=20)
    env.reset(5)
    before = env.mu.copy()
    for _ in range(20):
        env.step(3)
    assert np.array_equal(env.mu, before)


def test_B2_reward_is_normal_around_the_arm_mean():
    env = Bandit(k=5, horizon=5000, noise_std=2.0)
    env.reset(1)
    r = np.array([env.step(2).rewards[0] for _ in range(5000)])
    assert abs(r.mean() - env.mu[2]) < 4 * 2.0 / np.sqrt(5000)
    assert abs(r.std() - 2.0) < 0.1


def test_B3_all_options_always_legal():
    env = Bandit(k=7, horizon=10)
    obs = env.reset(0)
    while True:
        assert obs.n_options == 7 and np.array_equal(obs.option_ids, np.arange(7))
        assert np.array_equal(obs.options, np.eye(7, dtype=np.float32))
        assert (obs.option_refs == -1).all()
        st = env.step(0)
        if st.done:
            break
        obs = st.obs


def test_B4_lasts_exactly_horizon_steps():
    env = Bandit(k=4, horizon=17)
    env.reset(0)
    for t in range(1, 18):
        st = env.step(1)
        assert st.done == (t == 17)
    assert st.obs.n_options == 0


def test_B5_info_has_mu_chosen_and_mu_best():
    env = Bandit(k=6, horizon=5)
    env.reset(3)
    st = env.step(4)
    assert st.info["mu_chosen"] == env.mu[4] and st.info["mu_best"] == env.mu.max()


def test_spec_and_state_key():
    env = Bandit()
    sp = env.spec
    assert (sp.n_players, sp.state_dim, sp.option_dim, sp.entity_dim) == (1, 1, 10, 1)
    assert (sp.max_options, sp.max_entities, sp.n_actions, sp.max_steps) == (10, 0, 10, 300)
    obs = env.reset(0)
    assert obs.entities.shape == (0, 1) and obs.state[0] == 0.0
    assert env.state_key() == (0,)
    env.step(0)
    assert env.state_key() == (1,)
