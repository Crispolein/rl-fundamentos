import numpy as np
import pytest

from rlf.agents.baselines import HeuristicCartas, RandomPolicy
from rlf.core import EVAL_SEED_BASE
from rlf.envs.cartas import CartasEnv
from rlf.envs.wrappers import VsOpponent


def make(opp=None):
    return VsOpponent(lambda: CartasEnv(), opp or RandomPolicy(0))


def test_spec_is_single_player_view():
    env = make()
    sp = env.spec
    assert sp.n_players == 1 and sp.max_options == 13 and sp.state_dim == 12
    obs = env.reset(3)
    assert obs.player == 0 and obs.n_options >= 1


def test_learner_always_decides_and_reward_is_its_own():
    env = make()
    rng = np.random.default_rng(0)
    wins = 0
    n = 300
    for i in range(n):
        obs = env.reset(EVAL_SEED_BASE + i)
        done = False
        while not done:
            assert env.env.core.player == env.seat
            st = env.step(int(rng.integers(obs.n_options)))
            obs, done = st.obs, st.done
        assert st.rewards.shape == (1,) and abs(st.rewards[0]) == 1.0
        wins += st.rewards[0] > 0
    assert 0.3 < wins / n < 0.7            # random vs random, seats alternate


def test_seats_alternate_with_seed_parity_and_deterministic():
    env = make()
    env.reset(10)
    assert env.seat == 0
    env.reset(11)
    assert env.seat == 1

    def run(seed):
        e = make()
        obs = e.reset(seed)
        rng = np.random.default_rng(5)
        out = []
        done = False
        while not done:
            st = e.step(int(rng.integers(obs.n_options)))
            out.append((st.rewards[0], st.obs.state.tobytes()))
            obs, done = st.obs, st.done
        return out
    assert run(21) == run(21)


def test_strong_opponent_beats_a_random_learner():
    env = make(HeuristicCartas())
    rng = np.random.default_rng(0)
    wins = 0
    for i in range(200):
        obs = env.reset(EVAL_SEED_BASE + i)
        done = False
        while not done:
            st = env.step(int(rng.integers(obs.n_options)))
            obs, done = st.obs, st.done
        wins += st.rewards[0] > 0
    assert wins / 200 < 0.2
