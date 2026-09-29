import numpy as np
import pytest

from rlf.agents.baselines import RandomPolicy
from rlf.core import EVAL_SEED_BASE
from rlf.eval.arena import elo_ratings, evaluate_policy, play_games, round_robin
from tests.toy_env import ToyDuel, ToyEnv


class Last:
    """Deterministic: always the last legal option."""
    def act(self, obs):
        return (obs.n_options - 1).astype(np.int64)


class Fixed:
    def __init__(self, k):
        self.k = k

    def act(self, obs):
        return np.full(len(obs), self.k, np.int64)


def test_evaluate_policy_matches_manual_rollouts_by_seed():
    make = lambda: ToyEnv(1)
    res = evaluate_policy(make, Last(), 37, n_envs=10)
    assert res.shape == (37,) and res.dtype == np.float64
    for i in (0, 1, 9, 10, 36):
        env = make()
        obs = env.reset(EVAL_SEED_BASE + i)
        ret, done = 0.0, False
        while not done:
            st = env.step(obs.n_options - 1)
            ret += st.rewards[0]
            obs, done = st.obs, st.done
        assert res[i] == pytest.approx(ret)


def test_evaluate_policy_independent_of_n_envs_for_deterministic_policy():
    make = lambda: ToyEnv(1)
    assert np.array_equal(evaluate_policy(make, Last(), 40, n_envs=5),
                          evaluate_policy(make, Last(), 40, n_envs=20))


def test_play_games_deterministic():
    make = lambda: ToyDuel()
    r1 = play_games(make, RandomPolicy(1), RandomPolicy(2), 200, n_envs=20)
    r2 = play_games(make, RandomPolicy(1), RandomPolicy(2), 200, n_envs=20)
    assert r1 == r2 and r1.n == 200 and r1.wins_a + r1.wins_b == 200
    assert r1.win_rate_a.mean == r1.wins_a / 200


def test_seat_symmetry_identical_policies_split_exactly():
    make = lambda: ToyDuel(seat_bias=1.5)
    r = play_games(make, Fixed(1), Fixed(1), 400, n_envs=20)
    assert r.wins_a == 200 and r.wins_b == 200 and r.seat0_wins > 240


def test_seat_bias_shows_in_seat0_wins_but_not_in_win_rate():
    make = lambda: ToyDuel(seat_bias=1.5)
    r = play_games(make, RandomPolicy(1), RandomPolicy(2), 2000, n_envs=100)
    assert r.seat0_wins / r.n > 0.65
    assert abs(r.win_rate_a.mean - 0.5) < 0.05


def test_play_games_requires_even():
    with pytest.raises(ValueError):
        play_games(lambda: ToyDuel(), Fixed(0), Fixed(1), 11)


def test_round_robin_deterministic_and_shape():
    make = lambda: ToyDuel()
    ent = lambda: {"a": Fixed(0), "b": Fixed(1), "c": Fixed(2)}
    m1 = round_robin(make, ent(), 200, n_envs=20)
    m2 = round_robin(make, ent(), 200, n_envs=20)
    assert m1.shape == (3, 3) and np.array_equal(m1, m2, equal_nan=True)
    assert np.isnan(np.diag(m1)).all()
    off = ~np.eye(3, dtype=bool)
    assert np.allclose((m1 + m1.T)[off], 1.0)
    assert m1[2, 0] > 0.65 and m1[0, 2] < 0.35


def test_elo_recovers_true_ratings():
    make = lambda: ToyDuel()
    m = round_robin(make, {"a": Fixed(0), "b": Fixed(1), "c": Fixed(2)}, 1000, n_envs=100)
    r = elo_ratings(m, 1000, anchor=0, anchor_rating=1000.0)
    assert r[0] == pytest.approx(1000.0)
    assert abs(r[1] - 1100) <= 40 and abs(r[2] - 1200) <= 40


def test_elo_exact_on_a_consistent_matrix():
    p = lambda d: 1 / (1 + 10 ** (-d / 400))
    r_true = np.array([1000.0, 1150.0, 1300.0])
    m = np.array([[np.nan if i == j else p(r_true[i] - r_true[j]) for j in range(3)] for i in range(3)])
    r = elo_ratings(m, 1000)
    assert np.allclose(r, r_true, atol=1e-3)
