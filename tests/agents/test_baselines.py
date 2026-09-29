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


# ---- HeuristicCartas ---------------------------------------------------------------------------
from rlf.agents.baselines import HeuristicCartas  # noqa: E402
from rlf.core import BatchObs  # noqa: E402
from rlf.envs.cartas import (ATTACH, ATTACK, END_TURN, PLACE_ACTIVE, PLAY_CREATURE, PLAY_DRAW,  # noqa: E402
                             PLAY_HEAL, PROMOTE, CartasEnv)
from rlf.eval.arena import play_games  # noqa: E402
from rlf.eval.stats import passes_lower  # noqa: E402
from tests.envs.test_cartas import mk  # noqa: E402


def chosen_id(core, pol=None):
    env = CartasEnv()
    env.reset(0)
    env.core = core
    obs = env._obs()
    a = (pol or HeuristicCartas()).act(BatchObs.from_obs(obs, env.spec))[0]
    return int(obs.option_ids[a])


def test_heuristic_setup_picks_highest_damage_creature():
    c = mk()
    c.phase, c.player = "SETUP", 0
    c.hands[0] = [0, 1, 3, 4]
    assert chosen_id(c) == PLACE_ACTIVE + 3


def test_heuristic_promote_prefers_more_energy_then_hp():
    c = mk(board0=[None, [0, 3, 1], [2, 8, 1]])
    c.phase, c.player = "PROMOTE", 0
    assert chosen_id(c) == PROMOTE + 1               # same energy, higher HP
    c = mk(board0=[None, [0, 3, 2], [2, 8, 1]])
    c.phase, c.player = "PROMOTE", 0
    assert chosen_id(c) == PROMOTE + 0


def test_heuristic_main_priority_order():
    full = dict(hand0=[4, 5, 3, 6], board0=[[0, 1, 1], None, None])
    assert chosen_id(mk(**full)) == ATTACH + 0                                   # 1) attach to active
    c = mk(**full); c.attached = True
    assert chosen_id(c) == PLAY_DRAW                                             # 2) draw
    c = mk(**full); c.attached = True; c.support = True
    assert chosen_id(c) == PLAY_CREATURE + 3                                     # 3) creature
    c = mk(hand0=[6], board0=[[3, 2, 0], [0, 1, 0], None])
    assert chosen_id(c) == PLAY_HEAL + 0                                         # most damaged (4 vs 2)
    c = mk(hand0=[6], board0=[[0, 2, 0], [3, 2, 0], None])
    assert chosen_id(c) == PLAY_HEAL + 1                                         # most damaged (1 vs 4)
    c = mk(hand0=[], board0=[[0, 3, 1], None, None])
    assert chosen_id(c) == ATTACK                                                # 5) attack
    c = mk(hand0=[], board0=[[1, 5, 0], None, None])
    assert chosen_id(c) == END_TURN                                              # 6) end turn


def test_heuristic_is_deterministic():
    env = CartasEnv()
    obs = env.reset(2)
    b = BatchObs.from_obs(obs, env.spec)
    assert np.array_equal(HeuristicCartas().act(b), HeuristicCartas().act(b))


def test_heuristic_beats_random_mirror_aggro():
    r = play_games(lambda: CartasEnv(), HeuristicCartas(), RandomPolicy(0), 1000)
    assert r.n == 1000 and passes_lower(r.win_rate_a, 0.80), r.win_rate_a
