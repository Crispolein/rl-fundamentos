import numpy as np
import pytest
import torch

from rlf.agents.baselines import RandomPolicy
from rlf.agents.policy import ActorCriticPolicy
from rlf.agents.ppo import PPO, PPOConfig, compute_gae
from rlf.core import VecEnv
from rlf.envs.despacho import DespachoEnv
from rlf.eval.arena import evaluate_policy
from rlf.eval.stats import paired_diff_ci
from rlf.nets.mlp import MLPActorCritic

make = lambda: DespachoEnv.from_size("diminuto")


def test_gae_hand_computed():
    # 3 steps, one env; gamma = 0.9, lambda = 0.8, no episode ends
    r = np.array([[1.0], [2.0], [3.0]])
    v = np.array([[0.5], [1.0], [1.5]])
    d = np.zeros((3, 1), bool)
    last = np.array([2.0])
    adv, ret = compute_gae(r, v, d, last, gamma=0.9, lam=0.8)
    d2 = 3.0 + 0.9 * 2.0 - 1.5                         # delta_2 = 3.3
    d1 = 2.0 + 0.9 * 1.5 - 1.0                         # delta_1 = 2.35
    d0 = 1.0 + 0.9 * 1.0 - 0.5                         # delta_0 = 1.4
    a2 = d2
    a1 = d1 + 0.9 * 0.8 * a2
    a0 = d0 + 0.9 * 0.8 * a1
    assert adv[:, 0] == pytest.approx([a0, a1, a2])
    assert ret[:, 0] == pytest.approx(adv[:, 0] + v[:, 0])
    assert (a0, a1, a2) == pytest.approx((1.4 + 0.72 * (2.35 + 0.72 * 3.3), 2.35 + 0.72 * 3.3, 3.3))


def test_gae_cuts_at_episode_end():
    r = np.array([[1.0], [2.0], [3.0]])
    v = np.array([[0.5], [1.0], [1.5]])
    d = np.array([[False], [True], [False]])           # the transition at t=1 ended an episode
    adv, _ = compute_gae(r, v, d, np.array([2.0]), gamma=0.9, lam=0.8)
    a2 = 3.0 + 0.9 * 2.0 - 1.5
    a1 = 2.0 - 1.0                                     # terminal: no bootstrap, no propagation from t=2
    a0 = (1.0 + 0.9 * 1.0 - 0.5) + 0.72 * a1
    assert adv[:, 0] == pytest.approx([a0, a1, a2])


def test_gae_lambda_extremes():
    rng = np.random.default_rng(0)
    r, v = rng.normal(size=(6, 2)), rng.normal(size=(6, 2))
    d = np.zeros((6, 2), bool)
    last = rng.normal(size=2)
    a0, _ = compute_gae(r, v, d, last, 0.9, 0.0)                      # lambda = 0: one-step TD error
    v_next = np.vstack([v[1:], last[None]])
    assert np.allclose(a0, r + 0.9 * v_next - v)
    a1, ret = compute_gae(r, v, d, last, 0.9, 1.0)                    # lambda = 1: Monte-Carlo return - V
    mc = np.zeros_like(r)
    g = last.copy()
    for t in reversed(range(6)):
        g = r[t] + 0.9 * g
        mc[t] = g
    assert np.allclose(ret, mc)


def test_ppo_clip_limits_the_update():
    torch.manual_seed(0)
    net = MLPActorCritic(make().spec, hidden=(32, 32))
    ppo = PPO(net, PPOConfig(n_steps=32, epochs=4, minibatches=2, clip_eps=0.1), seed=0)
    vec = VecEnv(make, 8, seed=0)
    batch = ppo.collect(vec)
    info = ppo.update(batch)
    assert 0.0 <= info["clip_frac"] <= 1.0 and abs(info["approx_kl"]) < 0.1
    torch.manual_seed(0)
    net2 = MLPActorCritic(make().spec, hidden=(32, 32))
    free = PPO(net2, PPOConfig(n_steps=32, epochs=4, minibatches=2, clip_eps=float("inf")), seed=0)
    assert free.update(free.collect(VecEnv(make, 8, seed=0)))["clip_frac"] == 0.0
    assert np.isfinite(info["policy_loss"]) and np.isfinite(info["value_loss"])


def test_ppo_collect_shapes_and_legal_actions():
    net = MLPActorCritic(make().spec, hidden=(16,))
    ppo = PPO(net, PPOConfig(n_steps=16), seed=0)
    b = ppo.collect(VecEnv(make, 4, seed=0))
    assert b["actions"].shape == (16, 4) and b["advantages"].shape == (16, 4)
    assert (b["actions"] < b["obs"]["option_mask"].sum(-1)).all()


def test_ppo_learns_smoke():
    torch.manual_seed(0)
    net = MLPActorCritic(make().spec, hidden=(64, 64))
    ppo = PPO(net, PPOConfig(reward_scale=0.1), seed=0)
    ppo.train(VecEnv(make, 8, seed=0), 20_000)
    learned = evaluate_policy(make, ActorCriticPolicy(net, greedy=False), 1000)
    rand = evaluate_policy(make, RandomPolicy(0), 1000)
    assert paired_diff_ci(learned, rand).lo > 0


# ---- T30: self-play -----------------------------------------------------------------------------
from rlf.agents.ppo import SelfPlay, train_selfplay  # noqa: E402
from rlf.envs.cartas import CartasEnv  # noqa: E402
from rlf.eval.arena import play_games  # noqa: E402
from rlf.nets.pointer import PointerActorCritic  # noqa: E402


def test_selfplay_batch_is_consistent():
    torch.manual_seed(0)
    net = PointerActorCritic(CartasEnv().spec, d_model=32)
    sp = SelfPlay(net, PPOConfig(gamma=1.0, lam=0.95), lambda: CartasEnv(), n_envs=8, seed=0)
    b = sp.collect(8)
    n = b["actions"].shape[0]
    assert b["actions"].shape == (n, 1) and b["obs"]["state"].shape[:2] == (n, 1)
    assert np.isfinite(b["advantages"]).all() and np.isfinite(b["returns"]).all()
    assert (b["actions"][:, 0] < b["obs"]["option_mask"][:, 0].sum(-1)).all()          # legal indices only
    # returns = advantages + values, and with gamma = lambda = 1 the return of every stream's first step is +-1
    assert np.allclose(b["returns"], b["advantages"] + b["values"])


def test_selfplay_terminal_rewards_are_opposite_in_pure_self_play():
    torch.manual_seed(0)
    net = PointerActorCritic(CartasEnv().spec, d_model=32)
    sp = SelfPlay(net, PPOConfig(gamma=1.0, lam=1.0), lambda: CartasEnv(), n_envs=4, p_self=1.0, seed=1)
    b = sp.collect(20)
    # with lam = 1 and gamma = 1 each stream's return-to-go is its own final reward: every value is +1 or -1
    assert np.isin(np.round(b["returns"][:, 0], 6), [-1.0, 1.0]).all()
    assert abs(b["returns"].mean()) < 0.2                                                # zero-sum: both seats trained


def test_selfplay_pool_keeps_recent_snapshots():
    torch.manual_seed(0)
    net = PointerActorCritic(CartasEnv().spec, d_model=32)
    sp = SelfPlay(net, PPOConfig(), lambda: CartasEnv(), n_envs=8, pool_size=2, snapshot_every=1, seed=0)
    sp.train(3000, games_per_iter=8)
    assert 1 <= len(sp.pool) <= 2 and sp.games_played >= 8


def test_selfplay_learns_to_beat_random():
    torch.manual_seed(0)
    net = PointerActorCritic(CartasEnv().spec, d_model=64)
    sp = train_selfplay(net, PPOConfig(gamma=1.0, lam=0.95, lr=1e-3, epochs=4, minibatches=4), lambda: CartasEnv(),
                        total_steps=60_000, n_envs=32, seed=0)
    res = play_games(lambda: CartasEnv(), ActorCriticPolicy(net, greedy=True), RandomPolicy(0), 1000)
    assert res.win_rate_a.lo > 0.6, res.win_rate_a
