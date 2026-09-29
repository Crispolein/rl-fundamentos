import numpy as np
import pytest
import torch

from rlf.agents.baselines import RandomPolicy
from rlf.agents.policy import ActorCriticPolicy
from rlf.agents.reinforce import Episode, Reinforce, returns_to_go
from rlf.core import VecEnv
from rlf.envs.despacho import DespachoEnv
from rlf.eval.arena import evaluate_policy
from rlf.eval.stats import paired_diff_ci
from rlf.nets.mlp import MLPActorCritic

make = lambda: DespachoEnv.from_size("diminuto")


def new_agent(baseline, seed=0, **kw):
    torch.manual_seed(seed)
    net = MLPActorCritic(make().spec, hidden=(32, 32))
    return Reinforce(net, baseline=baseline, seed=seed, **kw)


def test_returns_to_go_by_hand():
    r = np.array([1.0, 2.0, 3.0])
    assert np.allclose(returns_to_go(r, 1.0), [6.0, 5.0, 3.0])
    assert np.allclose(returns_to_go(r, 0.5), [1 + 0.5 * (2 + 0.5 * 3), 2 + 0.5 * 3, 3.0])


def test_collect_returns_complete_on_policy_episodes():
    agent = new_agent("value")
    vec = VecEnv(make, 4, seed=0)
    eps = agent.collect(vec, 12)
    assert len(eps) >= 12
    for e in eps:
        assert isinstance(e, Episode) and len(e.actions) == len(e.rewards) == len(e.returns)
        assert np.isclose(e.returns[0], e.rewards.sum())              # gamma = 1: G_0 is the episode return
        assert (e.actions < e.obs["option_mask"].sum(1)).all()        # legal indices
    eps2 = agent.collect(vec, 12)
    assert len(eps2) >= 12


def test_loss_matches_formula_by_hand_for_each_baseline():
    agent = new_agent("none")
    vec = VecEnv(make, 2, seed=1)
    eps = agent.collect(vec, 4)
    obs, a, g = agent._stack(eps)
    logits, value = agent.net(obs)
    logp = torch.log_softmax(logits, -1).gather(1, a[:, None]).squeeze(1)
    expected = {"none": -(logp * g).mean(), "mean": -(logp * (g - g.mean())).mean(),
                "value": -(logp * (g - value.detach())).mean()}
    for baseline, want in expected.items():
        ag = new_agent(baseline)
        ag.net.load_state_dict(agent.net.state_dict())
        _, info = ag.loss(eps)
        assert info["policy_loss"] == pytest.approx(float(want), rel=1e-5, abs=1e-6)


def test_episode_gradient_shape_and_baseline_effect():
    agent = new_agent("none")
    vec = VecEnv(make, 2, seed=1)
    ep = agent.collect(vec, 2)[0]
    n_params = sum(p.numel() for p in agent.net.parameters())
    g0 = agent.episode_gradient(ep)
    g1 = agent.episode_gradient(ep, baseline_value=float(ep.returns.mean()))
    assert g0.shape == (n_params,) and g1.shape == (n_params,)
    assert not torch.allclose(g0, g1)


def test_reinforce_value_baseline_learns_smoke():
    agent = new_agent("value", lr=3e-3, reward_scale=0.1)
    agent.train(VecEnv(make, 8, seed=0), 150, episodes_per_update=16)
    pol = ActorCriticPolicy(agent.net, greedy=True)
    learned = evaluate_policy(make, pol, 1000)
    rand = evaluate_policy(make, RandomPolicy(0), 1000)
    assert paired_diff_ci(learned, rand).lo > 0
