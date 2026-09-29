import numpy as np
import pytest
import torch

from rlf.agents.baselines import RandomPolicy
from rlf.agents.dqn import DQN, DQNConfig, ReplayBuffer
from rlf.core import VecEnv
from rlf.envs.despacho import DespachoEnv
from rlf.eval.arena import evaluate_policy
from rlf.eval.stats import paired_diff_ci


def test_replay_buffer_ring_and_sampling():
    buf = ReplayBuffer(capacity=5, obs_dim=2, n_actions=3, seed=0)
    for i in range(8):
        buf.add(np.full(2, i, np.float32), i % 3, float(i), np.full(2, i + 1, np.float32), i == 7,
                np.array([True, False, True]))
    assert len(buf) == 5
    s, a, r, s2, d, m = buf.sample(4)
    assert s.shape == (4, 2) and m.shape == (4, 3)
    assert set(r.tolist()) <= {3.0, 4.0, 5.0, 6.0, 7.0}                # oldest 3 entries were overwritten


def test_act_returns_only_legal_indices():
    spec = DespachoEnv.from_size("mediano").spec
    dqn = DQN(spec, DQNConfig(), seed=0)
    vec = VecEnv(lambda: DespachoEnv.from_size("mediano"), 32, seed=1)
    obs = vec.reset()
    for _ in range(40):
        a = dqn.act(obs, eps=0.3)
        assert a.dtype == np.int64 and (a >= 0).all() and (a < obs.n_options).all()
        obs = vec.step(a).obs


def test_td_target_uses_only_legal_next_actions():
    spec = DespachoEnv.from_size("diminuto").spec
    dqn = DQN(spec, DQNConfig(gamma=1.0, use_target=False), seed=0)
    for p in dqn.q_net.parameters():
        torch.nn.init.zeros_(p)
    last = [m for m in dqn.q_net.modules() if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        last.bias.copy_(torch.tensor([100.0, 1.0, 5.0]))               # id 0 is huge but illegal in s'
    s = torch.zeros(1, dqn.obs_dim)
    mask = torch.tensor([[False, True, True]])
    target = dqn.td_target(torch.tensor([-2.0]), s, torch.tensor([False]), mask)
    assert target.item() == pytest.approx(-2.0 + 5.0)
    assert dqn.td_target(torch.tensor([-2.0]), s, torch.tensor([True]), mask).item() == pytest.approx(-2.0)


def test_dqn_learns_smoke_on_diminuto():
    make = lambda: DespachoEnv.from_size("diminuto")
    dqn = DQN(make().spec, DQNConfig(), seed=0)
    dqn.train(VecEnv(make, 8, seed=0), 40_000)
    learned = evaluate_policy(make, dqn, 1000)
    rand = evaluate_policy(make, RandomPolicy(0), 1000)
    assert paired_diff_ci(learned, rand).lo > 0
