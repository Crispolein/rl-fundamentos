import numpy as np
import pytest

from rlf.agents.tabular import value_iteration
from rlf.core import EVAL_SEED_BASE
from rlf.envs.despacho import MDP, DespachoConfig, DespachoEnv, build_mdp
from rlf.eval.stats import mean_ci, paired_diff_ci


def tiny_mdp():
    """s0 --a0--> s1 (r=1) or s2 (r=0) w.p. 0.5; s0 --a1--> s2 (r=0.4); s1 --a0--> s2 (r=1); s2 terminal."""
    legal = np.array([[1, 1], [1, 0], [0, 0]], bool)
    R = np.array([[0.5, 0.4], [1.0, 0.0], [0.0, 0.0]])
    # a0 from s0: reward E = 0.5*1 + 0.5*0 (folded into R); transitions to s1 / s2
    src, act, dst, prob = [0, 0, 0, 1], [0, 0, 1, 0], [1, 2, 2, 2], [0.5, 0.5, 1.0, 1.0]
    return MDP(keys=[("s0",), ("s1",), ("s2",)], index={("s0",): 0, ("s1",): 1, ("s2",): 2}, nA=2, legal=legal, R=R,
               src=np.array(src), act=np.array(act), dst=np.array(dst), prob=np.array(prob, float),
               terminal=np.array([False, False, True]), p0=np.array([1.0, 0, 0]))


def test_value_iteration_on_hand_mdp():
    V, Q, pi = value_iteration(tiny_mdp(), gamma=1.0)
    assert V[2] == 0 and V[1] == pytest.approx(1.0)
    assert Q[0, 0] == pytest.approx(0.5 + 0.5 * 1.0) and Q[0, 1] == pytest.approx(0.4)
    assert V[0] == pytest.approx(1.0) and pi.tolist() == [0, 0, -1]
    assert Q.dtype == np.float64 and pi.dtype == np.int64


def test_gamma_discounts():
    V1, _, _ = value_iteration(tiny_mdp(), gamma=1.0)
    V5, Q5, _ = value_iteration(tiny_mdp(), gamma=0.5)
    assert V5[0] == pytest.approx(0.5 + 0.5 * 0.5 * 1.0)
    assert V5[0] < V1[0]


@pytest.fixture(scope="module")
def solved():
    mdp = build_mdp(DespachoConfig.from_size("diminuto"))
    return mdp, value_iteration(mdp, gamma=1.0)


def test_bellman_residual_on_diminuto(solved):
    mdp, (V, Q, pi) = solved
    EV = np.zeros_like(Q)
    np.add.at(EV, (mdp.src, mdp.act), mdp.prob * V[mdp.dst])
    resid = np.abs(np.where(mdp.legal, Q - (mdp.R + EV), 0.0)).max()
    assert resid < 1e-9
    assert np.allclose(V[~mdp.terminal], np.where(mdp.legal, Q, -np.inf).max(1)[~mdp.terminal])
    assert (V[mdp.terminal] == 0).all() and (pi[mdp.terminal] == -1).all()
    assert mdp.legal[np.flatnonzero(~mdp.terminal), pi[~mdp.terminal]].all()


def optimal_returns(mdp, pi, n):
    env = DespachoEnv.from_size("diminuto")
    out = np.zeros(n)
    for i in range(n):
        obs = env.reset(EVAL_SEED_BASE + i)
        done = False
        while not done:
            a = pi[mdp.index[env.state_key()]]
            j = int(np.flatnonzero(obs.option_ids == a)[0])
            st = env.step(j)
            out[i] += st.rewards[0]
            obs, done = st.obs, st.done
    return out


def test_optimal_policy_matches_simulation_and_beats_greedy(solved):
    mdp, (V, Q, pi) = solved
    v_star = float(mdp.p0 @ V)
    assert v_star < 0
    rets = optimal_returns(mdp, pi, 2000)
    se = rets.std(ddof=1) / np.sqrt(2000)
    assert abs(rets.mean() - v_star) <= 3 * se
    from rlf.agents.baselines import GreedyDespacho
    from rlf.eval.arena import evaluate_policy
    greedy = evaluate_policy(lambda: DespachoEnv.from_size("diminuto"), GreedyDespacho(), 2000)
    assert paired_diff_ci(rets, greedy).lo > 0
