"""MiniDespacho rules D1-D14 (design.md section 4.2)."""
from dataclasses import replace

import numpy as np
import pytest

from rlf.envs.despacho import DespachoConfig, DespachoCore

SIZES = ("diminuto", "mediano", "grande")


def core(size="diminuto", **over):
    cfg = replace(DespachoConfig.from_size(size), **over)
    return DespachoCore(cfg)


def test_size_table():
    t = {n: DespachoConfig.from_size(n) for n in SIZES}
    assert [(c.width, c.height, c.n_trucks, c.capacity, c.max_pending, c.horizon) for c in t.values()] == [
        (3, 1, 2, 2, 1, 6), (5, 5, 3, 12, 4, 30), (8, 8, 6, 12, 12, 60)]
    assert [c.p_arrival for c in t.values()] == [0.5, 0.5, 0.8]
    assert [c.qty_range for c in t.values()] == [(1, 1), (1, 3), (1, 3)]
    assert [c.slack_range for c in t.values()] == [(2, 2), (4, 8), (4, 10)]
    assert all(c.dist_w == 1 and c.late_w == 2 for c in t.values())
    assert [c.U for c in t.values()] == [14, 68, 134]
    assert [c.max_steps for c in t.values()] == [12, 60, 120]
    assert [c.n_actions for c in t.values()] == [3, 13, 73]
    with pytest.raises(KeyError):
        DespachoConfig.from_size("enorme")


def test_D1_initial_state():
    c = core("mediano", p_arrival=0.0)
    c.reset(0)
    t, pend, trucks = c.state_key()
    assert t == 0 and pend == (None,) * 4
    assert trucks == ((0, 0, 12, 0),) * 3


def test_D1_tick0_arrivals_processed_at_reset():
    c = core("mediano", p_arrival=1.0)
    c.reset(0)
    assert sum(p is not None for p in c.state_key()[1]) == 1


def test_D2_arrival_frequency_and_fields():
    n, arrivals = 4000, 0
    locs, qtys, slacks = [], [], []
    for seed in range(n):
        c = core("mediano")
        c.reset(seed)
        for t in range(c.cfg.horizon):
            a = c.schedule[t]
            if a is not None:
                arrivals += 1
                (x, y), q, s = a
                locs.append((x, y)); qtys.append(q); slacks.append(s)
    total = n * 30
    assert abs(arrivals / total - 0.5) < 0.01
    assert set(locs) == {(x, y) for x in range(5) for y in range(5)}
    assert set(qtys) == {1, 2, 3} and set(slacks) == {4, 5, 6, 7, 8}
    assert abs(np.mean(qtys) - 2) < 0.03 and abs(np.mean(slacks) - 6) < 0.03


def test_D2_slot_lowest_index_and_due():
    c = core("mediano", p_arrival=1.0)
    c.reset(1)                     # tick 0 -> slot 0
    _, pend, _ = c.state_key()
    assert pend[0] is not None and pend[1:] == (None, None, None)
    assert pend[0][3] == c.schedule[0][2]           # due - t with t = 0
    c.step_action(c.wait_id)       # tick 1 -> slot 1
    assert [p is not None for p in c.state_key()[1]] == [True, True, False, False]


def test_D2_rejected_when_no_free_slot_costs_U():
    c = core("diminuto", p_arrival=1.0)
    c.reset(0)                     # slot 0 taken; M = 1
    r, done, _ = c.step_action(c.wait_id)
    assert not done and r == -c.cfg.U      # the tick-1 arrival is rejected


def test_D2_schedule_independent_of_decisions():
    c1, c2 = core("mediano"), core("mediano")
    c1.reset(3); c2.reset(3)
    rng = np.random.default_rng(0)
    for _ in range(10):
        c2.step_action(int(rng.choice(c2.legal_actions())))
    assert c1.schedule == c2.schedule


def test_D3_D4_D5_legality():
    c = core("mediano", p_arrival=0.0)
    c.reset(0)
    # slot 0 qty 3 pending; truck 0 busy for 2 ticks, truck 1 free with cap 2, truck 2 free cap 12
    c.set_state((0, ((1, 1, 3, 6), None, None, None), ((0, 0, 12, 2), (0, 0, 2, 0), (0, 0, 12, 0))))
    K, M = 3, 4
    assert c.legal_actions() == [0 * K + 2, c.wait_id]        # only (slot 0, truck 2) + wait
    assert c.wait_id == M * K
    c.set_state((0, (None,) * 4, ((0, 0, 12, 0),) * 3))
    assert c.legal_actions() == [c.wait_id]                   # D5: wait is always legal
    c.set_state((3, ((1, 1, 1, 6), None, None, None), ((0, 0, 12, 3), (0, 0, 2, 3), (0, 0, 12, 3))))
    assert c.legal_actions() == [c.wait_id]                   # nobody free


def test_D6_assign_effects_and_time_does_not_advance():
    c = core("diminuto", p_arrival=0.0)
    c.reset(0)
    c.set_state((0, ((2, 0, 1, 5),), ((0, 0, 2, 0), (0, 0, 2, 0))))
    r, done, _ = c.step_action(0 * 2 + 1)                     # slot 0, truck 1
    t, pend, trucks = c.state_key()
    assert t == 0 and not done and pend == (None,)
    assert trucks == ((0, 0, 2, 0), (2, 0, 1, 2))             # moved to (2,0); cap 1; busy 2 ticks
    assert r == -2


def test_D7_reward_distance_and_lateness():
    c = core("diminuto", p_arrival=0.0)
    c.reset(0)
    c.set_state((0, ((2, 0, 1, 2),), ((0, 0, 2, 0), (0, 0, 2, 0))))      # d=2, arrives at 2 = due
    assert c.step_action(0)[0] == -2
    c.set_state((0, ((2, 0, 1, 1),), ((0, 0, 2, 0), (0, 0, 2, 0))))      # 1 tick late
    assert c.step_action(0)[0] == -2 - 2 * 1
    c.set_state((0, ((2, 0, 1, -30),), ((0, 0, 2, 0), (0, 0, 2, 0))))    # hugely late: capped at T
    assert c.step_action(0)[0] == -2 - 2 * 6
    assert -c.cfg.U <= -2 - 2 * 6


def test_D8_wait_advances_time_and_reward_zero():
    c = core("mediano", p_arrival=0.0)
    c.reset(0)
    r, done, _ = c.step_action(c.wait_id)
    assert r == 0 and not done and c.state_key()[0] == 1


def test_D9_end_charges_pending():
    c = core("diminuto", p_arrival=0.0)
    c.reset(0)
    c.set_state((5, ((1, 0, 1, 3),), ((0, 0, 2, 0),) * 2))
    r, done, _ = c.step_action(c.wait_id)
    assert done and r == -c.cfg.U
    with pytest.raises(RuntimeError):
        c.step_action(c.wait_id)


def test_deterministic_given_seed_and_actions():
    def run(seed):
        c = core("mediano")
        c.reset(seed)
        rng = np.random.default_rng(9)
        out = []
        for _ in range(200):
            r, done, _ = c.step_action(int(rng.choice(c.legal_actions())))
            out.append((r, c.state_key()))
            if done:
                break
        return out
    assert run(4) == run(4)


@pytest.mark.parametrize("size", SIZES)
def test_D11_D13_property_1000_random_episodes(size):
    c = core(size)
    cfg = c.cfg
    rng = np.random.default_rng(0)
    for ep in range(1000):
        c.reset(ep)
        served_qty = np.zeros(cfg.n_trucks)
        ret, done, n_steps = 0.0, False, 0
        while not done:
            a = int(rng.choice(c.legal_actions()))
            slot_pend = c.state_key()[1]
            r, done, info = c.step_action(a)
            if a != c.wait_id:
                s, k = divmod(a, cfg.n_trucks)
                served_qty[k] += slot_pend[s][2]
            trucks = c.state_key()[2]
            assert all(tr[2] >= 0 for tr in trucks)                           # cap never negative
            ret += r
            n_steps += 1
            assert n_steps <= cfg.max_steps
        assert (served_qty <= cfg.capacity).all()
        b = info["cost_breakdown"]
        assert set(b) == {"distance", "lateness", "unserved"}
        assert ret == pytest.approx(-(b["distance"] + b["lateness"] + b["unserved"]))
        assert b["unserved"] % cfg.U == 0


# ---- T7: environment with observation --------------------------------------------------------
from rlf.envs.despacho import DespachoEnv  # noqa: E402


def env_at(size, key, **over):
    e = DespachoEnv(replace(DespachoConfig.from_size(size), p_arrival=0.0, **over))
    e.reset(0)
    e.core.set_state(key)
    return e, e._obs()


def test_env_specs():
    sp = [DespachoEnv.from_size(n).spec for n in SIZES]
    assert [(s.state_dim, s.option_dim, s.entity_dim) for s in sp] == [(3, 6, 6)] * 3
    assert [s.max_options for s in sp] == [3, 13, 73]
    assert [s.n_actions for s in sp] == [3, 13, 73]
    assert [s.max_entities for s in sp] == [3, 7, 18]
    assert [s.max_steps for s in sp] == [12, 60, 120]
    assert all(s.n_players == 1 for s in sp)


def test_D10_option_order_and_ids():
    key = (0, ((1, 1, 1, 6), None, (2, 2, 1, 7), None),
           ((0, 0, 12, 0), (0, 0, 12, 0), (0, 0, 12, 0)))
    e, obs = env_at("mediano", key)
    K, M = 3, 4
    expect = [0 * K + 0, 0 * K + 1, 0 * K + 2, 2 * K + 0, 2 * K + 1, 2 * K + 2, M * K]
    assert obs.option_ids.tolist() == expect
    assert (obs.options[:-1, 0] == 0).all() and obs.options[-1].tolist() == [1, 0, 0, 0, 0, 0]
    assert obs.option_refs[0].tolist() == [K + 0, 0] and obs.option_refs[4].tolist() == [K + 2, 1]
    assert obs.option_refs[-1].tolist() == [-1, -1]


def test_step_index_maps_to_option():
    key = (0, ((1, 1, 1, 6), None, None, None), ((0, 0, 12, 0), (0, 0, 12, 0), (0, 0, 12, 0)))
    e, obs = env_at("mediano", key)
    st = e.step(1)                                   # option 1 = (slot 0, truck 1)
    assert st.rewards[0] == -2 and not st.done
    assert e.core.state_key()[2][1][:3] == (1, 1, 11)


def test_option_features():
    # mediano: 5x5, T=30, Q=12. slot 0 at (2,2) qty 3, due in 10; truck 1 at (0,0) cap 12
    key = (5, ((2, 2, 3, 10), None, None, None), ((0, 0, 12, 0), (0, 0, 12, 0), (0, 0, 12, 0)))
    e, obs = env_at("mediano", key)
    row = obs.options[0]                             # (slot 0, truck 0): d = 4, arrives 9 <= due 15
    assert row[0] == 0 and row[1] == pytest.approx(4 / 8) and row[2] == 0
    assert row[3] == pytest.approx(9 / 12) and row[4] == pytest.approx(10 / 30) and row[5] == pytest.approx(3 / 12)


def test_option_lateness_feature():
    key = (0, ((4, 4, 1, 3), None, None, None), ((0, 0, 12, 0),) * 3)   # d = 8 > due 3 -> late 5
    e, obs = env_at("mediano", key)
    assert obs.options[0, 2] == pytest.approx(5 / 30)


def test_state_and_entities_encoding():
    key = (6, ((1, 1, 3, 10), None, None, None), ((2, 0, 6, 0), (0, 4, 12, 4), (0, 0, 12, 0)))
    e, obs = env_at("mediano", key)
    assert obs.state.tolist() == pytest.approx([6 / 30, 1 / 4, 2 / 3])
    ent = obs.entities
    assert ent.shape == (7, 6) and obs.entity_mask.tolist() == [True] * 3 + [True, False, False, False]
    assert ent[0].tolist() == pytest.approx([0, 2 / 4, 0, 6 / 12, 0, 1])
    assert ent[1].tolist() == pytest.approx([0, 0, 1, 1, 4 / 30, 0])             # busy: marca 0
    assert ent[3].tolist() == pytest.approx([1, 1 / 4, 1 / 4, 3 / 12, 10 / 30, 1])
    assert (ent[4:] == 0).all()


def test_entity_time_is_clipped():
    key = (10, ((1, 1, 1, -50), None, None, None), ((0, 0, 12, 0),) * 3)
    e, obs = env_at("mediano", key)
    assert obs.entities[3, 4] == -1.0 and obs.options[0, 4] == -1.0


def test_D12_state_key_format_and_hashable():
    e = DespachoEnv.from_size("mediano")
    e.reset(2)
    t, pend, trucks = e.state_key()
    hash(e.state_key())
    assert t == 0 and len(pend) == 4 and len(trucks) == 3
    assert trucks[0] == (0, 0, 12, 0)


def test_env_return_matches_cost_breakdown():
    e = DespachoEnv.from_size("diminuto")
    rng = np.random.default_rng(1)
    for ep in range(50):
        obs = e.reset(ep)
        ret, done = 0.0, False
        while not done:
            st = e.step(int(rng.integers(obs.n_options)))
            ret += st.rewards[0]
            obs, done = st.obs, st.done
        b = st.info["cost_breakdown"]
        assert ret == pytest.approx(-sum(b.values()))


# ---- T8: exact MDP and greedy holgura ---------------------------------------------------------
from rlf.agents.baselines import GreedyDespacho, RandomPolicy  # noqa: E402
from rlf.core import EVAL_SEED_BASE, BatchObs  # noqa: E402
from rlf.envs.despacho import build_mdp  # noqa: E402
from rlf.eval.arena import evaluate_policy  # noqa: E402
from rlf.eval.stats import paired_diff_ci, passes_lower  # noqa: E402


@pytest.fixture(scope="module")
def mdp():
    return build_mdp(DespachoConfig.from_size("diminuto"))


def test_mdp_structure(mdp):
    nS = len(mdp.keys)
    assert nS <= 200_000 and nS > 50
    assert mdp.nA == 3 and mdp.legal.shape == (nS, 3) and mdp.R.shape == (nS, 3)
    assert all(mdp.index[k] == i for i, k in enumerate(mdp.keys))
    assert mdp.p0.sum() == pytest.approx(1.0) and (mdp.p0 > 0).sum() == 4
    assert mdp.terminal.sum() > 0 and not mdp.legal[mdp.terminal].any()
    assert mdp.legal[~mdp.terminal, 2].all()                         # wait always legal


def test_mdp_p0_is_exact(mdp):
    assert mdp.p0[mdp.index[(0, (None,), ((0, 0, 2, 0),) * 2)]] == pytest.approx(0.5)
    one = [i for i in np.flatnonzero(mdp.p0) if mdp.keys[i][1][0] is not None]
    assert [mdp.p0[i] for i in one] == pytest.approx([1 / 6] * 3)


def test_mdp_rows_sum_to_one_and_reward_range(mdp):
    nS = len(mdp.keys)
    tot = np.zeros((nS, 3))
    np.add.at(tot, (mdp.src, mdp.act), mdp.prob)
    assert np.allclose(tot[mdp.legal], 1.0) and (tot[~mdp.legal] == 0).all()
    assert (mdp.R[mdp.legal] <= 1e-12).all() and (mdp.R[mdp.legal] >= -14 - 1e-9).all()
    assert (mdp.prob > 0).all()


def test_mdp_last_tick_wait_charges_pending_exactly(mdp):
    key = (5, ((1, 0, 1, 3),), ((0, 0, 2, 0),) * 2)
    if key in mdp.index:
        s = mdp.index[key]
        assert mdp.R[s, 2] == pytest.approx(-14.0)
        rows = mdp.dst[(mdp.src == s) & (mdp.act == 2)]
        assert len(rows) == 1 and mdp.terminal[rows[0]]


def test_mdp_fidelity_against_simulation(mdp):
    rng = np.random.default_rng(0)
    cands = [(s, a) for s in np.flatnonzero(~mdp.terminal) for a in np.flatnonzero(mdp.legal[s])]
    pick = rng.choice(len(cands), 20, replace=False)
    core = DespachoCore(DespachoConfig.from_size("diminuto"))
    core.reset(0)
    n = 20_000
    for pi in pick:
        s, a = cands[pi]
        key = mdp.keys[s]
        rs, nxt = np.zeros(n), []
        for i in range(n):
            core.set_state(key, seed=int(pi) if i == 0 else None)
            r, _, _ = core.step_action(int(a))
            rs[i] = r
            nxt.append(core.state_key())
        se = rs.std(ddof=1) / np.sqrt(n)
        assert abs(rs.mean() - mdp.R[s, a]) <= 3 * se + 1e-9
        sel = (mdp.src == s) & (mdp.act == a)
        expected = dict(zip((mdp.keys[d] for d in mdp.dst[sel]), mdp.prob[sel]))
        counts = {}
        for k in nxt:
            counts[k] = counts.get(k, 0) + 1
        assert set(counts) <= set(expected)                       # no impossible successor
        for k, p in expected.items():
            f = counts.get(k, 0) / n
            assert abs(f - p) <= 3 * np.sqrt(p * (1 - p) / n) + 1e-9


def batch_of(obs):
    return BatchObs.from_obs(obs, DespachoEnv.from_size("mediano").spec)


def test_greedy_picks_smallest_holgura_then_nd_then_index():
    # slots 1 and 2 tie on the smallest slack (3/30); slot 1 with truck 1 has distance 0
    key = (0, ((1, 0, 1, 9), (4, 4, 1, 3), (0, 1, 1, 3), None),
           ((0, 0, 12, 0), (4, 4, 12, 0), (0, 0, 12, 0)))
    e, obs = env_at("mediano", key)
    assert obs.option_ids[GreedyDespacho().act(batch_of(obs))[0]] == 1 * 3 + 1
    # one order, trucks 0 and 2 tie on distance -> lowest index
    key = (0, ((1, 0, 1, 9), None, None, None), ((0, 0, 12, 0), (4, 4, 12, 0), (0, 0, 12, 0)))
    e, obs = env_at("mediano", key)
    assert obs.option_ids[GreedyDespacho().act(batch_of(obs))[0]] == 0


def test_greedy_waits_only_if_nothing_to_assign_and_is_deterministic():
    e, obs = env_at("mediano", (0, (None,) * 4, ((0, 0, 12, 0),) * 3))
    assert obs.option_ids[GreedyDespacho().act(batch_of(obs))[0]] == 12
    key = (0, ((1, 0, 1, 9), None, None, None), ((0, 0, 12, 0),) * 3)
    e, obs = env_at("mediano", key)
    b = batch_of(obs)
    assert obs.option_ids[GreedyDespacho().act(b)[0]] != 12
    assert np.array_equal(GreedyDespacho().act(b), GreedyDespacho().act(b))


class NoisyGreedy:
    def __init__(self, eps, seed):
        self.g, self.eps, self.rng = GreedyDespacho(), eps, np.random.default_rng(seed)

    def act(self, obs):
        a = self.g.act(obs)
        rnd = self.rng.integers(0, obs.n_options)
        return np.where(self.rng.random(len(a)) < self.eps, rnd, a).astype(np.int64)


def test_D14_ii_headroom_over_greedy_mediano():
    make = lambda: DespachoEnv.from_size("mediano")
    n = 1000
    greedy = evaluate_policy(make, NoisyGreedy(0.0, 0), n)
    best = greedy.copy()
    for j in range(1, 20):
        best = np.maximum(best, evaluate_policy(make, NoisyGreedy(0.2, j), n))
    assert passes_lower(paired_diff_ci(best, greedy), 1e-12)
