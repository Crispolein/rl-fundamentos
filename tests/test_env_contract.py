"""Contract C1-C8 of the agent <-> environment interface (design.md section 3)."""
import numpy as np
import pytest

from rlf.core import BatchObs, VecEnv
from tests.env_registry import ENVS

PARAMS = list(ENVS.items())
IDS = [k for k, _ in PARAMS]


def rollout(env, seed, n_steps, choice_seed):
    rng = np.random.default_rng(choice_seed)
    obs = env.reset(seed)
    traj = [obs]
    rewards = []
    while len(rewards) < n_steps:
        j = int(rng.integers(obs.n_options))
        st = env.step(j)
        traj.append(st.obs)
        rewards.append((st.rewards.copy(), st.done))
        obs = st.obs
        if st.done:
            break
    return traj, rewards


def obs_equal(a, b):
    return (a.player == b.player and all(np.array_equal(getattr(a, f), getattr(b, f)) for f in
            ("state", "options", "option_ids", "option_refs", "entities", "entity_mask")))


@pytest.mark.parametrize("name,make", PARAMS, ids=IDS)
def test_C1_determinism(name, make):
    for seed in (0, 7, 123):
        t1, r1 = rollout(make(), seed, 500, 99)
        t2, r2 = rollout(make(), seed, 500, 99)
        assert len(t1) == len(t2)
        assert all(obs_equal(a, b) for a, b in zip(t1, t2))
        assert all(np.array_equal(a[0], b[0]) and a[1] == b[1] for a, b in zip(r1, r2))


@pytest.mark.parametrize("name,make", PARAMS, ids=IDS)
def test_C1_reset_none_records_seed(name, make):
    env = make()
    env.reset(None)
    s = env.seed
    o1 = env.reset(s)
    o2 = make().reset(s)
    assert obs_equal(o1, o2)


@pytest.mark.parametrize("name,make", PARAMS, ids=IDS)
def test_C2_C3_C4_legality_ids_shapes(name, make):
    env = make()
    sp = env.spec
    rng = np.random.default_rng(0)
    for ep in range(30):
        obs = env.reset(ep)
        done = False
        while not done:
            assert obs.n_options >= 1                                         # C2
            ids = obs.option_ids
            assert ids.dtype == np.int64 and np.all(np.diff(ids) > 0)         # C3
            assert ids[0] >= 0 and ids[-1] < sp.n_actions
            assert obs.options.shape == (len(ids), sp.option_dim)             # C4
            assert obs.option_refs.shape == (len(ids), 2)
            assert obs.n_options <= sp.max_options
            assert obs.state.shape == (sp.state_dim,)
            assert obs.entities.shape == (sp.max_entities, sp.entity_dim)
            assert obs.entity_mask.shape == (sp.max_entities,) and obs.entity_mask.dtype == bool
            for a in (obs.state, obs.options, obs.entities):
                assert a.dtype == np.float32 and np.isfinite(a).all()
            st = env.step(int(rng.integers(obs.n_options)))
            obs, done = st.obs, st.done


@pytest.mark.parametrize("name,make", PARAMS, ids=IDS)
def test_C5_errors(name, make):
    env = make()
    obs = env.reset(0)
    with pytest.raises(IndexError):
        env.step(obs.n_options)
    with pytest.raises(IndexError):
        env.step(-1)
    done = False
    while not done:
        st = env.step(0)
        done = st.done
    with pytest.raises(RuntimeError):
        env.step(0)


@pytest.mark.parametrize("name,make", PARAMS, ids=IDS)
def test_C6_termination_and_C7_players(name, make):
    env = make()
    sp = env.spec
    rng = np.random.default_rng(1)
    for ep in range(1000):
        obs = env.reset(ep)
        steps, done = 0, False
        while not done:
            assert 0 <= obs.player < sp.n_players                              # C7
            if sp.n_players == 1:
                assert obs.player == 0
            st = env.step(int(rng.integers(obs.n_options)))
            assert st.rewards.shape == (sp.n_players,)
            obs, done = st.obs, st.done
            steps += 1
            assert steps <= sp.max_steps
        assert obs.n_options == 0                                              # C6


@pytest.mark.parametrize("name,make", PARAMS, ids=IDS)
def test_C8_vecenv_matches_single_envs(name, make):
    B, T = 5, 60
    seed_fn = lambda b, e: 1000 * b + e
    vec = VecEnv(make, B, seed=0, seed_fn=seed_fn)
    singles = [make() for _ in range(B)]
    ep = [0] * B
    obs_s = [singles[b].reset(seed_fn(b, 0)) for b in range(B)]
    vobs = vec.reset()
    assert isinstance(vobs, BatchObs) and len(vobs.state) == B
    rng = np.random.default_rng(3)
    for _ in range(T):
        for b in range(B):
            o = vobs.select(np.array([b]))
            assert o.n_options[0] == obs_s[b].n_options
            assert np.array_equal(o.options[0, :obs_s[b].n_options], obs_s[b].options)
            assert np.array_equal(o.option_ids[0, :obs_s[b].n_options], obs_s[b].option_ids)
            assert np.array_equal(o.state[0], obs_s[b].state)
            assert o.player[0] == obs_s[b].player
        choices = np.array([int(rng.integers(o.n_options)) for o in obs_s], dtype=np.int64)
        bs = vec.step(choices)
        for b in range(B):
            st = singles[b].step(int(choices[b]))
            assert np.array_equal(bs.rewards[b], st.rewards)
            assert bool(bs.done[b]) == st.done
            if st.done:
                assert "episode_id" in bs.info[b]
                ep[b] += 1
                obs_s[b] = singles[b].reset(seed_fn(b, ep[b]))
            else:
                obs_s[b] = st.obs
        vobs = bs.obs
