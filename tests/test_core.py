import os
import random

import numpy as np
import pytest
import torch

from rlf import core
from rlf.core import (EVAL_SEED_BASE, EnvSpec, Obs, Step, derive_seed,
                      get_device, seed_everything)


def make_spec():
    return EnvSpec(name="t", n_players=1, state_dim=2, option_dim=3, entity_dim=0,
                   max_options=4, max_entities=0, n_actions=4, max_steps=10)


def make_obs(n_opts=2):
    return Obs(state=np.zeros(2, np.float32), options=np.zeros((n_opts, 3), np.float32),
               option_ids=np.arange(n_opts, dtype=np.int64),
               option_refs=-np.ones((n_opts, 2), np.int64),
               entities=np.zeros((0, 1), np.float32), entity_mask=np.zeros(0, bool), player=0)


def test_eval_seed_base():
    assert EVAL_SEED_BASE == 10_000_000


def test_spec_is_frozen():
    with pytest.raises(Exception):
        make_spec().name = "x"


def test_obs_n_options_and_step():
    assert make_obs(3).n_options == 3
    s = Step(obs=make_obs(0), rewards=np.zeros(1), done=True, info={})
    assert s.done and s.obs.n_options == 0


def test_derive_seed_deterministic_and_in_training_range():
    assert derive_seed(5, 1, 2) == derive_seed(5, 1, 2)
    vals = {derive_seed(5, b, e) for b in range(20) for e in range(20)}
    assert len(vals) > 300
    assert all(0 <= v < EVAL_SEED_BASE for v in vals)
    assert derive_seed(5, 1, 2) != derive_seed(5, 2, 1)


def test_seed_everything_reproducible():
    seed_everything(3)
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    seed_everything(3)
    b = (random.random(), np.random.rand(), torch.rand(1).item())
    assert a == b


def test_seed_everything_threads(monkeypatch):
    monkeypatch.setenv("RLF_THREADS", "2")
    seed_everything(0)
    assert torch.get_num_threads() == 2


def test_get_device_env_override(monkeypatch):
    monkeypatch.setenv("RLF_DEVICE", "cpu")
    assert get_device() == torch.device("cpu")


def test_conftest_defaults_to_cpu():
    assert os.environ.get("RLF_DEVICE") == "cpu"


# ---- batch helpers (T2) ----
from rlf.core import BatchObs, TorchObs, VecEnv, flatten_obs, to_torch  # noqa: E402
from tests.toy_env import ToyEnv  # noqa: E402


def test_batchobs_from_obs_pads():
    env = ToyEnv()
    o = env.reset(0)
    b = BatchObs.from_obs(o, env.spec)
    sp = env.spec
    assert b.options.shape == (1, sp.max_options, sp.option_dim)
    assert b.option_mask.shape == (1, sp.max_options)
    assert b.n_options[0] == o.n_options
    assert b.option_mask[0].sum() == o.n_options
    assert (b.option_ids[0, o.n_options:] == -1).all()
    assert (b.options[0, o.n_options:] == 0).all()
    assert (b.option_refs[0, o.n_options:] == -1).all()


def test_vecenv_shapes_select_and_autoreset():
    vec = VecEnv(lambda: ToyEnv(), 4, seed=0)
    obs = vec.reset()
    sp = vec.spec
    assert obs.state.shape == (4, sp.state_dim)
    assert obs.entities.shape == (4, sp.max_entities, sp.entity_dim)
    assert obs.player.shape == (4,)
    sub = obs.select(np.array([0, 2]))
    assert len(sub.state) == 2 and np.array_equal(sub.state[1], obs.state[2])
    sub2 = obs.select(np.array([True, False, True, False]))
    assert len(sub2.state) == 2
    ids = set()
    for _ in range(12):
        bs = vec.step(np.zeros(4, np.int64))
        assert bs.rewards.shape == (4, 1) and bs.done.shape == (4,) and len(bs.info) == 4
        assert (bs.obs.n_options >= 1).all()          # auto-reset: never a terminal obs
        for b in np.flatnonzero(bs.done):
            ids.add(bs.info[b]["episode_id"])
    assert len(ids) >= 4                              # unique ids


def test_vecenv_eval_seed_fn_pairs_episodes():
    from rlf.core import EVAL_SEED_BASE
    n = 3
    seeds = []
    make = lambda: ToyEnv()
    vec = VecEnv(make, n, seed_fn=lambda b, e: EVAL_SEED_BASE + e * n + b)
    obs = vec.reset()
    ref = [ToyEnv().reset(EVAL_SEED_BASE + b) for b in range(n)]
    for b in range(n):
        assert np.array_equal(obs.state[b], ref[b].state)
        assert np.array_equal(obs.options[b, :ref[b].n_options], ref[b].options)


def test_flatten_and_to_torch():
    vec = VecEnv(lambda: ToyEnv(), 3, seed=1)
    obs = vec.reset()
    sp = vec.spec
    flat = flatten_obs(obs)
    assert flat.dtype == np.float32 and flat.shape == (3, sp.state_dim + sp.max_entities * sp.entity_dim)
    assert np.array_equal(flat[:, :sp.state_dim], obs.state)
    t = to_torch(obs, torch.device("cpu"))
    assert isinstance(t, TorchObs)
    assert t.state.dtype == torch.float32 and t.option_mask.dtype == torch.bool
    assert t.option_ids.dtype == torch.int64 and t.player.dtype == torch.int64
    assert torch.equal(t.option_mask, torch.from_numpy(obs.option_mask))
