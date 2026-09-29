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
