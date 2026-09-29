"""Agent <-> environment interface, copied from the PTCG engine: the environment returns an
observation plus the list of legal options, and the agent answers with the index of one option."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

EVAL_SEED_BASE = 10_000_000


@dataclass(frozen=True)
class EnvSpec:
    name: str
    n_players: int      # 1 (Bandit, MiniDespacho) or 2 (MiniCartas)
    state_dim: int      # size of the global vector
    option_dim: int     # size of each option's feature vector
    entity_dim: int     # size of each entity's feature vector (0 if none)
    max_options: int    # cap on simultaneous options (for padding in batches)
    max_entities: int   # fixed number of entity rows (0 if none)
    n_actions: int      # size of the global action-id space (for tables and masks)
    max_steps: int      # upper bound on decisions per episode


@dataclass(frozen=True)
class Obs:
    """What the player who must decide sees."""
    state: np.ndarray        # float32 [state_dim]
    options: np.ndarray      # float32 [n_opts, option_dim]; one row per legal option
    option_ids: np.ndarray   # int64 [n_opts]; global id of each option, strictly increasing
    option_refs: np.ndarray  # int64 [n_opts, 2]; entity indices the option points to, or -1
    entities: np.ndarray     # float32 [max_entities, entity_dim]; empty rows are zero
    entity_mask: np.ndarray  # bool [max_entities]; True = row has content
    player: int              # player deciding now (0 if n_players == 1)

    @property
    def n_options(self) -> int:
        return len(self.options)


@dataclass(frozen=True)
class Step:
    obs: Obs                 # next observation; if done, n_options == 0
    rewards: np.ndarray      # float64 [n_players]; reward accumulated per player in this transition
    done: bool
    info: dict               # diagnostics; the agent never uses them to decide


class Env(Protocol):
    spec: EnvSpec

    def reset(self, seed: int | None = None) -> Obs: ...

    def step(self, choice: int) -> Step: ...

    def state_key(self) -> tuple: ...


def derive_seed(base: int, env_idx: int, episode_idx: int) -> int:
    """Training seed of episode `episode_idx` of env `env_idx`; always in [0, EVAL_SEED_BASE)."""
    return int(np.random.SeedSequence([base, env_idx, episode_idx]).generate_state(1)[0] % EVAL_SEED_BASE)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(os.environ.get("RLF_THREADS", min(4, os.cpu_count() or 1))))


def get_device() -> torch.device:
    forced = os.environ.get("RLF_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
