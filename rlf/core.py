"""Agent <-> environment interface, copied from the PTCG engine: the environment returns an
observation plus the list of legal options, and the agent answers with the index of one option."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Callable, Protocol

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


@dataclass(frozen=True)
class BatchObs:
    """Observations of B environments, padded up to `max_options`."""
    state: np.ndarray        # float32 [B, state_dim]
    options: np.ndarray      # float32 [B, max_options, option_dim]; padding is 0
    option_mask: np.ndarray  # bool [B, max_options]; True = legal option
    option_ids: np.ndarray   # int64 [B, max_options]; padding is -1
    option_refs: np.ndarray  # int64 [B, max_options, 2]; padding is -1
    entities: np.ndarray     # float32 [B, max_entities, entity_dim]
    entity_mask: np.ndarray  # bool [B, max_entities]
    player: np.ndarray       # int64 [B]

    def __len__(self) -> int:
        return len(self.state)

    @property
    def n_options(self) -> np.ndarray:
        return self.option_mask.sum(1).astype(np.int64)

    def select(self, idx: np.ndarray) -> "BatchObs":
        """Sub-batch by integer indices or by a boolean mask."""
        idx = np.asarray(idx)
        return BatchObs(**{f: getattr(self, f)[idx] for f in _BATCH_FIELDS})

    @classmethod
    def from_obs(cls, obs: Obs, spec: EnvSpec) -> "BatchObs":
        return stack_obs([obs], spec)


_BATCH_FIELDS = ("state", "options", "option_mask", "option_ids", "option_refs", "entities",
                 "entity_mask", "player")


def stack_obs(obs_list: list[Obs], spec: EnvSpec) -> BatchObs:
    B, M = len(obs_list), spec.max_options
    options = np.zeros((B, M, spec.option_dim), np.float32)
    option_mask = np.zeros((B, M), bool)
    option_ids = -np.ones((B, M), np.int64)
    option_refs = -np.ones((B, M, 2), np.int64)
    for b, o in enumerate(obs_list):
        n = o.n_options
        options[b, :n] = o.options
        option_mask[b, :n] = True
        option_ids[b, :n] = o.option_ids
        option_refs[b, :n] = o.option_refs
    return BatchObs(
        state=np.stack([o.state for o in obs_list]).astype(np.float32),
        options=options, option_mask=option_mask, option_ids=option_ids, option_refs=option_refs,
        entities=np.stack([o.entities for o in obs_list]).astype(np.float32),
        entity_mask=np.stack([o.entity_mask for o in obs_list]),
        player=np.array([o.player for o in obs_list], np.int64))


@dataclass(frozen=True)
class BatchStep:
    obs: BatchObs            # finished envs are already reset (auto-reset)
    rewards: np.ndarray      # float64 [B, n_players]
    done: np.ndarray         # bool [B]; True = this transition ended the episode
    info: list               # info[b] is the last step's; if done[b] it includes "episode_id"


class VecEnv:
    """Stacks `n_envs` single environments in a Python loop (correct by construction, C8).

    Episode `e` of env `b` uses seed `seed_fn(b, e)`; its `episode_id` is `e * n_envs + b`.
    """

    def __init__(self, make_env: Callable[[], Env], n_envs: int, seed: int = 0,
                 seed_fn: Callable[[int, int], int] | None = None):
        self.envs = [make_env() for _ in range(n_envs)]
        self.n_envs = n_envs
        self.spec = self.envs[0].spec
        self.seed_fn = seed_fn or (lambda b, e: derive_seed(seed, b, e))
        self._episode = [0] * n_envs
        self._obs: list[Obs] = []

    def reset(self) -> BatchObs:
        self._episode = [0] * self.n_envs
        self._obs = [env.reset(self.seed_fn(b, 0)) for b, env in enumerate(self.envs)]
        return stack_obs(self._obs, self.spec)

    def step(self, choices: np.ndarray) -> BatchStep:
        rewards = np.zeros((self.n_envs, self.spec.n_players))
        done = np.zeros(self.n_envs, bool)
        info = []
        for b, env in enumerate(self.envs):
            st = env.step(int(choices[b]))
            rewards[b] = st.rewards
            done[b] = st.done
            d = dict(st.info)
            if st.done:
                d["episode_id"] = self._episode[b] * self.n_envs + b
                self._episode[b] += 1
                self._obs[b] = env.reset(self.seed_fn(b, self._episode[b]))
            else:
                self._obs[b] = st.obs
            info.append(d)
        return BatchStep(stack_obs(self._obs, self.spec), rewards, done, info)


def flatten_obs(obs: BatchObs) -> np.ndarray:
    """state ++ flattened entities, float32 [B, state_dim + max_entities * entity_dim]."""
    B = len(obs)
    return np.concatenate([obs.state, obs.entities.reshape(B, -1)], axis=1).astype(np.float32)


@dataclass(frozen=True)
class TorchObs:
    state: torch.Tensor
    options: torch.Tensor
    option_mask: torch.Tensor
    option_ids: torch.Tensor
    option_refs: torch.Tensor
    entities: torch.Tensor
    entity_mask: torch.Tensor
    player: torch.Tensor


def to_torch(obs: BatchObs, device: torch.device) -> TorchObs:
    return TorchObs(**{f: torch.from_numpy(getattr(obs, f)).to(device) for f in _BATCH_FIELDS})
