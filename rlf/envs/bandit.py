"""k-armed bandit with Gaussian rewards, exposed through the option-list interface."""
from __future__ import annotations

import numpy as np

from rlf.core import EnvSpec, Obs, Step


class Bandit:
    def __init__(self, k: int = 10, horizon: int = 300, noise_std: float = 1.0):
        self.k, self.horizon, self.noise_std = k, horizon, noise_std
        self.spec = EnvSpec(name="bandit", n_players=1, state_dim=1, option_dim=k, entity_dim=1,
                            max_options=k, max_entities=0, n_actions=k, max_steps=horizon)
        self.seed = None
        self._done = True
        self._options = np.eye(k, dtype=np.float32)
        self._ids = np.arange(k, dtype=np.int64)
        self._refs = -np.ones((k, 2), np.int64)

    def reset(self, seed: int | None = None) -> Obs:
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0] % 10_000_000)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.mu = self.rng.normal(0.0, 1.0, self.k)      # B1
        self.t = 0
        self._done = False
        return self._obs()

    def _obs(self) -> Obs:
        n = 0 if self._done else self.k
        return Obs(state=np.array([self.t / self.horizon], np.float32), options=self._options[:n],
                   option_ids=self._ids[:n], option_refs=self._refs[:n],
                   entities=np.zeros((0, 1), np.float32), entity_mask=np.zeros(0, bool), player=0)

    def step(self, choice: int) -> Step:
        if self._done:
            raise RuntimeError("step() after done")
        if not 0 <= choice < self.k:
            raise IndexError(choice)
        r = self.rng.normal(self.mu[choice], self.noise_std)     # B2
        self.t += 1
        self._done = self.t >= self.horizon                       # B4
        info = {"mu_chosen": float(self.mu[choice]), "mu_best": float(self.mu.max())}   # B5
        return Step(self._obs(), np.array([r]), self._done, info)

    def state_key(self) -> tuple:
        return (self.t,)
