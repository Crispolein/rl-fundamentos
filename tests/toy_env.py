"""Tiny environments used to test the core, the contract and the arena without depending on real envs."""
from __future__ import annotations

import numpy as np

from rlf.core import EnvSpec, Obs, Step

N_STEPS = 5
MAX_OPTS = 4


class ToyEnv:
    """Variable number of options (1..4), 2 entities, `N_STEPS` decisions.

    1 player: reward of a step is the chosen option's value (option index / 10).
    2 players: players alternate; the last decision ends the game with rewards [+1,-1] or [-1,+1]
    depending on the parity of the sum of chosen indices.
    """

    def __init__(self, n_players: int = 1):
        self.n_players = n_players
        self.spec = EnvSpec(name=f"toy{n_players}", n_players=n_players, state_dim=2, option_dim=3,
                            entity_dim=2, max_options=MAX_OPTS, max_entities=2, n_actions=MAX_OPTS,
                            max_steps=N_STEPS)
        self.seed = None
        self._done = True

    def reset(self, seed: int | None = None) -> Obs:
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0] % 10_000_000)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.total = 0
        self._done = False
        self._draw()
        return self._obs()

    def _draw(self):
        self._n = int(self.rng.integers(1, MAX_OPTS + 1))
        self._vals = self.rng.random((self._n, 3)).astype(np.float32)

    def _obs(self) -> Obs:
        if self._done:
            n = 0
            options = np.zeros((0, 3), np.float32)
        else:
            n = self._n
            options = self._vals
        ent = np.zeros((2, 2), np.float32)
        mask = np.zeros(2, bool)
        if not self._done:
            ent[0] = [self.t / N_STEPS, 1.0]
            mask[0] = True
        return Obs(state=np.array([self.t / N_STEPS, self.total / 10], np.float32), options=options,
                   option_ids=np.arange(n, dtype=np.int64), option_refs=-np.ones((n, 2), np.int64),
                   entities=ent, entity_mask=mask,
                   player=0 if (self.n_players == 1 or self._done) else self.t % 2)

    def step(self, choice: int) -> Step:
        if self._done:
            raise RuntimeError("step() after done")
        if not 0 <= choice < self._n:
            raise IndexError(choice)
        rewards = np.zeros(self.n_players)
        if self.n_players == 1:
            rewards[0] = choice / 10
        self.total += choice
        self.t += 1
        done = self.t >= N_STEPS
        if done:
            self._done = True
            if self.n_players == 2:
                rewards = np.array([1.0, -1.0]) if self.total % 2 == 0 else np.array([-1.0, 1.0])
        else:
            self._draw()
        return Step(obs=self._obs(), rewards=rewards, done=done, info={"t": self.t})

    def state_key(self) -> tuple:
        return (self.t, self.total)
