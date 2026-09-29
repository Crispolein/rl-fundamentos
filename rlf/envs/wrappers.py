"""Env wrappers."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from rlf.core import BatchObs, Env, Obs, Step


class VsOpponent:
    """Single-player view of a two-player env: the learner plays against a fixed `opponent` policy.

    The learner takes seat `seed % 2` (so seats alternate across episodes); the opponent's moves are played inside
    `step`, and rewards and observations are those of the learner (`obs.player` is always 0). If the opponent has
    a `reseed(seed)` method it is called at each reset, which keeps episodes deterministic given the seed.
    """

    def __init__(self, make_env, opponent, configure=None):
        self.env: Env = make_env()
        self.opponent = opponent
        self.configure = configure               # optional hook(env, learner_seat) run before each reset
        self.spec = replace(self.env.spec, n_players=1)
        self.seed = None
        self.seat = 0

    def reset(self, seed: int | None = None) -> Obs:
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0] % 10_000_000)
        self.seat = seed % 2
        if self.configure is not None:
            self.configure(self.env, self.seat)
        obs = self.env.reset(seed)
        self.seed = self.env.seed
        if hasattr(self.opponent, "reseed"):
            self.opponent.reseed(self.seed + 1)
        while obs.player != self.seat:                 # the opponent may move first
            obs = self.env.step(self._opponent_move(obs)).obs
        return replace(obs, player=0)

    def _opponent_move(self, obs: Obs) -> int:
        return int(self.opponent.act(BatchObs.from_obs(obs, self.env.spec))[0])

    def step(self, choice: int) -> Step:
        st = self.env.step(choice)
        reward = float(st.rewards[self.seat])
        while not st.done and st.obs.player != self.seat:
            st = self.env.step(self._opponent_move(st.obs))
            reward += float(st.rewards[self.seat])
        return Step(replace(st.obs, player=0), np.array([reward]), st.done, st.info)

    def state_key(self) -> tuple:
        return self.env.state_key()
