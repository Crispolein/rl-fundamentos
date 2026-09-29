"""Reference policies: uniform random, and (added by later tasks) the hand-written heuristics."""
from __future__ import annotations

import numpy as np

from rlf.core import BatchObs


class RandomPolicy:
    """Picks uniformly among the legal options."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs: BatchObs) -> np.ndarray:
        return self.rng.integers(0, obs.n_options).astype(np.int64)
