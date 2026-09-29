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


class GreedyDespacho:
    """Deterministic: assign the order with the smallest slack (options col 4); ties by smaller
    normalized distance (col 1), then by lower index. Waits only if nothing can be assigned."""

    def act(self, obs: BatchObs) -> np.ndarray:
        out = np.zeros(len(obs), np.int64)
        for b in range(len(obs)):
            n = int(obs.n_options[b])
            opts = obs.options[b, :n]
            cand = np.flatnonzero(opts[:, 0] == 0)
            if len(cand) == 0:
                out[b] = n - 1                                   # WAIT is always the last option
                continue
            order = np.lexsort((cand, opts[cand, 1], opts[cand, 4]))   # slack, then distance, then index
            out[b] = cand[order[0]]
        return out
