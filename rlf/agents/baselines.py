"""Reference policies: uniform random, and (added by later tasks) the hand-written heuristics."""
from __future__ import annotations

import numpy as np

from rlf.core import BatchObs
from rlf.envs.cartas import CARDS_DMG, CARDS_HP


class RandomPolicy:
    """Picks uniformly among the legal options."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reseed(self, seed: int) -> None:
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


class HeuristicCartas:
    """Deterministic, reads only the observation.

    SETUP: place the creature with the highest damage. PROMOTE: most attached energies, then highest HP.
    MAIN priority: 1) ATTACH to the active, 2) PLAY_DRAW, 3) PLAY_CREATURE (highest damage),
    4) PLAY_HEAL on the most damaged creature, 5) ATTACK, 6) END_TURN. Ties: lowest option index.
    """

    def act(self, obs: BatchObs) -> np.ndarray:
        out = np.zeros(len(obs), np.int64)
        for b in range(len(obs)):
            n = int(obs.n_options[b])
            best = None
            for i in range(n):
                o = obs.options[b, i]
                kind = int(np.argmax(o[:8]))
                ctype = int(np.argmax(o[8:15])) if o[8:15].any() else -1
                ent = obs.entities[b]
                ref = int(obs.option_refs[b, i, 0])
                if kind == 6:                                            # PLACE_ACTIVE
                    key = (0, -CARDS_DMG[ctype])
                elif kind == 7:                                          # PROMOTE
                    key = (0, -ent[ref, 11], -ent[ref, 10])
                elif kind == 2:                                          # ATTACH
                    key = (1,) if o[15] == 1 else (99,)
                elif kind == 4:                                          # PLAY_DRAW
                    key = (2,)
                elif kind == 3:                                          # PLAY_CREATURE
                    key = (3, -CARDS_DMG[ctype])
                elif kind == 5:                                          # PLAY_HEAL
                    ctype_target = int(np.argmax(ent[ref, 6:10]))
                    key = (4, -(CARDS_HP[ctype_target] / 8 - ent[ref, 10]))
                elif kind == 1:                                          # ATTACK
                    key = (5,)
                else:                                                    # END_TURN
                    key = (6,)
                key = key + (i,)
                if best is None or key < best[0]:
                    best = (key, i)
            out[b] = best[1]
        return out
