"""MiniCartas: two-player card game with hidden information (rules R1-R24 in design.md 4.3).

`CartasCore` holds the game rules; the environment that wraps it with observations is added below.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_TYPES = 7
ENERGY, STUDY, POTION = 4, 5, 6
CARDS_HP = np.array([3, 5, 8, 6, 0, 0, 0])
CARDS_DMG = np.array([2, 3, 2, 5, 0, 0, 0])
CARDS_COST = np.array([1, 2, 2, 3, 0, 0, 0])
ARCHETYPES = {"aggro": [6, 3, 0, 0, 8, 2, 1], "tank": [0, 2, 5, 0, 8, 1, 4], "dragon": [2, 3, 0, 3, 8, 3, 1]}
MAXC = np.array([max(v[t] for v in ARCHETYPES.values()) for t in range(N_TYPES)])

# action ids (R22)
END_TURN, ATTACK, ATTACH, PLAY_CREATURE, PLAY_DRAW, PLAY_HEAL, PLACE_ACTIVE, PROMOTE = 0, 1, 2, 5, 9, 10, 13, 17
N_ACTIONS = 19


@dataclass(frozen=True)
class CartasConfig:
    deck_p0: str = "aggro"
    deck_p1: str = "aggro"


class CartasCore:
    """State: `board[p]` = [active, bench0, bench1], each None or [type, hp, energies]."""

    def __init__(self, cfg: CartasConfig):
        self.cfg = cfg
        self.done = True

    def reset(self, seed: int | None = None) -> None:
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0] % 10_000_000)
        self.seed = seed
        rng = self.rng = np.random.default_rng(seed)
        names = []
        for choice in (self.cfg.deck_p0, self.cfg.deck_p1):
            names.append(list(ARCHETYPES)[int(rng.integers(3))] if choice == "random" else choice)
        self.deck_names = names
        decks = []
        for p in (0, 1):                                            # R2: player 0 shuffled first
            cards = [t for t, n in enumerate(ARCHETYPES[names[p]]) for _ in range(n)]
            decks.append([int(x) for x in rng.permutation(cards)])
        self.decks, self.hands, self.prizes = decks, [[], []], [[], []]
        self.discard = [[], []]
        self.mulligans = [0, 0]
        for p in (0, 1):
            while True:                                             # R3
                self.hands[p] = [self.decks[p].pop(0) for _ in range(5)]
                if any(c < 4 for c in self.hands[p]):
                    break
                self.mulligans[p] += 1
                self.decks[p] = [int(x) for x in rng.permutation(self.decks[p] + self.hands[p])]
                self.hands[p] = []
            self.prizes[p] = [self.decks[p].pop(0) for _ in range(3)]      # R4
        self.board = [[None, None, None], [None, None, None]]
        self.prizes_taken = [0, 0]
        self.attached = False
        self.support = False
        self.winner = None
        self.done = False
        self.phase = "SETUP"                                        # R5
        self.player = 0

    # ---- legal actions -----------------------------------------------------------------------
    def legal_actions(self) -> list[int]:
        h = self.hands[self.player]
        if self.phase == "SETUP":
            return [PLACE_ACTIVE + t for t in sorted({c for c in h if c < 4})]
        return []

    def step_action(self, a: int):
        if self.done:
            raise RuntimeError("step after done")
        if a not in self.legal_actions():
            raise ValueError(f"illegal action {a}")
        p = self.player
        if self.phase == "SETUP":
            t = a - PLACE_ACTIVE
            self.hands[p].remove(t)
            self.board[p][0] = [t, int(CARDS_HP[t]), 0]
            if p == 0:
                self.player = 1
            else:
                self.phase, self.player = "MAIN", 0
        return np.zeros(2), self.done, {"mulligans": list(self.mulligans)}
