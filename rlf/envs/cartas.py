"""MiniCartas: two-player card game with hidden information (rules R1-R24 in design.md 4.3).

`CartasCore` holds the game rules; the environment that wraps it with observations is added below.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rlf.core import EnvSpec, Obs, Step

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
        self.turns = [0, 0]
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

    # ---- turn flow ---------------------------------------------------------------------------
    def _win(self, w: int) -> None:
        self.done, self.winner, self.phase = True, w, "DONE"

    def _start_turn(self, p: int) -> None:
        """R6: the player of the turn draws one card; an empty deck loses at once."""
        self.player, self.phase = p, "MAIN"
        self.attached = self.support = False
        self.turns[p] += 1
        if not self.decks[p]:
            self._win(1 - p)
            return
        self.hands[p].append(self.decks[p].pop(0))

    def _n_bench(self, p: int) -> int:
        return sum(s is not None for s in self.board[p][1:])

    # ---- legal actions -----------------------------------------------------------------------
    def legal_actions(self) -> list[int]:
        if self.done:
            return []
        p = self.player
        h, board = self.hands[p], self.board[p]
        if self.phase == "SETUP":
            return [PLACE_ACTIVE + t for t in sorted({c for c in h if c < 4})]
        if self.phase == "PROMOTE":
            return [PROMOTE + j for j in range(2) if board[1 + j] is not None]
        acts = [END_TURN]                                           # R8
        act = board[0]
        if act is not None and act[2] >= CARDS_COST[act[0]]:        # R13
            acts.append(ATTACK)
        if ENERGY in h and not self.attached:                       # R10
            acts += [ATTACH + j for j in range(3) if board[j] is not None]
        if self._n_bench(p) < 2:                                    # R9
            acts += [PLAY_CREATURE + t for t in sorted({c for c in h if c < 4})]
        if not self.support:
            if STUDY in h:                                          # R11
                acts.append(PLAY_DRAW)
            if POTION in h:                                         # R12
                acts += [PLAY_HEAL + j for j in range(3)
                         if board[j] is not None and board[j][1] < CARDS_HP[board[j][0]]]
        return sorted(acts)

    def step_action(self, a: int):
        if self.done:
            raise RuntimeError("step after done")
        if a not in self.legal_actions():
            raise ValueError(f"illegal action {a}")
        p = self.player
        if self.phase == "SETUP":                                   # R5
            t = a - PLACE_ACTIVE
            self.hands[p].remove(t)
            self.board[p][0] = [t, int(CARDS_HP[t]), 0]
            if p == 0:
                self.player = 1
            else:
                self._start_turn(0)
        elif self.phase == "PROMOTE":                               # R17
            j = a - PROMOTE
            self.board[p][0], self.board[p][1 + j] = self.board[p][1 + j], None
            self._start_turn(p)
        elif a == END_TURN:                                         # R18
            self._start_turn(1 - p)
        elif a == ATTACK:
            self._attack(p)
        elif ATTACH <= a < ATTACH + 3:
            self.hands[p].remove(ENERGY)
            self.board[p][a - ATTACH][2] += 1
            self.attached = True
        elif PLAY_CREATURE <= a < PLAY_CREATURE + 4:
            t = a - PLAY_CREATURE
            self.hands[p].remove(t)
            slot = 1 if self.board[p][1] is None else 2
            self.board[p][slot] = [t, int(CARDS_HP[t]), 0]
        elif a == PLAY_DRAW:
            self.hands[p].remove(STUDY)
            self.discard[p].append(STUDY)
            for _ in range(min(2, len(self.decks[p]))):
                self.hands[p].append(self.decks[p].pop(0))
            self.support = True
        else:                                                       # PLAY_HEAL
            j = a - PLAY_HEAL
            self.hands[p].remove(POTION)
            self.discard[p].append(POTION)
            cr = self.board[p][j]
            cr[1] = min(int(CARDS_HP[cr[0]]), cr[1] + 3)
            self.support = True
        rewards = np.zeros(2)
        if self.done:                                               # R20
            rewards[self.winner], rewards[1 - self.winner] = 1.0, -1.0
        return rewards, self.done, {"mulligans": list(self.mulligans)}

    def _attack(self, p: int) -> None:
        q = 1 - p
        target = self.board[q][0]
        target[1] -= int(CARDS_DMG[self.board[p][0][0]])            # R13: energies are not spent
        if target[1] > 0:
            self._start_turn(q)
            return
        self.discard[q] += [target[0]] + [ENERGY] * target[2]       # R14
        self.board[q][0] = None
        self.hands[p].append(self.prizes[p].pop(0))
        self.prizes_taken[p] += 1
        if self.prizes_taken[p] >= 3:                               # R15
            self._win(p)
        elif self._n_bench(q) == 0:                                 # R16
            self._win(p)
        else:                                                       # R17
            self.phase, self.player = "PROMOTE", q


def _revealed(core: CartasCore, p: int) -> np.ndarray:
    """Copies per card type of player p that everybody has seen: board creatures, attached energies, discard."""
    cnt = np.zeros(N_TYPES, np.int64)
    for slot in core.board[p]:
        if slot is not None:
            cnt[slot[0]] += 1
            cnt[ENERGY] += slot[2]
    for card in core.discard[p]:
        cnt[card] += 1
    return cnt


def rival_bounds(env, player: int) -> tuple[np.ndarray, np.ndarray]:
    """R23: (lower, upper) int64[7] for the rival of `player`.

    lower[t] = copies of type t the rival has revealed; upper[t] = max(0, MAXC[t] - lower[t]) is an upper
    bound on the copies not yet revealed (in hand, deck or prizes)."""
    core = getattr(env, "core", env)
    lower = _revealed(core, 1 - player)
    return lower, np.maximum(0, MAXC - lower)


class CartasEnv:
    """MiniCartas with the option-list interface (`rlf.core.Env`)."""

    def __init__(self, cfg: CartasConfig | None = None):
        self.cfg = cfg or CartasConfig()
        self.core = CartasCore(self.cfg)
        self.spec = EnvSpec(name="cartas", n_players=2, state_dim=12, option_dim=19, entity_dim=26,
                            max_options=13, max_entities=8, n_actions=N_ACTIONS, max_steps=120)
        self.seed = None
        self._actions: list[int] = []

    def reset(self, seed: int | None = None):
        self.core.reset(seed)
        self.seed = self.core.seed
        return self._obs()

    def step(self, choice: int):
        if self.core.done:
            raise RuntimeError("step after done")
        if not 0 <= choice < len(self._actions):
            raise IndexError(choice)
        rewards, done, info = self.core.step_action(self._actions[choice])
        if done:
            info = dict(info, winner=self.core.winner)
        return Step(self._obs(), rewards, done, info)

    def state_key(self) -> tuple:
        raise NotImplementedError("MiniCartas is not tabular")

    # ---- observation -------------------------------------------------------------------------
    def _unseen_own(self, p: int) -> np.ndarray:
        """Own deck + prizes: the deck list minus every card already seen (hand, board, attached, discard)."""
        core = self.core
        cnt = np.array(ARCHETYPES[core.deck_names[p]], np.int64) - _revealed(core, p)
        for card in core.hands[p]:
            cnt[card] -= 1
        return cnt

    def _obs(self):
        core = self.core
        p = core.player if not core.done else 0
        q = 1 - p
        acts = [] if core.done else core.legal_actions()
        self._actions = acts
        board = core.board
        hand = np.bincount(core.hands[p], minlength=N_TYPES)
        main = core.phase == "MAIN"
        state = np.array([len(core.prizes[p]) / 3, len(core.prizes[q]) / 3, len(core.hands[p]) / 20,
                          len(core.hands[q]) / 20, len(core.decks[p]) / 20, len(core.decks[q]) / 20,
                          float(core.attached and main), float(core.support and main),
                          core.phase == "SETUP", main, core.phase == "PROMOTE", p == 0], np.float32)
        ent = np.zeros((8, 26), np.float32)
        mask = np.zeros(8, bool)
        for side, who in ((0, p), (3, q)):
            for j in range(3):
                slot = board[who][j]
                if slot is None:
                    continue
                row = ent[side + j]
                row[0], row[3], row[4], row[5] = 1, 1, who == p, j == 0
                row[6 + slot[0]] = 1
                row[10], row[11] = slot[1] / 8, slot[2] / 3
                mask[side + j] = True
        lower, upper = rival_bounds(self, p)
        ent[6, 1], ent[6, 3], ent[6, 4] = 1, 1, 1
        ent[6, 12:19], ent[6, 19:26] = hand / 8, self._unseen_own(p) / 8
        ent[7, 2], ent[7, 3] = 1, 1
        ent[7, 12:19], ent[7, 19:26] = lower / 8, upper / 8
        mask[6] = mask[7] = True
        n = len(acts)
        opts = np.zeros((n, 19), np.float32)
        refs = -np.ones((n, 2), np.int64)
        for i, a in enumerate(acts):
            o = opts[i]
            if a == END_TURN:
                o[0] = 1
            elif a == ATTACK:
                t = board[p][0][0]
                o[1], o[8 + t], o[18] = 1, 1, CARDS_DMG[t] / 8
                refs[i] = (0, 3)
            elif ATTACH <= a < ATTACH + 3:
                j = a - ATTACH
                o[2], o[8 + ENERGY], o[15 + j] = 1, 1, 1
                refs[i] = (j, -1)
            elif PLAY_CREATURE <= a < PLAY_CREATURE + 4:
                o[3], o[8 + a - PLAY_CREATURE] = 1, 1
            elif a == PLAY_DRAW:
                o[4], o[8 + STUDY] = 1, 1
            elif PLAY_HEAL <= a < PLAY_HEAL + 3:
                j = a - PLAY_HEAL
                o[5], o[8 + POTION], o[15 + j] = 1, 1, 1
                refs[i] = (j, -1)
            elif PLACE_ACTIVE <= a < PLACE_ACTIVE + 4:
                o[6], o[8 + a - PLACE_ACTIVE] = 1, 1
            else:                                                   # PROMOTE
                j = a - PROMOTE
                o[7], o[8 + board[p][1 + j][0]], o[15 + 1 + j] = 1, 1, 1
                refs[i] = (1 + j, -1)
        return Obs(state=state, options=opts, option_ids=np.array(acts, np.int64), option_refs=refs,
                   entities=ent, entity_mask=mask, player=p)
