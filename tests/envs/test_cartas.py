"""MiniCartas rules R1-R24 (design.md section 4.3)."""
import numpy as np
import pytest

from rlf.envs.cartas import (ARCHETYPES, CARDS_HP, CARDS_DMG, CARDS_COST, N_TYPES, CartasConfig, CartasCore,
                             PLACE_ACTIVE)

CREATURES = (0, 1, 2, 3)


def make(deck0="aggro", deck1="aggro", seed=0):
    c = CartasCore(CartasConfig(deck0, deck1))
    c.reset(seed)
    return c


def total_counts(c, p):
    cnt = np.zeros(N_TYPES, int)
    for card in c.decks[p] + c.hands[p] + c.prizes[p] + c.discard[p]:
        cnt[card] += 1
    for slot in c.board[p]:
        if slot is not None:
            cnt[slot[0]] += 1
            cnt[4] += slot[2]
    return cnt


def test_catalog_values():
    assert list(CARDS_HP[:4]) == [3, 5, 8, 6]
    assert list(CARDS_DMG[:4]) == [2, 3, 2, 5]
    assert list(CARDS_COST[:4]) == [1, 2, 2, 3]
    assert ARCHETYPES == {"aggro": [6, 3, 0, 0, 8, 2, 1], "tank": [0, 2, 5, 0, 8, 1, 4],
                          "dragon": [2, 3, 0, 3, 8, 3, 1]}


def test_R1_decks_have_the_archetype_counts():
    for name, counts in ARCHETYPES.items():
        assert sum(counts) == 20
        c = make(name, name, seed=3)
        assert (total_counts(c, 0) == counts).all() and (total_counts(c, 1) == counts).all()


def test_R1_random_deck_choice_is_uniform_over_archetypes():
    picks = {k: 0 for k in ARCHETYPES}
    for seed in range(1500):
        c = make("random", "aggro", seed)
        for name, counts in ARCHETYPES.items():
            if (total_counts(c, 0) == counts).all():
                picks[name] += 1
    assert sum(picks.values()) == 1500
    assert all(abs(v / 1500 - 1 / 3) < 0.05 for v in picks.values())


def test_R2_same_seed_same_order_and_p0_shuffled_first():
    a, b = make(seed=5), make(seed=5)
    assert a.decks == b.decks and a.hands == b.hands and a.prizes == b.prizes
    assert make(seed=5).decks != make(seed=6).decks
    # player 0's whole setup does not depend on player 1's deck: player 0 is shuffled first
    x, y = make("aggro", "tank", 9), make("aggro", "dragon", 9)
    assert x.hands[0] == y.hands[0] and x.decks[0] == y.decks[0] and x.prizes[0] == y.prizes[0]


def test_R3_opening_hand_has_five_cards_and_a_creature_with_mulligan():
    mull = 0
    n = 2000
    for seed in range(n):
        c = make("tank", "tank", seed)
        for p in (0, 1):
            assert len(c.hands[p]) == 5 and any(card in CREATURES for card in c.hands[p])
        mull += c.mulligans[0] > 0
    # P(no creature in 5 cards) = C(13,5)/C(20,5) = 0.083
    assert abs(mull / n - 1287 / 15504) < 0.02


def test_R4_prizes_and_deck_size():
    c = make(seed=1)
    for p in (0, 1):
        assert len(c.prizes[p]) == 3 and len(c.decks[p]) == 12 and len(c.hands[p]) == 5


def test_R5_placement_order_and_result():
    c = make("dragon", "dragon", 4)
    assert c.phase == "SETUP" and c.player == 0
    types0 = sorted({card for card in c.hands[0] if card in CREATURES})
    assert c.legal_actions() == [PLACE_ACTIVE + t for t in types0] == [13 + t for t in types0]
    a = c.legal_actions()[0]
    t0 = a - 13
    c.step_action(a)
    assert c.board[0][0] == [t0, int(CARDS_HP[t0]), 0] and len(c.hands[0]) == 4
    assert c.phase == "SETUP" and c.player == 1
    types1 = sorted({card for card in c.hands[1] if card in CREATURES})
    assert c.legal_actions() == [13 + t for t in types1]
    c.step_action(c.legal_actions()[-1])
    assert c.board[1][0] is not None and len(c.hands[1]) == 4
    assert c.phase == "MAIN" and c.player == 0
    assert c.board[0][1] is None and c.board[0][2] is None
