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


# ---- T10: turns, actions, KO, promotion, end of game and reward -------------------------------
from rlf.envs.cartas import (ATTACH, ATTACK, END_TURN, PLAY_CREATURE, PLAY_DRAW, PLAY_HEAL,  # noqa: E402
                             PROMOTE)


def mk(hand0=(), hand1=(), board0=None, board1=None, deck0=12, deck1=12, prizes=(3, 3), player=0):
    """Mid-game state in phase MAIN; the decks are filled with Energy cards (type 4)."""
    c = make(seed=0)
    c.hands = [list(hand0), list(hand1)]
    c.board = [board0 or [[0, 3, 0], None, None], board1 or [[0, 3, 0], None, None]]
    c.decks = [[4] * deck0, [4] * deck1]
    c.prizes = [[6] * prizes[0], [6] * prizes[1]]
    c.prizes_taken = [3 - prizes[0], 3 - prizes[1]]
    c.discard = [[], []]
    c.phase, c.player, c.attached, c.support, c.done, c.winner = "MAIN", player, False, False, False, None
    return c


def test_R5_then_R6_first_turn_draws_for_player_0():
    c = make(seed=2)
    c.step_action(c.legal_actions()[0])
    h0, d0 = len(c.hands[0]), len(c.decks[0])
    c.step_action(c.legal_actions()[0])
    assert c.phase == "MAIN" and c.player == 0
    assert len(c.hands[0]) == h0 + 1 and len(c.decks[0]) == d0 - 1


def test_R6_draw_at_turn_start_and_empty_deck_loses():
    c = mk(deck1=5)
    c.step_action(END_TURN)
    assert c.player == 1 and len(c.hands[1]) == 1 and len(c.decks[1]) == 4
    c = mk(deck1=0)
    rewards, done, _ = c.step_action(END_TURN)
    assert done and c.winner == 0 and list(rewards) == [1.0, -1.0]


def test_R8_end_turn_always_legal_and_illegal_raises():
    c = mk()
    assert c.legal_actions() == [END_TURN]
    with pytest.raises(ValueError):
        c.step_action(ATTACK)


def test_R9_play_creature():
    c = mk(hand0=[1, 1, 3])
    assert c.legal_actions() == [END_TURN, PLAY_CREATURE + 1, PLAY_CREATURE + 3]
    c.step_action(PLAY_CREATURE + 1)
    assert c.board[0][1] == [1, 5, 0] and c.hands[0] == [1, 3]
    c.step_action(PLAY_CREATURE + 3)
    assert c.board[0][2] == [3, 6, 0]
    assert PLAY_CREATURE + 1 not in c.legal_actions()                # bench full


def test_R10_attach_once_per_turn_one_option_per_creature():
    c = mk(hand0=[4, 4], board0=[[0, 3, 0], [1, 5, 0], None])
    assert c.legal_actions() == [END_TURN, ATTACH + 0, ATTACH + 1]
    c.step_action(ATTACH + 1)
    assert c.board[0][1][2] == 1 and c.hands[0] == [4]
    assert not any(ATTACH <= a < ATTACH + 3 for a in c.legal_actions())      # once per turn


def test_R11_play_draw():
    c = mk(hand0=[5, 5], deck0=1)
    c.step_action(PLAY_DRAW)
    assert len(c.decks[0]) == 0 and len(c.hands[0]) == 2 and c.discard[0] == [5]   # drew only 1
    assert PLAY_DRAW not in c.legal_actions()                        # support already played
    c = mk(hand0=[5], deck0=0)
    c.step_action(PLAY_DRAW)                                         # empty deck by effect: no loss
    assert not c.done and c.hands[0] == []
    c = mk(hand0=[5], deck0=10)
    c.step_action(PLAY_DRAW)
    assert len(c.hands[0]) == 2


def test_R12_play_heal():
    c = mk(hand0=[6, 6], board0=[[1, 2, 0], [2, 8, 0], [0, 1, 0]])
    assert c.legal_actions() == [END_TURN, PLAY_HEAL + 0, PLAY_HEAL + 2]     # slot 1 is at full HP
    c.step_action(PLAY_HEAL + 2)
    assert c.board[0][2][1] == 3 and c.discard[0] == [6]                     # healed to max 3 (cap)
    assert not any(PLAY_HEAL <= a < PLAY_HEAL + 3 for a in c.legal_actions())      # support used
    c = mk(hand0=[6], board0=[[3, 1, 0], None, None])
    c.step_action(PLAY_HEAL + 0)
    assert c.board[0][0][1] == 4                                             # +3


def test_R11_R12_share_the_support_limit():
    c = mk(hand0=[5, 6], board0=[[1, 2, 0], None, None])
    c.step_action(PLAY_DRAW)
    assert PLAY_HEAL not in c.legal_actions()


def test_R13_attack_legality_damage_energy_kept_and_turn_ends():
    c = mk(board0=[[1, 5, 1], None, None], board1=[[2, 8, 0], [0, 3, 0], None])
    assert ATTACK not in c.legal_actions()                                   # Lobo needs 2 energies
    c = mk(board0=[[1, 5, 2], None, None], board1=[[2, 8, 0], [0, 3, 0], None])
    assert ATTACK in c.legal_actions()
    c.step_action(ATTACK)
    assert c.board[1][0][1] == 5 and c.board[0][0][2] == 2                   # 3 damage, energies kept
    assert c.player == 1 and c.phase == "MAIN" and len(c.hands[1]) == 1     # turn over, rival drew


def test_R14_ko_moves_to_discard_and_attacker_takes_prize():
    c = mk(board0=[[1, 5, 2], None, None], board1=[[0, 3, 2], [0, 3, 0], None], prizes=(3, 3))
    c.step_action(ATTACK)
    assert c.board[1][0] is None
    assert c.discard[1] == [0, 4, 4]
    assert c.hands[0] == [6] and len(c.prizes[0]) == 2 and c.prizes_taken[0] == 1


def test_R15_third_prize_wins():
    c = mk(board0=[[1, 5, 2], None, None], board1=[[0, 3, 0], [0, 3, 0], None], prizes=(1, 3))
    rewards, done, _ = c.step_action(ATTACK)
    assert done and c.winner == 0 and list(rewards) == [1.0, -1.0]


def test_R16_wipeout_wins_with_fewer_than_three_prizes():
    c = mk(board0=[[1, 5, 2], None, None], board1=[[0, 3, 0], None, None], prizes=(3, 3))
    rewards, done, _ = c.step_action(ATTACK)
    assert done and c.winner == 0 and c.prizes_taken[0] == 1


def test_R17_promotion_then_rival_turn():
    c = mk(board0=[[1, 5, 2], None, None], board1=[[0, 3, 0], [2, 8, 0], [3, 6, 0]])
    c.step_action(ATTACK)
    assert c.phase == "PROMOTE" and c.player == 1
    assert c.legal_actions() == [PROMOTE + 0, PROMOTE + 1]
    h = len(c.hands[1])
    c.step_action(PROMOTE + 1)
    assert c.board[1] == [[3, 6, 0], [2, 8, 0], None]                        # bench slot freed
    assert c.phase == "MAIN" and c.player == 1 and len(c.hands[1]) == h + 1  # R6: rival draws
    c = mk(board0=[[1, 5, 2], None, None], board1=[[0, 3, 0], None, [2, 8, 0]])
    c.step_action(ATTACK)
    assert c.legal_actions() == [PROMOTE + 1]


def test_R18_end_turn_resets_flags():
    c = mk(hand0=[4, 5], hand1=[])
    c.step_action(ATTACH + 0)
    c.step_action(PLAY_DRAW)
    assert c.attached and c.support
    c.step_action(END_TURN)
    assert c.player == 1 and not c.attached and not c.support


def test_R19_games_end_within_bounds_without_draws():
    rng = np.random.default_rng(0)
    for seed in range(1000):
        c = make("random", "random", seed)
        n, turns0 = 0, 0
        while not c.done:
            r, done, _ = c.step_action(int(rng.choice(c.legal_actions())))
            n += 1
            assert n <= 120
        assert c.winner in (0, 1) and max(c.turns) <= 13


def test_R20_reward_zero_until_the_end():
    rng = np.random.default_rng(1)
    for seed in range(200):
        c = make("random", "random", seed)
        while True:
            r, done, _ = c.step_action(int(rng.choice(c.legal_actions())))
            if done:
                assert sorted(r) == [-1.0, 1.0] and r[c.winner] == 1.0
                break
            assert (r == 0).all()
