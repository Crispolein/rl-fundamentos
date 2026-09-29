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


# ---- T11: environment with observation, canonical ids, rival_bounds, contract -----------------
from rlf.envs.cartas import MAXC, CartasEnv, rival_bounds  # noqa: E402


def play_random(env, n_steps, seed):
    rng = np.random.default_rng(seed)
    obs = env.reset(seed)
    for _ in range(n_steps):
        st = env.step(int(rng.integers(obs.n_options)))
        if st.done:
            return None
        obs = st.obs
    return obs


def obs_equal(a, b):
    return a.player == b.player and all(np.array_equal(getattr(a, f), getattr(b, f)) for f in
            ("state", "options", "option_ids", "option_refs", "entities", "entity_mask"))


def test_spec_and_state_key_not_implemented():
    e = CartasEnv()
    sp = e.spec
    assert (sp.n_players, sp.state_dim, sp.option_dim, sp.entity_dim) == (2, 12, 19, 26)
    assert (sp.max_options, sp.max_entities, sp.n_actions, sp.max_steps) == (13, 8, 19, 120)
    e.reset(0)
    with pytest.raises(NotImplementedError):
        e.state_key()


def test_R22_ids_and_option_features():
    e = CartasEnv()
    e.reset(0)
    e.core.hands[0] = [0, 3, 4, 5, 6]
    e.core.board[0] = [[0, 1, 0], [1, 5, 0], None]
    e.core.phase, e.core.player = "MAIN", 0
    obs = e._obs()
    ids = obs.option_ids.tolist()
    assert ids == [END_TURN, ATTACH + 0, ATTACH + 1, PLAY_CREATURE + 0, PLAY_CREATURE + 3, PLAY_DRAW, PLAY_HEAL + 0]
    o = {i: obs.options[j] for j, i in enumerate(ids)}
    assert o[END_TURN][0] == 1 and o[END_TURN][8:15].sum() == 0
    assert o[ATTACH + 1][2] == 1 and o[ATTACH + 1][8 + 4] == 1 and o[ATTACH + 1][15 + 1] == 1
    assert o[PLAY_CREATURE + 3][3] == 1 and o[PLAY_CREATURE + 3][8 + 3] == 1 and o[PLAY_CREATURE + 3][15:18].sum() == 0
    assert o[PLAY_DRAW][4] == 1 and o[PLAY_DRAW][8 + 5] == 1
    assert o[PLAY_HEAL][5] == 1 and o[PLAY_HEAL][8 + 6] == 1 and o[PLAY_HEAL][15] == 1
    refs = {i: obs.option_refs[j].tolist() for j, i in enumerate(ids)}
    assert refs[END_TURN] == [-1, -1] and refs[ATTACH + 1] == [1, -1] and refs[PLAY_HEAL] == [0, -1]
    assert refs[PLAY_CREATURE + 3] == [-1, -1]
    # ATTACK: active Lobo with 2 energies, dmg 3 -> feature 3/8; refs (0, 3)
    e.core.board[0] = [[1, 5, 2], None, None]
    obs = e._obs()
    j = obs.option_ids.tolist().index(ATTACK)
    assert obs.options[j][1] == 1 and obs.options[j][8 + 1] == 1 and obs.options[j][18] == pytest.approx(3 / 8)
    assert obs.option_refs[j].tolist() == [0, 3]


def test_promote_and_place_features():
    e = CartasEnv()
    obs = e.reset(3)
    assert obs.state[8] == 1 and obs.player == 0                      # phase SETUP
    assert (obs.options[:, 6] == 1).all() and (obs.option_ids >= 13).all()
    e.core.phase, e.core.player = "PROMOTE", 1
    e.core.board[1] = [None, [2, 8, 0], [3, 6, 0]]
    obs = e._obs()
    assert obs.player == 1 and obs.state[10] == 1
    assert obs.option_ids.tolist() == [PROMOTE, PROMOTE + 1]
    assert obs.options[0][7] == 1 and obs.options[0][8 + 2] == 1 and obs.options[0][15 + 1] == 1
    assert obs.options[1][15 + 2] == 1
    assert obs.option_refs.tolist() == [[1, -1], [2, -1]]


def test_state_and_entities_encoding():
    e = CartasEnv()
    e.reset(0)
    c = e.core
    c.phase, c.player, c.attached = "MAIN", 1, True
    c.hands = [[0, 0, 4], [1, 4, 4, 5]]
    c.board = [[[0, 2, 1], None, None], [[1, 5, 2], [2, 8, 0], None]]
    c.discard = [[6], [4, 4]]
    c.decks = [[4] * 10, [4] * 8]
    c.prizes = [[6] * 3, [6] * 2]
    obs = e._obs()
    assert obs.state.tolist() == pytest.approx([2 / 3, 1, 4 / 20, 3 / 20, 8 / 20, 10 / 20, 1, 0, 0, 1, 0, 0])
    ent = obs.entities
    assert obs.entity_mask.tolist() == [True, True, False, True, False, False, True, True]
    assert ent[0][:3].tolist() == [1, 0, 0] and ent[0][3] == 1
    # player 1 decides: "own" rows 0..2 are player 1's creatures
    assert ent[0][4] == 1 and ent[0][5] == 1 and ent[0][6 + 1] == 1 and ent[0][10] == pytest.approx(5 / 8)
    assert ent[0][11] == pytest.approx(2 / 3)
    assert ent[1][6 + 2] == 1 and ent[1][5] == 0
    assert ent[3][4] == 0 and ent[3][5] == 1 and ent[3][6 + 0] == 1 and ent[3][10] == pytest.approx(2 / 8)
    assert (ent[2] == 0).all() and (ent[4] == 0).all()
    assert ent[6][:3].tolist() == [0, 1, 0] and ent[7][:3].tolist() == [0, 0, 1]
    # own hand counts / 8 (hand of player 1 = [1, 4, 4, 5])
    assert ent[6][12:19].tolist() == pytest.approx(np.array([0, 1, 0, 0, 2, 1, 0]) / 8)
    # rival bounds: lower = board creature 0 + 1 energy + discard Potion
    lower, upper = rival_bounds(e, 1)
    assert lower.tolist() == [1, 0, 0, 0, 1, 0, 1]
    assert ent[7][12:19].tolist() == pytest.approx(lower / 8) and ent[7][19:26].tolist() == pytest.approx(upper / 8)


def test_R21_no_information_leak():
    for seed in range(40):
        e = CartasEnv()
        obs = play_random(e, 12 + seed % 7, seed)
        if obs is None or obs.n_options == 0:
            continue
        p = e.core.player
        q = 1 - p
        before = e._obs()
        c = e.core
        c.hands[q] = [4] * len(c.hands[q])
        c.decks[q] = [6] * len(c.decks[q])
        c.prizes[q] = [5] * len(c.prizes[q])
        assert obs_equal(before, e._obs())


def test_R21_own_hand_changes_observation():
    e = CartasEnv()
    obs = play_random(e, 14, 5)
    before = e._obs()
    e.core.hands[e.core.player] = [3, 3, 3]
    assert not np.array_equal(before.entities, e._obs().entities)


def test_R23_bounds_never_violated_1000_games():
    rng = np.random.default_rng(0)
    e = CartasEnv(CartasConfig("random", "random"))
    informative = 0
    for seed in range(1000):
        obs = e.reset(seed)
        done = False
        while not done:
            for pl in (0, 1):
                lo, up = rival_bounds(e, pl)
                total = np.array(ARCHETYPES[e.core.deck_names[1 - pl]])
                assert (lo <= total).all() and ((total - lo) <= up).all()
                assert lo.dtype == np.int64 and up.dtype == np.int64 and (up >= 0).all()
            informative += 1
            st = e.step(int(rng.integers(obs.n_options)))
            obs, done = st.obs, st.done
    assert informative > 1000
    lo, up = rival_bounds(e, 0)
    assert (up == np.maximum(0, MAXC - lo)).all()


def test_R24_determinism_with_mulligans_and_random_decks():
    def run(seed):
        e = CartasEnv(CartasConfig("random", "tank"))
        rng = np.random.default_rng(3)
        obs = e.reset(seed)
        out = [obs.state.copy(), obs.entities.copy(), e.core.mulligans[:]]
        while True:
            st = e.step(int(rng.integers(obs.n_options)))
            out.append((st.rewards.copy(), st.obs.state.copy(), st.obs.option_ids.copy()))
            obs = st.obs
            if st.done:
                return out
    for seed in (1, 2, 3):
        a, b = run(seed), run(seed)
        assert len(a) == len(b)
        assert np.array_equal(a[0], b[0]) and a[2] == b[2]
        assert all(np.array_equal(x[1], y[1]) and np.array_equal(x[2], y[2]) for x, y in zip(a[3:], b[3:]))
