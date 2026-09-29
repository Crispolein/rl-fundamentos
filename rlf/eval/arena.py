"""Evaluation arena: fixed-seed evaluation, seat-swapped matches, round-robin and Elo."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rlf.core import EVAL_SEED_BASE, VecEnv
from rlf.eval.stats import Estimate, proportion_ci


def evaluate_policy(make_env, policy, n_episodes: int, n_envs: int = 100,
                    seed_base: int = EVAL_SEED_BASE) -> np.ndarray:
    """Return of player 0 in each episode; episode i uses seed `seed_base + i` (sorted by seed)."""
    n_envs = min(n_envs, n_episodes)
    vec = VecEnv(make_env, n_envs, seed_fn=lambda b, e: seed_base + e * n_envs + b)
    obs = vec.reset()
    returns = np.zeros(n_episodes)
    running = np.zeros(n_envs)
    finished = 0
    while finished < n_episodes:
        st = vec.step(policy.act(obs))
        running += st.rewards[:, 0]
        for b in np.flatnonzero(st.done):
            i = st.info[b]["episode_id"]
            if i < n_episodes:
                returns[i] = running[b]
                finished += 1
            running[b] = 0.0
        obs = st.obs
    return returns


@dataclass(frozen=True)
class MatchResult:
    wins_a: int
    wins_b: int
    n: int
    win_rate_a: Estimate    # Wilson over wins_a / n
    seat0_wins: int         # games won by seat 0, over both seat assignments


def play_games(make_env, policy_a, policy_b, n_games: int, n_envs: int = 100,
               seed_base: int = EVAL_SEED_BASE) -> MatchResult:
    """Game i uses seed `seed_base + i // 2`; A takes seat 0 in even games and seat 1 in odd games."""
    if n_games % 2:
        raise ValueError("n_games must be even")
    n_envs = min(n_envs, n_games)
    while n_games % n_envs or n_envs % 2:       # every env plays the same number of games; parity = seat of A
        n_envs -= 1
    quota = n_games // n_envs
    vec = VecEnv(make_env, n_envs, seed_fn=lambda b, e: seed_base + (e * n_envs + b) // 2)
    seat_a = np.arange(n_envs) % 2              # seat played by A in env b
    obs = vec.reset()
    played = np.zeros(n_envs, int)
    wins_a = seat0 = 0
    while (played < quota).any():
        is_a = obs.player == seat_a
        choices = np.zeros(n_envs, np.int64)
        if is_a.any():
            choices[is_a] = policy_a.act(obs.select(is_a))
        if (~is_a).any():
            choices[~is_a] = policy_b.act(obs.select(~is_a))
        st = vec.step(choices)
        for b in np.flatnonzero(st.done):
            if played[b] >= quota:
                continue
            played[b] += 1
            r = st.rewards[b]
            assert r[0] != r[1], "draws are not supported"
            winner = int(r[1] > r[0])
            wins_a += winner == seat_a[b]
            seat0 += winner == 0
        obs = st.obs
    return MatchResult(int(wins_a), n_games - int(wins_a), n_games,
                       proportion_ci(int(wins_a), n_games), int(seat0))


def round_robin(make_env, entrants: dict, n_games: int, n_envs: int = 100,
                seed_base: int = EVAL_SEED_BASE) -> np.ndarray:
    """Win rate of the row entrant against the column entrant; NaN on the diagonal."""
    names = list(entrants)
    E = len(names)
    m = np.full((E, E), np.nan)
    for i in range(E):
        for j in range(i + 1, E):
            r = play_games(make_env, entrants[names[i]], entrants[names[j]], n_games, n_envs, seed_base)
            m[i, j] = r.wins_a / n_games
            m[j, i] = r.wins_b / n_games
    return m


def elo_ratings(win_matrix: np.ndarray, n_games: int, anchor: int = 0,
                anchor_rating: float = 1000.0) -> np.ndarray:
    """Maximum-likelihood ratings under P(i beats j) = 1 / (1 + 10^((r_j - r_i) / 400)), by Newton."""
    W = np.clip(np.asarray(win_matrix, dtype=np.float64), 1e-3, 1 - 1e-3)
    E = len(W)
    off = ~np.eye(E, dtype=bool) & ~np.isnan(np.asarray(win_matrix))
    W = np.where(off, W, 0.0)
    scale = np.log(10) / 400
    theta = np.zeros(E)
    keep = np.array([i for i in range(E) if i != anchor])
    for _ in range(100):
        p = 1 / (1 + np.exp(-(theta[:, None] - theta[None, :])))
        p = np.where(off, p, 0.0)
        grad = n_games * ((W - p) * off).sum(1)
        w = n_games * p * (1 - p) * off
        H = -w.copy()
        H[np.diag_indices(E)] = w.sum(1)              # negative Hessian (positive semi-definite)
        step = np.zeros(E)
        step[keep] = np.linalg.solve(H[np.ix_(keep, keep)], grad[keep])
        theta += step
        if np.abs(step).max() < 1e-9:
            break
    return anchor_rating + (theta - theta[anchor]) / scale
