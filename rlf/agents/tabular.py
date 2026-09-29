"""Tabular agents: value iteration (known model) and Q-learning (model-free)."""
from __future__ import annotations

import numpy as np

from rlf.envs.despacho import MDP


def value_iteration(mdp: MDP, gamma: float = 1.0, tol: float = 1e-10):
    """Bellman-optimality sweeps until max|dV| < tol. Returns (V [nS], Q [nS, nA], policy [nS]).

    `policy[s]` is the global action id (lowest id on ties), -1 in terminal states, where V = 0.
    """
    nS, nA = mdp.legal.shape
    flat = mdp.src * nA + mdp.act
    V = np.zeros(nS)
    for _ in range(100_000):
        EV = np.bincount(flat, weights=mdp.prob * V[mdp.dst], minlength=nS * nA).reshape(nS, nA)
        Q = np.where(mdp.legal, mdp.R + gamma * EV, -np.inf)
        V_new = np.where(mdp.terminal, 0.0, Q.max(1))
        delta = np.abs(V_new - V).max()
        V = V_new
        if delta < tol:
            break
    EV = np.bincount(flat, weights=mdp.prob * V[mdp.dst], minlength=nS * nA).reshape(nS, nA)
    Q = np.where(mdp.legal, mdp.R + gamma * EV, 0.0)
    policy = np.where(mdp.terminal, -1, np.where(mdp.legal, Q, -np.inf).argmax(1)).astype(np.int64)
    return V, Q, policy


class QLearning:
    """Tabular Q-learning with epsilon-greedy exploration restricted to the legal options.

    `Q` maps `state_key` to a float64 array over the global action ids; `option_ids` of each observation
    tell which entries are legal.
    """

    def __init__(self, n_actions: int, alpha: float = 0.1, gamma: float = 1.0, eps: float = 0.1, seed: int = 0):
        self.n_actions, self.alpha, self.gamma, self.eps = n_actions, alpha, gamma, eps
        self.rng = np.random.default_rng(seed)
        self.Q: dict = {}

    def values(self, key) -> np.ndarray:
        q = self.Q.get(key)
        if q is None:
            q = self.Q[key] = np.zeros(self.n_actions)
        return q

    def act(self, key, option_ids: np.ndarray, eps: float | None = None) -> int:
        """Index (not id) of the chosen option; ties go to the lowest index."""
        eps = self.eps if eps is None else eps
        if eps > 0 and self.rng.random() < eps:
            return int(self.rng.integers(len(option_ids)))
        return int(np.argmax(self.values(key)[option_ids]))

    def update(self, key, action_id: int, reward: float, next_key, next_option_ids: np.ndarray, done: bool) -> None:
        q = self.values(key)
        target = reward if done else reward + self.gamma * self.values(next_key)[next_option_ids].max()
        q[action_id] += self.alpha * (target - q[action_id])

    def train(self, env, n_episodes: int, seed: int = 0, eps_fn=None) -> np.ndarray:
        """Episodes with training seeds `derive_seed(seed, 0, e)`; returns the (undiscounted) return of each."""
        from rlf.core import derive_seed
        returns = np.zeros(n_episodes)
        for e in range(n_episodes):
            eps = eps_fn(e) if eps_fn else self.eps
            obs = env.reset(derive_seed(seed, 0, e))
            key, done = env.state_key(), False
            while not done:
                j = self.act(key, obs.option_ids, eps)
                st = env.step(j)
                key2 = env.state_key()
                self.update(key, int(obs.option_ids[j]), float(st.rewards[0]), key2, st.obs.option_ids, st.done)
                returns[e] += st.rewards[0]
                key, obs, done = key2, st.obs, st.done
        return returns

    def evaluate(self, make_env, n_episodes: int, seed_base: int | None = None) -> np.ndarray:
        """Undiscounted returns of the greedy policy (eps = 0) on evaluation seeds `seed_base + i`."""
        from rlf.core import EVAL_SEED_BASE
        seed_base = EVAL_SEED_BASE if seed_base is None else seed_base
        env = make_env()
        out = np.zeros(n_episodes)
        for i in range(n_episodes):
            obs = env.reset(seed_base + i)
            done = False
            while not done:
                st = env.step(self.act(env.state_key(), obs.option_ids, 0.0))
                out[i] += st.rewards[0]
                obs, done = st.obs, st.done
        return out
