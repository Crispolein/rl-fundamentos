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
