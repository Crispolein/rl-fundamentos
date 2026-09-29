"""MiniDespacho: assign pending orders (location + time window) to trucks, or wait.

`DespachoCore` holds the dynamics without any observation encoding; the environment that wraps it
with observations lives further down (`DespachoEnv`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rlf.core import EnvSpec, Obs, Step


@dataclass(frozen=True)
class DespachoConfig:
    width: int
    height: int
    n_trucks: int
    capacity: int            # units per truck for the whole episode
    max_pending: int         # M order slots
    horizon: int             # T
    p_arrival: float         # probability that one order arrives at each tick
    qty_range: tuple
    slack_range: tuple
    dist_w: float = 1.0
    late_w: float = 2.0

    @property
    def U(self) -> float:
        """Cost of an unserved order: the worst "-distance -lateness" of one order."""
        return self.dist_w * (self.width - 1 + self.height - 1) + self.late_w * self.horizon

    @property
    def max_steps(self) -> int:
        return 2 * self.horizon

    @property
    def n_actions(self) -> int:
        return self.max_pending * self.n_trucks + 1

    @classmethod
    def from_size(cls, name: str) -> "DespachoConfig":
        return cls(**_SIZES[name])


_SIZES = {
    "diminuto": dict(width=3, height=1, n_trucks=2, capacity=2, max_pending=1, horizon=6,
                     p_arrival=0.5, qty_range=(1, 1), slack_range=(2, 2)),
    "mediano": dict(width=5, height=5, n_trucks=3, capacity=12, max_pending=4, horizon=30,
                    p_arrival=0.5, qty_range=(1, 3), slack_range=(4, 8)),
    "grande": dict(width=8, height=8, n_trucks=6, capacity=12, max_pending=12, horizon=60,
                   p_arrival=0.8, qty_range=(1, 3), slack_range=(4, 10)),
}


class DespachoCore:
    """Dynamics D1-D9, D11, D13. Actions are global ids: `s*K + k` assigns slot s to truck k; `M*K` waits."""

    def __init__(self, cfg: DespachoConfig):
        self.cfg = cfg
        self.wait_id = cfg.max_pending * cfg.n_trucks
        self.done = True
        self.rng = np.random.default_rng(0)

    # ---- setup -------------------------------------------------------------------------------
    def _sample_schedule(self, rng) -> list:
        """Whole-episode arrival calendar: entry t is None or ((x, y), qty, slack)."""
        c = self.cfg
        T = c.horizon
        arrives = rng.random(T) < c.p_arrival
        cell = rng.integers(0, c.width * c.height, T)
        qty = rng.integers(c.qty_range[0], c.qty_range[1] + 1, T)
        slack = rng.integers(c.slack_range[0], c.slack_range[1] + 1, T)
        return [((int(cell[t] % c.width), int(cell[t] // c.width)), int(qty[t]), int(slack[t]))
                if arrives[t] else None for t in range(T)]

    def _clear_counters(self):
        self.dist_cost = 0.0
        self.late_cost = 0.0
        self.n_unserved = 0

    def reset(self, seed: int | None = None) -> None:
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0] % 10_000_000)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.schedule = self._sample_schedule(self.rng)
        c = self.cfg
        self.t = 0
        self.pend = [None] * c.max_pending
        self.trucks = [[0, 0, c.capacity, 0] for _ in range(c.n_trucks)]    # x, y, cap, busy_until (D1)
        self.done = False
        self._clear_counters()
        self._arrive(0)

    def set_state(self, key: tuple, seed: int | None = None) -> None:
        """Force a state given by `state_key()` and resample the arrivals of the remaining ticks."""
        t, pend, trucks = key
        c = self.cfg
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        self.rng = rng
        sched = self._sample_schedule(rng)
        self.schedule = [None if tt <= t else sched[tt] for tt in range(c.horizon)]
        self.t = t
        self.pend = [None if p is None else [p[0], p[1], p[2], t + p[3]] for p in pend]
        self.trucks = [[x, y, cap, t + rem] for (x, y, cap, rem) in trucks]
        self.done = t >= c.horizon
        self._clear_counters()

    # ---- dynamics ----------------------------------------------------------------------------
    def _arrive(self, t: int) -> float:
        """D2: process the arrival of tick t; returns the reward (-U if rejected)."""
        if t >= self.cfg.horizon or self.schedule[t] is None:
            return 0.0
        (x, y), qty, slack = self.schedule[t]
        for s, p in enumerate(self.pend):
            if p is None:
                self.pend[s] = [x, y, qty, t + slack]
                return 0.0
        self.n_unserved += 1
        return -self.cfg.U

    def _free(self, k: int) -> bool:
        return self.trucks[k][3] <= self.t                       # D3

    def legal_actions(self) -> list[int]:
        """Legal action ids in ascending order (D4, D5, D10)."""
        acts = []
        K = self.cfg.n_trucks
        for s, p in enumerate(self.pend):
            if p is None:
                continue
            for k in range(K):
                if self._free(k) and self.trucks[k][2] >= p[2]:
                    acts.append(s * K + k)
        acts.append(self.wait_id)
        return acts

    def dist(self, s: int, k: int) -> int:
        p, tr = self.pend[s], self.trucks[k]
        return abs(tr[0] - p[0]) + abs(tr[1] - p[1])

    def lateness(self, s: int, k: int) -> int:
        return min(self.cfg.horizon, max(0, self.t + self.dist(s, k) - self.pend[s][3]))

    def step_action(self, a: int) -> tuple[float, bool, dict]:
        if self.done:
            raise RuntimeError("step after done")
        if a not in self.legal_actions():
            raise ValueError(f"illegal action {a}")
        c = self.cfg
        if a == self.wait_id:                                     # D8
            self.t += 1
            reward = self._arrive(self.t)
            if self.t >= c.horizon:                               # D9
                n = sum(p is not None for p in self.pend)
                reward -= n * c.U
                self.n_unserved += n
                self.pend = [None] * c.max_pending
                self.done = True
        else:                                                     # D6, D7
            s, k = divmod(a, c.n_trucks)
            d, late = self.dist(s, k), self.lateness(s, k)
            reward = -c.dist_w * d - c.late_w * late
            self.dist_cost += c.dist_w * d
            self.late_cost += c.late_w * late
            tr = self.trucks[k]
            tr[0], tr[1] = self.pend[s][0], self.pend[s][1]
            tr[2] -= self.pend[s][2]
            tr[3] = self.t + d
            self.pend[s] = None
        return reward, self.done, {"cost_breakdown": self.cost_breakdown()}

    def cost_breakdown(self) -> dict:
        return {"distance": self.dist_cost, "lateness": self.late_cost,
                "unserved": self.cfg.U * self.n_unserved}

    def state_key(self) -> tuple:
        """D12: (t, pend, trucks) with pend[s] = None or (x, y, qty, due - t)."""
        t = self.t
        pend = tuple(None if p is None else (p[0], p[1], p[2], p[3] - t) for p in self.pend)
        trucks = tuple((x, y, cap, max(0, busy - t)) for (x, y, cap, busy) in self.trucks)
        return (t, pend, trucks)


class DespachoEnv:
    """MiniDespacho with the option-list interface (`rlf.core.Env`)."""

    def __init__(self, cfg: DespachoConfig):
        self.cfg = cfg
        self.core = DespachoCore(cfg)
        K, M = cfg.n_trucks, cfg.max_pending
        self.spec = EnvSpec(name="despacho", n_players=1, state_dim=3, option_dim=6, entity_dim=6,
                            max_options=M * K + 1, max_entities=K + M, n_actions=M * K + 1,
                            max_steps=cfg.max_steps)
        self.seed = None
        self._actions: list[int] = []
        self._nx = max(1, cfg.width - 1)
        self._ny = max(1, cfg.height - 1)
        self._nd = max(1, cfg.width + cfg.height - 2)

    @classmethod
    def from_size(cls, name: str) -> "DespachoEnv":
        return cls(DespachoConfig.from_size(name))

    def reset(self, seed: int | None = None):
        self.core.reset(seed)
        self.seed = self.core.seed
        return self._obs()

    def state_key(self) -> tuple:
        return self.core.state_key()

    def step(self, choice: int):
        if not 0 <= choice < len(self._actions):
            if self.core.done:
                raise RuntimeError("step after done")
            raise IndexError(choice)
        reward, done, info = self.core.step_action(self._actions[choice])
        return Step(self._obs(), np.array([reward]), done, info)

    def _obs(self):
        c, core = self.cfg, self.core
        K, M, T, Q = c.n_trucks, c.max_pending, c.horizon, c.capacity
        t = core.t
        pend, trucks = core.pend, core.trucks
        opts, ids, refs = [], [], []
        self._actions = [] if core.done else core.legal_actions()
        for a in self._actions:
            ids.append(a)
            if a == core.wait_id:
                opts.append([1, 0, 0, 0, 0, 0])
                refs.append([-1, -1])
                continue
            s, k = divmod(a, K)
            p = pend[s]
            opts.append([0, core.dist(s, k) / self._nd, core.lateness(s, k) / T,
                         (trucks[k][2] - p[2]) / Q, min(1.0, max(-1.0, (p[3] - t) / T)), p[2] / Q])
            refs.append([K + s, k])
        n = len(ids)
        ent = np.zeros((K + M, 6), np.float32)
        mask = np.zeros(K + M, bool)
        n_free = 0
        for k, (x, y, cap, busy) in enumerate(trucks):
            free = busy <= t
            n_free += free
            ent[k] = [0, x / self._nx, y / self._ny, cap / Q, max(0, busy - t) / T, 1.0 if free else 0.0]
            mask[k] = True
        n_pend = 0
        for s, p in enumerate(pend):
            if p is None:
                continue
            n_pend += 1
            ent[K + s] = [1, p[0] / self._nx, p[1] / self._ny, p[2] / Q,
                          min(1.0, max(-1.0, (p[3] - t) / T)), 1.0]
            mask[K + s] = True
        state = np.array([t / T, n_pend / M, n_free / K], np.float32)
        return Obs(state=state, options=np.array(opts, np.float32).reshape(n, 6),
                   option_ids=np.array(ids, np.int64), option_refs=np.array(refs, np.int64).reshape(n, 2),
                   entities=ent, entity_mask=mask, player=0)
