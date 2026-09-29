"""DQN: TD loss with a target network and a replay buffer, epsilon-greedy over the legal options."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from rlf.core import BatchObs, EnvSpec, VecEnv, flatten_obs, get_device
from rlf.nets.mlp import MLP


@dataclass(frozen=True)
class DQNConfig:
    gamma: float = 1.0
    lr: float = 1e-3
    hidden: tuple = (64, 64)
    buffer_size: int = 50_000
    batch_size: int = 64
    learning_starts: int = 1000
    train_ratio: float = 0.25        # gradient steps per environment transition
    target_update: int = 500         # gradient steps between hard copies to the target network
    use_target: bool = True
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.5      # fraction of the training steps over which epsilon decays linearly
    grad_clip: float = 10.0


class ReplayBuffer:
    """Ring buffer of (s, a_id, r, s', done, legal-mask of s')."""

    def __init__(self, capacity: int, obs_dim: int, n_actions: int, seed: int = 0):
        self.cap, self.n, self.pos = capacity, 0, 0
        self.s = np.zeros((capacity, obs_dim), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.s2 = np.zeros((capacity, obs_dim), np.float32)
        self.d = np.zeros(capacity, bool)
        self.m = np.zeros((capacity, n_actions), bool)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n

    def add(self, s, a, r, s2, d, mask) -> None:
        i = self.pos
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i], self.m[i] = s, a, r, s2, d, mask
        self.pos = (self.pos + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.n, batch_size)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx], self.m[idx]


class DQN:
    def __init__(self, spec: EnvSpec, cfg: DQNConfig = DQNConfig(), seed: int = 0):
        assert spec.n_players == 1, "DQN here is single-player"
        self.spec, self.cfg = spec, cfg
        self.device = get_device()
        self.obs_dim = spec.state_dim + spec.max_entities * spec.entity_dim
        torch.manual_seed(seed)
        self.q_net = MLP(self.obs_dim, cfg.hidden, spec.n_actions).to(self.device)
        self.target_net = MLP(self.obs_dim, cfg.hidden, spec.n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.opt = torch.optim.Adam(self.q_net.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size, self.obs_dim, spec.n_actions, seed)
        self.rng = np.random.default_rng(seed)
        self.grad_steps = 0
        self.episode_returns: list = []

    # ---- acting ------------------------------------------------------------------------------
    def act(self, obs: BatchObs, eps: float = 0.0) -> np.ndarray:
        """Epsilon-greedy over legal options; returns option indices."""
        with torch.no_grad():
            x = torch.from_numpy(flatten_obs(obs)).to(self.device)
            q = self.q_net(x).cpu().numpy()                                # [B, n_actions]
        ids = np.where(obs.option_mask, obs.option_ids, 0)
        qo = np.take_along_axis(q, ids, axis=1)
        qo = np.where(obs.option_mask, qo, -np.inf)
        greedy = qo.argmax(1)
        explore = self.rng.random(len(obs)) < eps
        random_idx = self.rng.integers(0, obs.n_options)
        return np.where(explore, random_idx, greedy).astype(np.int64)

    # ---- learning ----------------------------------------------------------------------------
    def td_target(self, reward, next_state, done, next_mask) -> torch.Tensor:
        """r + gamma * max over the LEGAL actions of the next state (0 bootstrap when done)."""
        net = self.target_net if self.cfg.use_target else self.q_net
        with torch.no_grad():
            q2 = net(next_state).masked_fill(~next_mask, float("-inf")).max(1).values
            return torch.where(done, reward, reward + self.cfg.gamma * q2)

    def update(self) -> float:
        s, a, r, s2, d, m = (torch.from_numpy(x).to(self.device) for x in self.buffer.sample(self.cfg.batch_size))
        q = self.q_net(s).gather(1, a[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(q, self.td_target(r, s2, d, m))
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg.grad_clip)
        self.opt.step()
        self.grad_steps += 1
        if self.grad_steps % self.cfg.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return float(loss.detach())

    def train(self, vec: VecEnv, total_steps: int) -> list:
        """Collect `total_steps` environment transitions (across the batch) with epsilon decaying linearly."""
        cfg, spec, B = self.cfg, self.spec, vec.n_envs
        obs = vec.reset()
        running = np.zeros(B)
        steps, owed = 0, 0.0
        rows = np.arange(B)
        while steps < total_steps:
            frac = min(1.0, steps / max(1, cfg.eps_decay_frac * total_steps))
            eps = cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)
            a = self.act(obs, eps)
            st = vec.step(a)
            s, s2 = flatten_obs(obs), flatten_obs(st.obs)
            ids = obs.option_ids[rows, a]
            mask2 = np.zeros((B, spec.n_actions), bool)
            for b in range(B):
                mask2[b, st.obs.option_ids[b][st.obs.option_mask[b]]] = True
            for b in range(B):
                self.buffer.add(s[b], ids[b], st.rewards[b, 0], s2[b], st.done[b], mask2[b])
            running += st.rewards[:, 0]
            for b in np.flatnonzero(st.done):
                self.episode_returns.append(float(running[b]))
                running[b] = 0.0
            obs = st.obs
            steps += B
            if len(self.buffer) >= cfg.learning_starts:
                owed += cfg.train_ratio * B
                while owed >= 1.0:
                    self.update()
                    owed -= 1.0
        return self.episode_returns
