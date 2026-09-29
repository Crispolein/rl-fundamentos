"""REINFORCE with an optional baseline: none, batch mean, or a learned state value."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rlf.core import _BATCH_FIELDS, BatchObs, TorchObs, VecEnv, get_device, to_torch


@dataclass
class Episode:
    obs: dict                # BatchObs field name -> array stacked over the steps of the episode
    actions: np.ndarray      # int64 [T]; option indices
    rewards: np.ndarray      # float64 [T]
    returns: np.ndarray      # float64 [T]; discounted return-to-go G_t (scaled by reward_scale)


def returns_to_go(rewards: np.ndarray, gamma: float) -> np.ndarray:
    g, out = 0.0, np.zeros(len(rewards))
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g
        out[t] = g
    return out


class Reinforce:
    """loss = -sum_t log pi(a_t|s_t) * (G_t - b(s_t)), averaged over steps; with baseline "value" the critic head
    of the network regresses G_t and provides b(s_t)."""

    def __init__(self, net, baseline: str = "value", gamma: float = 1.0, lr: float = 1e-3, value_coef: float = 0.5,
                 entropy_coef: float = 0.0, reward_scale: float = 1.0, normalize_adv: bool = False, seed: int = 0):
        assert baseline in ("none", "mean", "value")
        self.net, self.baseline, self.gamma = net, baseline, gamma
        self.value_coef, self.entropy_coef, self.reward_scale = value_coef, entropy_coef, reward_scale
        self.normalize_adv = normalize_adv
        self.device = next(net.parameters()).device
        self.opt = torch.optim.Adam(net.parameters(), lr=lr)
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(seed)
        self._obs: BatchObs | None = None
        self._vec = None
        self._fresh: np.ndarray | None = None
        self.history: list = []

    # ---- data collection ---------------------------------------------------------------------
    @torch.no_grad()
    def _sample(self, obs: BatchObs) -> np.ndarray:
        logits, _ = self.net(to_torch(obs, self.device))
        a = torch.multinomial(torch.softmax(logits, -1).cpu(), 1, generator=self.gen).squeeze(1)
        return a.numpy().astype(np.int64)

    def collect(self, vec: VecEnv, n_episodes: int) -> list[Episode]:
        """Complete on-policy episodes. Envs that are mid-episode when a call starts wait for their next episode."""
        if self._obs is None or self._vec is not vec:
            self._vec = vec
            self._obs, self._fresh = vec.reset(), np.ones(vec.n_envs, bool)
        obs, record = self._obs, self._fresh.copy()
        bufs = [{"rows": [], "a": [], "r": []} for _ in range(vec.n_envs)]
        done_eps: list[Episode] = []
        while len(done_eps) < n_episodes:
            a = self._sample(obs)
            st = vec.step(a)
            for b in range(vec.n_envs):
                if record[b]:
                    bufs[b]["rows"].append({f: getattr(obs, f)[b] for f in _BATCH_FIELDS})
                    bufs[b]["a"].append(a[b])
                    bufs[b]["r"].append(st.rewards[b, 0])
                if st.done[b]:
                    if record[b]:
                        done_eps.append(self._make_episode(bufs[b]))
                    bufs[b] = {"rows": [], "a": [], "r": []}
                    record[b] = True
            obs = st.obs
        self._obs = obs
        # unfinished episodes are discarded: only envs that just started a new episode are recorded next time
        self._fresh = np.array([record[b] and not bufs[b]["rows"] for b in range(vec.n_envs)])
        return done_eps

    def _make_episode(self, buf: dict) -> Episode:
        rows = buf["rows"]
        r = np.array(buf["r"]) * self.reward_scale
        return Episode({f: np.stack([row[f] for row in rows]) for f in _BATCH_FIELDS}, np.array(buf["a"], np.int64),
                       np.array(buf["r"]), returns_to_go(r, self.gamma))

    # ---- loss --------------------------------------------------------------------------------
    def _stack(self, episodes: list[Episode]):
        obs = TorchObs(**{f: torch.from_numpy(np.concatenate([e.obs[f] for e in episodes])).to(self.device)
                          for f in _BATCH_FIELDS})
        a = torch.from_numpy(np.concatenate([e.actions for e in episodes])).to(self.device)
        g = torch.from_numpy(np.concatenate([e.returns for e in episodes])).float().to(self.device)
        return obs, a, g

    def loss(self, episodes: list[Episode]):
        obs, a, g = self._stack(episodes)
        logits, value = self.net(obs)
        logp_all = torch.log_softmax(logits, -1)
        logp = logp_all.gather(1, a[:, None]).squeeze(1)
        if self.baseline == "none":
            b = torch.zeros_like(g)
        elif self.baseline == "mean":
            b = g.mean().detach().expand_as(g)
        else:
            b = value.detach()
        adv = g - b
        if self.normalize_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        policy_loss = -(logp * adv).mean()
        value_loss = ((value - g) ** 2).mean() if self.baseline == "value" else torch.zeros((), device=g.device)
        probs = logp_all.exp()
        entropy = -(probs * logp_all.masked_fill(~obs.option_mask, 0.0)).sum(-1).mean()
        total = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        return total, {"policy_loss": float(policy_loss.detach()), "value_loss": float(value_loss.detach()), "entropy": float(entropy.detach())}

    def update(self, episodes: list[Episode]) -> dict:
        total, info = self.loss(episodes)
        self.opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.opt.step()
        return info

    def train(self, vec: VecEnv, n_updates: int, episodes_per_update: int = 16) -> list:
        for _ in range(n_updates):
            eps = self.collect(vec, episodes_per_update)
            info = self.update(eps)
            info["mean_return"] = float(np.mean([e.rewards.sum() for e in eps]))
            info["n_steps"] = int(sum(len(e.actions) for e in eps))
            self.history.append(info)
        return self.history

    def episode_gradient(self, episode: Episode, baseline_value: float | None = None) -> torch.Tensor:
        """Flat gradient of sum_t log pi(a_t|s_t) * (G_t - b) for ONE episode; b = baseline_value, else the
        critic's V(s_t) for baseline "value", else 0."""
        obs, a, g = self._stack([episode])
        logits, value = self.net(obs)
        logp = torch.log_softmax(logits, -1).gather(1, a[:, None]).squeeze(1)
        if baseline_value is not None:
            b = torch.full_like(g, baseline_value)
        elif self.baseline == "value":
            b = value.detach()
        else:
            b = torch.zeros_like(g)
        obj = (logp * (g - b)).sum()
        grads = torch.autograd.grad(obj, [p for p in self.net.parameters()], allow_unused=True)
        return torch.cat([(gr if gr is not None else torch.zeros_like(p)).reshape(-1)
                          for gr, p in zip(grads, self.net.parameters())])
