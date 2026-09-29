"""PPO with GAE. Actor-critics implement `forward(TorchObs) -> (logits [B, max_options] with -inf on illegal, value [B])`."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rlf.core import _BATCH_FIELDS, TorchObs, VecEnv, to_torch


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, last_value: np.ndarray,
                gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Generalized advantage estimation over arrays [T, B].

    `dones[t]` is True if the transition at step t ended the episode (no bootstrap, no propagation across it).
    delta_t = r_t + gamma * V_{t+1} * (1 - done_t) - V_t;  A_t = delta_t + gamma * lam * (1 - done_t) * A_{t+1}.
    Returns (advantages, returns = advantages + values).
    """
    T = len(rewards)
    adv = np.zeros_like(rewards, dtype=np.float64)
    next_adv = np.zeros(rewards.shape[1:])
    next_value = np.asarray(last_value, dtype=np.float64)
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        next_adv = delta + gamma * lam * nonterminal * next_adv
        adv[t] = next_adv
        next_value = values[t]
    return adv, adv + values


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 1.0
    lam: float = 0.95
    lr: float = 1e-3
    n_steps: int = 64               # rollout length per env
    epochs: int = 4
    minibatches: int = 4
    clip_eps: float = 0.2           # float("inf") disables the clipping
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    reward_scale: float = 1.0
    max_grad_norm: float = 0.5
    normalize_adv: bool = True


class PPO:
    def __init__(self, net, cfg: PPOConfig = PPOConfig(), seed: int = 0):
        self.net, self.cfg = net, cfg
        self.device = next(net.parameters()).device
        self.opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self._obs = None
        self._vec = None
        self._running = None
        self.episode_returns: list = []
        self.history: list = []

    @torch.no_grad()
    def _act(self, obs):
        t = to_torch(obs, self.device)
        logits, value = self.net(t)
        probs = torch.softmax(logits, -1)
        a = torch.multinomial(probs.cpu(), 1, generator=self.gen).squeeze(1).to(self.device)
        logp = torch.log_softmax(logits, -1).gather(1, a[:, None]).squeeze(1)
        return a.cpu().numpy().astype(np.int64), logp.cpu().numpy(), value.cpu().numpy()

    def collect(self, vec: VecEnv) -> dict:
        """One rollout of `n_steps` per env (auto-reset envs keep going); returns arrays shaped [T, B, ...]."""
        cfg = self.cfg
        if self._obs is None or self._vec is not vec:
            self._vec, self._obs = vec, vec.reset()
            self._running = np.zeros(vec.n_envs)
        T, B = cfg.n_steps, vec.n_envs
        obs_buf = {f: [] for f in _BATCH_FIELDS}
        acts, logps, vals, rews, dones = [], [], [], [], []
        obs = self._obs
        for _ in range(T):
            a, logp, v = self._act(obs)
            st = vec.step(a)
            for f in _BATCH_FIELDS:
                obs_buf[f].append(getattr(obs, f))
            acts.append(a); logps.append(logp); vals.append(v)
            rews.append(st.rewards[:, 0] * cfg.reward_scale); dones.append(st.done.copy())
            self._running += st.rewards[:, 0]
            for b in np.flatnonzero(st.done):
                self.episode_returns.append(float(self._running[b]))
                self._running[b] = 0.0
            obs = st.obs
        self._obs = obs
        with torch.no_grad():
            _, last_v = self.net(to_torch(obs, self.device))
        rews, vals, dones = np.array(rews), np.array(vals), np.array(dones)
        adv, ret = compute_gae(rews, vals, dones, last_v.cpu().numpy(), cfg.gamma, cfg.lam)
        return {"obs": {f: np.stack(obs_buf[f]) for f in _BATCH_FIELDS}, "actions": np.array(acts),
                "logp": np.array(logps), "values": vals, "advantages": adv, "returns": ret}

    def update(self, batch: dict) -> dict:
        cfg = self.cfg
        T, B = batch["actions"].shape
        n = T * B
        flat = lambda x: x.reshape(n, *x.shape[2:])
        obs_all = TorchObs(**{f: torch.from_numpy(flat(batch["obs"][f])).to(self.device) for f in _BATCH_FIELDS})
        a_all = torch.from_numpy(flat(batch["actions"])).to(self.device)
        old_logp = torch.from_numpy(flat(batch["logp"])).float().to(self.device)
        adv_all = torch.from_numpy(flat(batch["advantages"])).float().to(self.device)
        ret_all = torch.from_numpy(flat(batch["returns"])).float().to(self.device)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0, "approx_kl": 0.0}
        n_updates = 0
        for _ in range(cfg.epochs):
            perm = torch.from_numpy(self.rng.permutation(n)).to(self.device)
            for idx in perm.chunk(cfg.minibatches):
                obs = TorchObs(**{f: getattr(obs_all, f)[idx] for f in _BATCH_FIELDS})
                logits, value = self.net(obs)
                logp_all = torch.log_softmax(logits, -1)
                logp = logp_all.gather(1, a_all[idx][:, None]).squeeze(1)
                adv = adv_all[idx]
                if cfg.normalize_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                ratio = torch.exp(logp - old_logp[idx])
                clipped = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
                policy_loss = -torch.min(ratio * adv, clipped * adv).mean()
                value_loss = ((value - ret_all[idx]) ** 2).mean()
                entropy = -(logp_all.exp() * logp_all.masked_fill(~obs.option_mask, 0.0)).sum(-1).mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()
                n_updates += 1
                with torch.no_grad():
                    stats["policy_loss"] += float(policy_loss)
                    stats["value_loss"] += float(value_loss)
                    stats["entropy"] += float(entropy)
                    stats["clip_frac"] += float(((ratio - 1).abs() > cfg.clip_eps).float().mean())
                    stats["approx_kl"] += float((old_logp[idx] - logp).mean())
        return {k: v / n_updates for k, v in stats.items()}

    def train(self, vec: VecEnv, total_steps: int) -> list:
        """Roll out and update until `total_steps` environment transitions have been collected."""
        steps_per_iter = self.cfg.n_steps * vec.n_envs
        for _ in range(max(1, total_steps // steps_per_iter)):
            info = self.update(self.collect(vec))
            recent = self.episode_returns[-50:]
            info["mean_return"] = float(np.mean(recent)) if recent else float("nan")
            info["steps"] = (len(self.history) + 1) * steps_per_iter
            self.history.append(info)
        return self.history
