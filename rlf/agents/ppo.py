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


class SelfPlay:
    """PPO self-play for two-player envs: the same network controls both seats.

    Each game is either pure self-play (both seats use the live network and both streams of decisions are trained on)
    or live-network-vs-snapshot (the opponent seat uses a network sampled from the pool of recent snapshots; only the
    live seat is trained on). Rewards are the terminal +1/0/-1 of each player, assigned to the last decision of that
    player's stream; GAE runs per (game, player) stream.
    """

    def __init__(self, net, cfg: PPOConfig, make_env, n_envs: int = 64, pool_size: int = 4, snapshot_every: int = 4,
                 p_self: float = 0.5, seed: int = 0):
        import copy
        self._copy = copy
        self.net, self.cfg = net, cfg
        self.ppo = PPO(net, cfg, seed)
        self.vec = VecEnv(make_env, n_envs, seed=seed)
        assert self.vec.spec.n_players == 2
        self.device = self.ppo.device
        self.pool: list = []
        self.pool_size, self.snapshot_every, self.p_self = pool_size, snapshot_every, p_self
        self.rng = np.random.default_rng(seed)
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(seed)
        B = n_envs
        self.obs = self.vec.reset()
        self.mode = np.zeros(B, int)                 # -1 = pure self-play, k >= 0 = snapshot k
        self.seat = np.zeros(B, int)                 # live seat when playing a snapshot
        for b in range(B):
            self._new_game(b)
        self.streams: dict = {}
        self.history: list = []
        self.games_played = 0
        self.iteration = 0

    def _new_game(self, b: int) -> None:
        if self.pool and self.rng.random() >= self.p_self:
            self.mode[b] = int(self.rng.integers(len(self.pool)))
        else:
            self.mode[b] = -1
        self.seat[b] = int(self.rng.integers(2))

    @torch.no_grad()
    def _sample(self, net, obs_sub: BatchObs):
        t = to_torch(obs_sub, self.device)
        logits, value = net(t)
        probs = torch.softmax(logits, -1)
        a = torch.multinomial(probs.cpu(), 1, generator=self.gen).squeeze(1).to(self.device)
        logp = torch.log_softmax(logits, -1).gather(1, a[:, None]).squeeze(1)
        return a.cpu().numpy().astype(np.int64), logp.cpu().numpy(), value.cpu().numpy()

    def collect(self, n_games: int) -> dict:
        """Play until `n_games` more games finish; return a flat batch of trainable decisions ([N, 1, ...] arrays)."""
        B = self.vec.n_envs
        finished: list = []
        n_done = 0
        while n_done < n_games:
            obs = self.obs
            player = obs.player
            live = (self.mode == -1) | (player == self.seat)
            actions = np.zeros(B, np.int64)
            if live.any():
                idx = np.flatnonzero(live)
                a, logp, val = self._sample(self.net, obs.select(idx))
                actions[idx] = a
                for j, b in enumerate(idx):
                    s = self.streams.setdefault((b, int(player[b])), {"rows": [], "a": [], "logp": [], "v": []})
                    s["rows"].append({f: getattr(obs, f)[b] for f in _BATCH_FIELDS})
                    s["a"].append(a[j]); s["logp"].append(logp[j]); s["v"].append(val[j])
            for k in np.unique(self.mode[~live]):
                idx = np.flatnonzero(~live & (self.mode == k))
                actions[idx] = self._sample(self.pool[k], obs.select(idx))[0]
            st = self.vec.step(actions)
            for b in np.flatnonzero(st.done):
                for p in (0, 1):
                    s = self.streams.pop((b, p), None)
                    if s is not None:
                        finished.append((s, float(st.rewards[b, p])))
                self._new_game(b)
                n_done += 1
            self.obs = st.obs
        self.games_played += n_done
        return self._batch(finished)

    def _batch(self, finished: list) -> dict:
        cfg = self.cfg
        rows, acts, logps, vals, advs, rets = [], [], [], [], [], []
        for s, final_reward in finished:
            T = len(s["a"])
            r = np.zeros((T, 1)); r[-1, 0] = final_reward * cfg.reward_scale
            d = np.zeros((T, 1), bool); d[-1, 0] = True
            v = np.array(s["v"])[:, None]
            adv, ret = compute_gae(r, v, d, np.zeros(1), cfg.gamma, cfg.lam)
            rows += s["rows"]; acts += s["a"]; logps += s["logp"]; vals += list(v[:, 0])
            advs += list(adv[:, 0]); rets += list(ret[:, 0])
        obs = {f: np.stack([row[f] for row in rows])[:, None] for f in _BATCH_FIELDS}
        return {"obs": obs, "actions": np.array(acts)[:, None], "logp": np.array(logps)[:, None],
                "values": np.array(vals)[:, None], "advantages": np.array(advs)[:, None], "returns": np.array(rets)[:, None]}

    def train(self, total_steps: int, games_per_iter: int = 64) -> list:
        """Iterate collect + PPO update until at least `total_steps` trainable decisions were used."""
        used = sum(h.get("decisions", 0) for h in self.history)
        while used < total_steps:
            batch = self.collect(games_per_iter)
            info = self.ppo.update(batch)
            info["decisions"] = int(batch["actions"].shape[0])
            info["games"] = self.games_played
            used += info["decisions"]
            info["steps"] = used
            self.iteration += 1
            if self.iteration % self.snapshot_every == 0:
                snap = self._copy.deepcopy(self.net).eval()
                self.pool = (self.pool + [snap])[-self.pool_size:]
            self.history.append(info)
        return self.history


def train_selfplay(net, cfg: PPOConfig, make_env, total_steps: int, n_envs: int = 64, pool_size: int = 4,
                   snapshot_every: int = 4, p_self: float = 0.5, games_per_iter: int = 64, seed: int = 0) -> SelfPlay:
    """Self-play PPO; returns the `SelfPlay` object (its `history` and snapshot `pool`)."""
    sp = SelfPlay(net, cfg, make_env, n_envs, pool_size, snapshot_every, p_self, seed)
    sp.train(total_steps, games_per_iter)
    return sp
