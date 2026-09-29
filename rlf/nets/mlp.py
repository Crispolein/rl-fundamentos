"""MLP building blocks and an actor-critic that scores the global action ids."""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from rlf.core import EnvSpec, TorchObs


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int], out_dim: int):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def option_logits_from_ids(full_logits: torch.Tensor, obs: TorchObs) -> torch.Tensor:
    """Pick the logit of each legal option's global id: [B, n_actions] -> [B, max_options]; padding is -inf."""
    picked = full_logits.gather(1, obs.option_ids.clamp(min=0))
    return picked.masked_fill(~obs.option_mask, float("-inf"))


def flatten_torch_obs(obs: TorchObs) -> torch.Tensor:
    """state ++ flattened entities, the torch twin of `rlf.core.flatten_obs`."""
    return torch.cat([obs.state, obs.entities.reshape(obs.state.shape[0], -1)], dim=1)


class MLPActorCritic(nn.Module):
    """Shared MLP trunk over the flattened observation; the policy head scores every global action id and
    `option_logits_from_ids` keeps only the legal ones. `forward` returns (logits [B, max_options], value [B])."""

    def __init__(self, spec: EnvSpec, hidden: Sequence[int] = (64, 64)):
        super().__init__()
        in_dim = spec.state_dim + spec.max_entities * spec.entity_dim
        self.trunk = MLP(in_dim, hidden[:-1], hidden[-1]) if len(hidden) > 1 else nn.Identity()
        last = hidden[-1] if len(hidden) > 1 else in_dim
        self.pi = nn.Linear(last, spec.n_actions)
        self.v = nn.Linear(last, 1)

    def forward(self, obs: TorchObs) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.relu(self.trunk(flatten_torch_obs(obs)))
        return option_logits_from_ids(self.pi(h), obs), self.v(h).squeeze(-1)
