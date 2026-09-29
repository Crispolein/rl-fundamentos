"""Pointer-attention actor-critic: the logit of each legal option is a dot product between a query built from the
global state and a key built from the option's own features, so nothing is tied to the number of options, entities
or action ids of one particular environment size."""
from __future__ import annotations

import math

import torch
from torch import nn

from rlf.core import EnvSpec, TorchObs
from rlf.nets.mlp import MLP


class PointerActorCritic(nn.Module):
    """h_global = MLP([state, masked mean of MLP_ent(entities)]); keys = W_k(MLP_opt(options)); query = W_q(h_global);
    logits = query . keys / sqrt(d) with illegal options at -inf; value from h_global."""

    def __init__(self, spec: EnvSpec, d_model: int = 64, hidden: int = 64):
        super().__init__()
        self.d = d_model
        self.ent = MLP(spec.entity_dim, (hidden,), hidden)
        self.glob = MLP(spec.state_dim + hidden, (hidden,), d_model)
        self.opt = MLP(spec.option_dim, (hidden,), d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, 1)
        self.hidden = hidden

    def encode_global(self, obs: TorchObs) -> torch.Tensor:
        B = obs.state.shape[0]
        if obs.entities.shape[1] == 0:
            pooled = obs.state.new_zeros(B, self.hidden)
        else:
            e = torch.relu(self.ent(obs.entities))                                   # [B, E, hidden]
            m = obs.entity_mask.unsqueeze(-1).to(e.dtype)
            pooled = (e * m).sum(1) / m.sum(1).clamp(min=1.0)                        # masked mean
        return torch.relu(self.glob(torch.cat([obs.state, pooled], dim=-1)))

    def forward(self, obs: TorchObs) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode_global(obs)                                                  # [B, d]
        keys = self.w_k(torch.relu(self.opt(obs.options)))                           # [B, n_opts, d]
        query = self.w_q(h)                                                          # [B, d]
        logits = (keys * query.unsqueeze(1)).sum(-1) / math.sqrt(self.d)
        return logits.masked_fill(~obs.option_mask, float("-inf")), self.v(h).squeeze(-1)
