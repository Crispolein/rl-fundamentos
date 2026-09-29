"""MiniPTCGNet: entities and options as tokens, a small Transformer encoder, and a pointer head.

Tokens = 1 global + `max_entities` entities + `max_options` options, each projected linearly to `d_model` and given a
token-type embedding (no positional embedding: the encoder is permutation-equivariant over entities and options).
`option_refs` add to an option's token the embedding of the entities it points to. Padded rows are masked out of
attention (`key_padding_mask`, where True = ignore). The action head is a pointer: query from the global token's output,
keys from the option tokens' outputs.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from rlf.core import EnvSpec, TorchObs


class MiniPTCGNet(nn.Module):
    def __init__(self, spec: EnvSpec, d_model: int = 64, n_layers: int = 2, n_heads: int = 4, dim_ff: int = 128):
        super().__init__()
        self.d = d_model
        self.proj_global = nn.Linear(spec.state_dim, d_model)
        self.proj_entity = nn.Linear(spec.entity_dim, d_model)
        self.proj_option = nn.Linear(spec.option_dim, d_model)
        self.type_emb = nn.Embedding(3, d_model)                          # 0 global, 1 entity, 2 option
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_ff, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, 1)

    def tokens(self, obs: TorchObs):
        """Input embeddings [B, 1 + E + O, d] and the validity mask [B, 1 + E + O] (True = real token)."""
        B, E, O = obs.state.shape[0], obs.entities.shape[1], obs.options.shape[1]
        g = self.proj_global(obs.state).unsqueeze(1) + self.type_emb.weight[0]
        ent = self.proj_entity(obs.entities) + self.type_emb.weight[1]                 # [B, E, d]
        opt = self.proj_option(obs.options) + self.type_emb.weight[2]                  # [B, O, d]
        if E > 0:
            for k in range(2):                                                          # up to two referenced entities
                ref = obs.option_refs[:, :, k]
                picked = ent.gather(1, ref.clamp(min=0).unsqueeze(-1).expand(-1, -1, self.d))
                opt = opt + picked * (ref >= 0).unsqueeze(-1).to(opt.dtype)
        x = torch.cat([g, ent, opt], dim=1)
        valid = torch.cat([obs.state.new_ones(B, 1, dtype=torch.bool), obs.entity_mask, obs.option_mask], dim=1)
        return x, valid

    def forward(self, obs: TorchObs) -> tuple[torch.Tensor, torch.Tensor]:
        x, valid = self.tokens(obs)
        out = self.encoder(x, src_key_padding_mask=~valid)                              # True = ignore
        E = obs.entities.shape[1]
        h_global, h_opts = out[:, 0], out[:, 1 + E:]
        logits = (self.w_k(h_opts) * self.w_q(h_global).unsqueeze(1)).sum(-1) / math.sqrt(self.d)
        return logits.masked_fill(~obs.option_mask, float("-inf")), self.value(h_global).squeeze(-1)
