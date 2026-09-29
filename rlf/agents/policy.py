"""Adapter that turns an actor-critic network into a `Policy` (`act(BatchObs) -> option indices`)."""
from __future__ import annotations

import numpy as np
import torch

from rlf.core import BatchObs, to_torch


class ActorCriticPolicy:
    """Samples from the network's masked distribution, or takes its argmax if `greedy`."""

    def __init__(self, net, greedy: bool = False, seed: int = 0, device: torch.device | None = None):
        self.net, self.greedy = net, greedy
        self.device = device or next(net.parameters()).device
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(seed)

    @torch.no_grad()
    def act(self, obs: BatchObs) -> np.ndarray:
        logits, _ = self.net(to_torch(obs, self.device))
        if self.greedy:
            a = logits.argmax(-1)
        else:
            a = torch.multinomial(torch.softmax(logits, -1).cpu(), 1, generator=self.gen).squeeze(1)
        return a.cpu().numpy().astype(np.int64)
