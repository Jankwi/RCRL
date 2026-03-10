from typing import Tuple

import gin
import torch
import torch.nn as nn


@gin.configurable
class SpecialStateManager():
    """Class for detecting and dispensing special tokens representing special states"""

    def __init__(self, val: int, shape: Tuple[int, ...], device: torch.device):
        self.device = device
        self.val = val
        self.shape = shape
        self.template = torch.full(shape, val, device=self.device)
        self.detect_dim = tuple(range(1, len(shape) + 1))

    @torch.no_grad()
    def detect_special_states(self, states):
        """Return a boolean mask describing which states are special"""
        return torch.all(states == self.template, dim=self.detect_dim)

    @torch.no_grad()
    def create_many(self, n_special_states):
        """Creates n special states"""
        return self.template.unsqueeze(0).expand(n_special_states, *self.shape).clone()

class SpecialStateModelWrapper(nn.Module):
    """Module wrapper: returns an embedding for the special token, base model output otherwise"""

    def __init__(self, base_model: nn.Module, spec_mgr: SpecialStateManager, embed_dim: int):
        super().__init__()
        self.base_model = base_model
        self.spec_mgr = spec_mgr
        self.embed_dim = embed_dim
        self.special_embed = nn.Parameter(torch.randn(embed_dim))

    def forward(self, x):
        spec_mask = self.spec_mgr.detect_special_states(x)
        result = torch.zeros(x.shape[0], self.embed_dim, device=self.spec_mgr.device)
        result[spec_mask] += self.special_embed
        result[~spec_mask] = self.base_model(x[~spec_mask])
        return result
