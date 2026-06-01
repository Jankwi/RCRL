from typing import Tuple

import gin
import torch
import torch.nn as nn


@gin.configurable
class SpecialStateManager:
    """Class for detecting and dispensing special tokens representing special states"""

    def __init__(
        self, val: int, shape: Tuple[int, ...], device: torch.device, dt_string: str = None
    ) -> None:
        self.device = device
        self.val = val
        self.shape = shape
        self.template = torch.full(shape, val, device=self.device)
        if dt_string is not None:
            self.template = self.template.to(dtype=getattr(torch, dt_string))
        self.detect_dim = tuple(range(1, len(shape) + 1))

    @torch.no_grad()
    def detect_special_states(self, states: torch.Tensor) -> torch.Tensor:
        """Return a boolean mask describing which states are special"""
        return torch.all(states == self.template, dim=self.detect_dim)

    @torch.no_grad()
    def create_many(self, n_special_states: int) -> torch.Tensor:
        """Creates n special states"""
        return self.template.unsqueeze(0).expand(n_special_states, *self.shape).clone()


@gin.configurable
class SpecialStateModelWrapper(nn.Module):
    """Module wrapper: returns an embedding for the special token, base model output otherwise"""

    def __init__(
        self,
        base_model: nn.Module,
        spec_mgr: SpecialStateManager,
        embed_dim: int,
        trainable_base: bool = True,
        trainable_special: bool = True,
        base_checkpoint_path: str = None,
        whole_checkpoint_path: str = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.spec_mgr = spec_mgr
        self.embed_dim = embed_dim

        self.trainable_base = trainable_base
        if not self.trainable_base:
            self.base_model.requires_grad_(False)

        if trainable_special:
            self.special_embed = nn.Parameter(torch.randn(embed_dim))
        else:
            self.register_buffer('special_embed', torch.zeros(embed_dim))

        self._load_checkpoint(base_checkpoint_path, whole_checkpoint_path)

    def _load_checkpoint(self, base_checkpoint_path: str, whole_checkpoint_path: str) -> None:
        if base_checkpoint_path is not None and whole_checkpoint_path is not None:
            raise ValueError('Cannot specify both base_checkpoint_path and whole_checkpoint_path')
        if base_checkpoint_path is not None:
            self.base_model.load_state_dict(torch.load(base_checkpoint_path))
        if whole_checkpoint_path is not None:
            state_dict = torch.load(whole_checkpoint_path)
            self.load_state_dict(state_dict)

    def train(self, mode: bool = True) -> None:
        super().train(mode=mode)
        if not self.trainable_base:
            self.base_model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec_mask = self.spec_mgr.detect_special_states(x)
        result = torch.zeros(x.shape[0], self.embed_dim, device=self.spec_mgr.device)
        if spec_mask.any():
            result = result.index_put((spec_mask,), self.special_embed.expand(spec_mask.sum(), -1))
        if (~spec_mask).any():
            result = result.index_put((~spec_mask,), self.base_model(x[~spec_mask]))
        return result
