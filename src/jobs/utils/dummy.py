import gin
import torch

from src.contrastivelib.utils.special import SpecialStateManager


@gin.configurable
class DummySpecialStateManager(SpecialStateManager):
    def __init__(self, device: torch.device) -> None:
        super().__init__(val=0, shape=(1,), device=device)

    def detect_special_states(self, states: torch.Tensor) -> torch.Tensor:
        return None

    def create_many(self, n_special_states: int) -> torch.Tensor:
        return torch.zeros((n_special_states, 1), dtype=torch.bool, device=self.device)
