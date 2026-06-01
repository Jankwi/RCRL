import gin
import torch

from src.contrastivelib.utils.special import SpecialStateManager
from src.envs.takuzu import VectorizedTakuzu


@gin.configurable
class ArbitraryGoalTakuzu(VectorizedTakuzu):
    """
    Vectorized environment for the game of Takuzu,
    which uses random board pixel fills as arbitrary goals
    (not necessarily satisfying the rules of Takuzu).
    """

    def __init__(
        self,
        n: int,
        device: torch.device,
        spec_mgr: SpecialStateManager = None,  # for compatibility, always None though
        blue_marker: int = 1,
        yellow_marker: int = 2,
    ) -> None:
        super().__init__(
            n=n, spec_mgr=None, device=device, blue_marker=blue_marker, yellow_marker=yellow_marker
        )

    def env_string(self) -> str:
        return 'arbitrary_goal'

    def generate_goals(self, states: torch.Tensor) -> torch.Tensor:
        full = torch.randint(1, 3, size=states.shape, device=states.device, dtype=states.dtype)
        full[states != 0] = states[states != 0]
        return full

    def is_solved(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        return (states == goals).all(dim=(1, 2))
