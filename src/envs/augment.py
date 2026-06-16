from abc import ABC, abstractmethod
from typing import Tuple, Union

import torch

# Negative only, or (negative, custom_positive) when custom_positive is requested.
InnerResult = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
# What __call__ returns, depending on (traj_positive, custom_positive):
#   (False, False) -> negative
#   (True,  False) -> (negative, traj_positive)
#   (False, True)  -> (negative, custom_positive)
#   (True,  True)  -> ((negative, custom_positive), traj_positive)
CallResult = Union[
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
]


class AbstractAugmentation(ABC):
    def __init__(self, default_temporal_dist: int = None) -> None:
        self.default_temporal_dist = default_temporal_dist

    @abstractmethod
    def _inner_call(
        self,
        trajectories: torch.Tensor,
        lengths: torch.Tensor,
        pivot_idx: torch.Tensor,
        temporal_dist: torch.Tensor,
        batch_idx: torch.Tensor,
        custom_positive: bool = False,
    ) -> InnerResult:
        pass

    def supports_custom_positive(self) -> bool:
        return False

    def __call__(
        self,
        trajectories: torch.Tensor,
        lengths: torch.Tensor,
        pivot_idx: torch.Tensor,
        temporal_dist: torch.Tensor = None,
        traj_positive: bool = False,
        custom_positive: bool = False,
    ) -> CallResult:
        """
        For each trajectory, augments the state at pivot_idx by temporal_dist steps.
        (Usually in the wrong direction)

        The return shape depends on the requested positives (see CallResult):
        the negative alone by default, optionally paired with a custom positive
        (from _inner_call) and/or the in-trajectory positive. Callers that request
        both (e.g. the augment preview) get the nested
        ((negative, custom_positive), traj_positive).
        """
        assert temporal_dist is not None or self.default_temporal_dist is not None
        temporal_dist = (
            temporal_dist
            if temporal_dist is not None
            else torch.full_like(pivot_idx, self.default_temporal_dist)
        )

        batch_idx = torch.arange(trajectories.shape[0], device=trajectories.device)
        inner_ret = self._inner_call(
            trajectories=trajectories,
            lengths=lengths,
            pivot_idx=pivot_idx,
            temporal_dist=temporal_dist,
            batch_idx=batch_idx,
            custom_positive=custom_positive,
        )

        if traj_positive:
            # Precondition: caller bounds temporal_dist so pivot_idx + temporal_dist <= lengths - 1.
            positive = trajectories[batch_idx, pivot_idx + temporal_dist]
            return inner_ret, positive
        return inner_ret
