import gin
import torch
import torch.nn as nn

from src.contrastivelib.metrics.sims import SimilarityMeasure
from src.envs import ValueEstimator


def _forward_chunked(model: nn.Module, x: torch.Tensor, batch_size: int) -> torch.Tensor:
    if batch_size is None or len(x) <= batch_size:
        return model(x)
    return torch.cat(
        [model(x[s : s + batch_size]) for s in range(0, len(x), batch_size)],
        dim=0,
    )


@gin.configurable
class SingleGoalValueEstimator(ValueEstimator):
    def __init__(
        self,
        model: nn.Module,
        sim_fn: SimilarityMeasure,
        device: torch.device,
        batch_size: int = None,
    ) -> None:
        super().__init__(device=device)
        self.model = model
        self.sim_fn = sim_fn
        self.batch_size = batch_size

    def get_solved_distance(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        return self.sim_fn(
            _forward_chunked(self.model, states, self.batch_size),
            _forward_chunked(self.model, goals, self.batch_size),
        )


@gin.configurable
class ManyGoalsValueEstimator(ValueEstimator):
    def __init__(
        self,
        model: nn.Module,
        sim_fn: SimilarityMeasure,
        device: torch.device,
        batch_size: int = None,
    ) -> None:
        super().__init__(device=device)
        self.model = model
        self.sim_fn = sim_fn
        self.batch_size = batch_size

    def get_solved_distance(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        # pass to model in (n*k) x state_shape
        # take the min i=1, ..., k for distance to each goal
        if goals.dim() == states.dim():
            goals = goals.unsqueeze(1)
        n, k = goals.shape[0], goals.shape[1]

        goals_flat = goals.reshape(n * k, *goals.shape[2:])
        goals_emb = _forward_chunked(self.model, goals_flat, self.batch_size)
        states_emb = _forward_chunked(self.model, states, self.batch_size)
        states_emb_repeated = states_emb.unsqueeze(1).repeat(1, k, 1).reshape(n * k, -1)
        sims = self.sim_fn(states_emb_repeated, goals_emb, 'elementwise').reshape(n, k)
        return torch.min(sims, dim=1).values


@gin.configurable
class LogitValueEstimator(ValueEstimator):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        batch_size: int = None,
        sim_fn: SimilarityMeasure = None,  # for compatibility with gin configs, not actually used
    ) -> None:
        super().__init__(device=device)
        self.model = model
        self.batch_size = batch_size

    def get_solved_distance(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        # Lower output = more solvable
        # (sigmoid closer to 1 → raw output positive → negated for distance)
        return -_forward_chunked(self.model, states, self.batch_size).squeeze()
