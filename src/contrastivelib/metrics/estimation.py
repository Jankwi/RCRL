import gin
import torch
import torch.nn as nn

from src.contrastivelib.metrics.sims import SimilarityMeasure
from src.envs import ValueEstimator


@gin.configurable
class SingleGoalValueEstimator(ValueEstimator):
    def __init__(self, model: nn.Module, sim_fn: SimilarityMeasure, device: torch.device) -> None:
        super().__init__(device=device)
        self.model = model
        self.sim_fn = sim_fn

    def get_solved_distance(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        return self.sim_fn(self.model(states), self.model(goals))


@gin.configurable
class ManyGoalsValueEstimator(ValueEstimator):
    def __init__(self, model: nn.Module, sim_fn: SimilarityMeasure, device: torch.device) -> None:
        super().__init__(device=device)
        self.model = model
        self.sim_fn = sim_fn

    def get_solved_distance(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        # pass to model in (n*k) x state_shape
        # take the min i=1, ..., k for distance to each goal
        n = states.shape[0]
        if goals.dim() == states.dim():
            goals = goals.unsqueeze(1)
        k = goals.shape[1]
        goals_flat = goals.reshape(n * k, *goals.shape[2:])
        goals_emb = self.model(goals_flat)
        states_emb = self.model(states)
        states_emb_repeated = states_emb.unsqueeze(1).repeat(1, k, 1).reshape(n * k, -1)
        sims = self.sim_fn(states_emb_repeated, goals_emb).reshape(n, k)
        return torch.min(sims, dim=1).values
