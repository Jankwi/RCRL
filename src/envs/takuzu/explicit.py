import gin
import torch

from src.envs.takuzu import AbstractVectorizedTakuzu


@gin.configurable
class ExplicitGoalTakuzu(AbstractVectorizedTakuzu):
    """
    Vectorized environment for the game of Takuzu,
    which requires the solved state beforehand, and uses it as the goal.
    """

    def env_string(self) -> str:
        return 'explicit_goal_takuzu'

    def requires_answers_for_goal_generation(self) -> bool:
        return True

    def generate_goals(self, states: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
        return answers

    def is_solved(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        return (states == goals).all(dim=(1, 2))
