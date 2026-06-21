import gin
import torch

from src.envs.maze import MaxOneTargetMaze


@gin.configurable
class ExplicitGoalMaze(MaxOneTargetMaze):
    """
    A maze env in which the goal is the provided terminal state.
    """

    def env_string(self) -> str:
        return 'maze_explicit'

    def requires_answers_for_goal_generation(self) -> bool:
        return True

    def generate_goals(self, states: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
        return answers

    def is_solved(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        return (states == goals).all(dim=(1, 2))
