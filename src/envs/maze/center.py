import gin
import torch

from src.envs.maze import MaxOneTargetMaze


@gin.configurable
class VectorizedCenterMaze(MaxOneTargetMaze):
    """
    A maze env in which the goal is to reach the center (in the graph theory sense)
    of the maze. Works with mazes both containing and not containing target markers.
    """

    def generate_goals(self, states: torch.Tensor) -> torch.Tensor:
        return self.spec_mgr.create_many(len(states))

    # For now, we don't compute the center position on the fly,
    # but rather rely on ready trajectories.
    def requires_answers_for_solved_checking(self) -> bool:
        return True

    def is_solved(
        self, states: torch.Tensor, goals: torch.Tensor, answers: torch.Tensor
    ) -> torch.Tensor:
        return torch.all(states == answers, dim=(1, 2))
