import torch

from src.envs import ValueEstimator, VectorizedEnvironment


class MockVectorizedEnvironment(VectorizedEnvironment):
    """
    Mock environment for testing:
        States:
            Pairs of ints, where:
                - the first int is just the goal
                - the second int is the current position index
        State space:
            All position indices from [0, max_idx] (and any goal)
        Action space:
            Adding 1 or 2 to the position idx (if within bounds)
        Solved condition:
            The environment is solved, when a state of form (a,a) is reached,
            i.e. when the position index is equal to the goal
    """

    def __init__(self, max_idx: int, device: torch.device) -> None:
        super().__init__(torch.tensor([[0, 1], [0, 2]], device=device))
        self.device = device
        self.max_idx_tensor = torch.tensor(max_idx, device=self.device)

    def generate_goals(self, states: torch.Tensor) -> torch.Tensor:
        return states[:, 0].clone()

    def preview_actions(
        self, states: torch.Tensor, action_tensors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states += action_tensors
        legality = states[:, 1] <= self.max_idx_tensor
        return states, legality

    def is_solved(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        assert len(states) > 0
        return states[:, 1] == goals

    def render_state(self, state) -> None:
        return None


class MockValueEstimator(ValueEstimator):
    """
    A mock value estimator:
        - prefers states, with a smaller remainder (mod 3)
        - out of states with the same remainder, prefers the one close to the goal
            (shouldn't happen with default params)
    """

    def __init__(self, device: torch.device, mod_base: int = 3, c: float = 10.0):
        super().__init__(device=device)
        self.mod_base_tensor = torch.tensor([mod_base], device=self.device)
        self.c = torch.tensor([c], device=self.device)

    def get_solved_distance(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        mod_rems = torch.remainder(states[:, 1], self.mod_base_tensor).to(torch.float)
        dist = torch.abs(goals - states[:, 1]).to(torch.float)
        return mod_rems + 1 / (dist + self.c)
