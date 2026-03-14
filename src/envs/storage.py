import torch

from src.envs import VectorizedEnvironment


class StateStorage:
    """
    A layer of abstraction over the current states.
    Allows the user to operate on local indexing (of active games).
    """

    def __init__(
        self, starting_positions: torch.Tensor, env: VectorizedEnvironment, device: torch.device
    ) -> None:
        self.device = device
        self.env = env

        self.num_games, self.solved_cnt = starting_positions.shape[0], 0
        self.active_idx = torch.arange(self.num_games, device=self.device)

        self.states = starting_positions.clone()
        self.goals = self.env.generate_goals(self.states)
        self.sets = self.env.get_state_sets(n_sets=len(self.states))

    def active_states(self) -> torch.Tensor:
        """Returns all the states, for which exploration hasn't yet finished."""
        solved_mask = self.env.is_solved(self.states[self.active_idx], self.goals[self.active_idx])
        self.solved_cnt += solved_mask.sum().item()
        self.active_idx = self.active_idx[~solved_mask]

        chosen = self.states[self.active_idx].clone()
        self.sets.add_to_sets(chosen, self.active_idx)
        return chosen

    def active_goals(self) -> torch.Tensor:
        return self.goals[self.active_idx]

    def process_chosen_moves(self, chosen_moves: torch.Tensor, still_active: torch.Tensor) -> None:
        self.active_idx = self.active_idx[still_active]
        self.states[self.active_idx] = chosen_moves[still_active]

    def wasnt_visited(self, candidates: torch.Tensor) -> torch.Tensor:
        return self.sets.missing_from_sets(candidates, self.active_idx)

    def get_solved_rate(self) -> float:
        return self.solved_cnt / self.num_games

    def __len__(self) -> int:
        return len(self.active_idx)
