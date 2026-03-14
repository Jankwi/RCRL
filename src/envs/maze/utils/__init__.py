import torch


def get_pos_indices(states: torch.Tensor, grid_dim: int) -> torch.Tensor:
    """Returns the agent's position in a gird_dim x grid_dim maze"""
    flat_states = states.view(-1, grid_dim * grid_dim)
    flat_indices = torch.argmax(flat_states, dim=1)
    rows = torch.div(flat_indices, grid_dim, rounding_mode='floor')
    cols = torch.remainder(flat_indices, grid_dim)
    return torch.stack([rows, cols], dim=1)
