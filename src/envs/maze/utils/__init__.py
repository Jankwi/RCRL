import torch


def get_pos_indices(states: torch.Tensor, grid_dim: int) -> torch.Tensor:
    """Returns the agent's position in a gird_dim x grid_dim maze"""
    flat_states = states.view(-1, grid_dim * grid_dim)
    flat_indices = torch.argmax(flat_states, dim=1)
    rows = torch.div(flat_indices, grid_dim, rounding_mode='floor')
    cols = torch.remainder(flat_indices, grid_dim)
    return torch.stack([rows, cols], dim=1)


def generate_border_positions(border: int, grid_dim: int, device: torch.device) -> torch.Tensor:
    """
    Generates a (2*border - 1, grid_dim, grid_dim) shaped tensor, where each 2D tensor,
    contains just one 1 in a board full of 0's,
    outlining the border of the [border, border] shaped square on the [grid_dim, grid_dim] board
    """
    pos_list = [[i, border - 1] for i in range(border)]
    pos_list.extend([[border - 1, i] for i in reversed(range(border - 1))])
    n_pos = len(pos_list)

    pos_tensor = torch.tensor(pos_list, device=device)
    batch_idx = torch.arange(n_pos, device=device)

    all_pos_tensor = torch.zeros((n_pos, grid_dim, grid_dim), dtype=torch.bool, device=device)
    all_pos_tensor[batch_idx, pos_tensor[:, 0], pos_tensor[:, 1]] = 1
    return all_pos_tensor
