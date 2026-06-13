import torch


def extract_answers(trajectories: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    answer_idx = lengths.to(torch.long) - 1
    batch_idx = torch.arange(len(trajectories), device=trajectories.device)
    return trajectories[batch_idx, answer_idx].clone()
