import torch


def sample_random_on(mask: torch.Tensor, k: int) -> torch.Tensor:
    """
    Samples k random indices where mask is true, for each batch element.
    Returns a tuple of (indices, valid_mask), both of shape (batch_size, min(k, N)).
    `indices` contains the flattened indices, and `valid_mask` indicates which ones are valid.
    """
    batch_size = mask.shape[0]
    flat_mask = mask.view(batch_size, -1)

    rand_vals = torch.rand_like(flat_mask.float())
    rand_vals[~flat_mask] = float('inf')

    actual_k = min(k, flat_mask.shape[1])
    vals, flat_indices = torch.topk(rand_vals, actual_k, dim=1, largest=False)

    flat_valid = vals < float('inf')

    flat_indices[~flat_valid] = -1
    return flat_indices, flat_valid
