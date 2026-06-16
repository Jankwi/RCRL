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


def indices_to_mask(indices: torch.Tensor, valid: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """
    Builds a boolean mask over `like` from flattened (indices, valid) pairs, as returned
    by `sample_random_on`. `like` is a (batch_size, *board) template; the returned mask has
    the same shape, with True at every valid sampled index. Invalid samples are scattered
    into a throwaway column and dropped.
    """
    batch_size = like.shape[0]
    flat_dim = like.numel() // batch_size

    mask_flat = torch.zeros(batch_size, flat_dim + 1, dtype=torch.bool, device=like.device)
    safe_indices = torch.where(valid, indices, flat_dim)
    mask_flat.scatter_(1, safe_indices, True)
    return mask_flat[:, :-1].view_as(like)
