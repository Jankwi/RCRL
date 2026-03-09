"""Utils for contrastivelib"""
import random

import torch


def fix_seed(seed: int = 42):
    """Fixes random seeds for reproducibility across random, torch, and CUDA."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
