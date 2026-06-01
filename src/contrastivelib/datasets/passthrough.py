from typing import Dict, List, Tuple

import gin
import torch

from src.contrastivelib.datasets import AbstractContrastiveDataset
from src.contrastivelib.utils.special import SpecialStateManager


@gin.configurable
class PassthroughDataset(AbstractContrastiveDataset):
    """
    Concrete subclass that returns input data as-is.
    Useful for code, which doesn't perform contrstive picks,
    but needs access to AbstractContrastiveDataset's methods
    """

    def __init__(
        self,
        file_sets_path: str,
        device: torch.device,
        chunk_size: int = 1024,
        spec_mgr: SpecialStateManager = None,
        tensor_names: List[str] = ['lengths'],
        master_tn: str = 'trajectories',
    ) -> None:
        super().__init__(
            file_sets_path=file_sets_path,
            tensor_names=tensor_names,
            device=device,
            chunk_size=chunk_size,
            master_tn=master_tn,
            spec_mgr=spec_mgr,
        )

    def contrastive_pick(self, **kwargs) -> Tuple[Dict[str, torch.Tensor], None]:  # noqa: ANN003
        return kwargs, None
