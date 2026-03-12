import os
import random
import re
from abc import abstractmethod
from typing import Dict, Iterator, List, Tuple

import gin
import joblib
import torch
from tensordict import TensorDict
from torch.utils.data import IterableDataset


@gin.configurable
class AbstractContrastiveDataset(IterableDataset):
    """
    Abstract class implementing utils for contrative datasets,
    leaving the contrastive_pick method abstract
    """

    def __init__(
        self,
        file_sets_path: str,
        tensor_names: List[str],
        device: torch.device,
        chunk_size: int,
        master_tn: str = 'trajectories',
    ) -> None:
        self.device, self.chunk_size = device, chunk_size
        self.master_tn, self.tensor_names = master_tn, tensor_names
        self.file_sets = self._get_all_file_names(file_sets_path, tensor_names, master_tn)

    def _get_all_file_names(
        self, file_sets_path: str, tensor_names: List[str], master_tn: str
    ) -> List[Dict[str, str]]:
        """For each file, with the master tensor .pkl name, scans for all other tensor files"""
        all_files, file_sets = set(os.listdir(file_sets_path)), []

        for master_filename in all_files:
            match = re.match(rf'^(.*)_{master_tn}\.pkl$', master_filename)
            if match:
                prefix = match.group(1)
                file_sets.append({})
                for suffix in tensor_names + [master_tn]:
                    extra_filename = f'{prefix}_{suffix}.pkl'
                    assert extra_filename in all_files
                    file_sets[-1][suffix] = f'{file_sets_path}/{extra_filename}'
        return file_sets

    def get_batches_per_epoch(self) -> int:
        """Returns the number of batches per epoch"""
        if not hasattr(self, 'batches_per_epoch'):
            self.batches_per_epoch = 0
            for file_set in self.file_sets:
                master_tensor = torch.from_numpy(joblib.load(file_set[self.master_tn]))
                self.batches_per_epoch += len(master_tensor) // self.chunk_size
                del master_tensor
        return self.batches_per_epoch

    def _load_file_set(self, file_set: Dict[str, str]) -> Tuple[TensorDict, int]:
        """Loads and trims (for even chunking) tensors, from  paths specified in the dictionary"""
        master_tensor = torch.from_numpy(joblib.load(file_set[self.master_tn]))
        perm = torch.randperm(len(master_tensor))
        ub = (len(master_tensor) // self.chunk_size) * self.chunk_size

        data = {self.master_tn: master_tensor[perm][:ub]}
        for tn in self.tensor_names:
            data[tn] = torch.from_numpy(joblib.load(file_set[tn]))[perm][:ub]
        return TensorDict(data, batch_size=[ub]), ub // self.chunk_size

    @abstractmethod
    def contrastive_pick(self, **kwargs: torch.Tensor) -> Tuple[TensorDict, TensorDict]:
        """Returns a tuple of tensordicts, containing state tensors, and extras respectively"""
        pass

    def __iter__(self) -> Iterator[Tuple[TensorDict, TensorDict]]:
        """Iterates over file sets (and chunks in them) in random order, yields contrastive picks"""
        random.shuffle(self.file_sets)
        for file_set in self.file_sets:
            data, num_batches = self._load_file_set(file_set)
            for i in range(num_batches):
                lb, ub = i * self.chunk_size, (i + 1) * self.chunk_size
                yield self.contrastive_pick(**data[lb:ub].to(self.device))
