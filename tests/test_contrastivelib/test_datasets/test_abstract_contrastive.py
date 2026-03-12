import os

import joblib
import numpy as np
import pytest
import torch
from tensordict import TensorDict
from torch.testing import assert_close

from src.contrastivelib.datasets.abstract_contrastive import AbstractContrastiveDataset
from src.contrastivelib.utils import fix_seed


class _PassthroughDataset(AbstractContrastiveDataset):
    """Concrete subclass that returns input data as-is."""

    def contrastive_pick(self, **kwargs):
        return TensorDict(kwargs), TensorDict({})


def _sort_dict_by_key(tensors, sort_key):
    """Sorts all tensors in a dict by the flattened sum of the sort_key tensor."""
    t = tensors[sort_key]
    indices = torch.argsort(t.reshape(t.shape[0], -1).sum(dim=-1))
    return {k: v[indices] for k, v in tensors.items()}


class TestAbstractContrastiveFileLoading:
    """Tests file loading, iteration, and chunking of AbstractContrastiveDataset."""

    def setup_method(self, method):
        self.seed = 42

        # traj_len: 2D states (2x2 maze grids), two file sets (big + small)
        # Designed so that numbers of 1's in each trajectory are unique: big0=2, big1=3, small0=4
        self.big_trajectories = np.array(
            [
                [[[1, 0], [0, 0]], [[0, 1], [0, 0]]],
                [[[1, 1], [0, 1]], [[0, 0], [0, 0]]],
            ],
            dtype=np.float32,
        )  # (2, 2, 2, 2)
        self.big_lengths = np.array([2, 1], dtype=np.float32)

        self.small_trajectories = np.array(
            [
                [[[0, 0], [0, 1]], [[1, 1], [1, 0]]],
            ],
            dtype=np.float32,
        )  # (1, 2, 2, 2)
        self.small_lengths = np.array([2], dtype=np.float32)

        # traj_len_abs: 1D states (3-bit binary numbers), abstract val <-> all prime, one file set
        # traj 0: [1,0,1]=5 -> [0,1,1]=3, prime=yes; traj 1: [1,0,0]=4 -> pad, prime=no
        # Flattened sums: traj0=4, traj1=1 (unique)
        self.abs_trajectories = np.array(
            [
                [[1, 0, 1], [0, 1, 1]],
                [[1, 0, 0], [0, 0, 0]],
            ],
            dtype=np.float32,
        )  # (2, 2, 3)
        self.abs_lengths = np.array([2, 1], dtype=np.float32)
        self.abs_abstract = np.array([1, 0], dtype=np.float32)

    @pytest.fixture(params=['cpu', 'cuda'])
    def device(self, request):
        """Provides available devices, skipping CUDA if unavailable."""
        if request.param == 'cuda' and not torch.cuda.is_available():
            pytest.skip('CUDA not available')
        return request.param

    @pytest.fixture
    def data_dirs(self, tmp_path):
        """Creates traj_len and traj_len_abs directories with mock .pkl data on CPU."""
        traj_len = tmp_path / 'traj_len'
        traj_len.mkdir()
        joblib.dump(self.big_trajectories, traj_len / 'big_trajectories.pkl')
        joblib.dump(self.big_lengths, traj_len / 'big_lengths.pkl')
        joblib.dump(self.small_trajectories, traj_len / 'small_trajectories.pkl')
        joblib.dump(self.small_lengths, traj_len / 'small_lengths.pkl')

        traj_len_abs = tmp_path / 'traj_len_abs'
        traj_len_abs.mkdir()
        joblib.dump(self.abs_trajectories, traj_len_abs / 'some_prefix__trajectories.pkl')
        joblib.dump(self.abs_lengths, traj_len_abs / 'some_prefix__lengths.pkl')
        joblib.dump(self.abs_abstract, traj_len_abs / 'some_prefix__abstract.pkl')

        yield {
            'traj_len': str(traj_len),
            'traj_len_abs': str(traj_len_abs),
        }

    @pytest.mark.parametrize(
        'file_path, tensor_names, success',
        [
            ('traj_len', ['lengths'], True),
            ('traj_len_abs', ['lengths', 'abstract'], True),
            ('traj_len', ['lengths', 'abstract'], False),
        ],
    )
    @pytest.mark.parametrize('chunk_size', [1, 2])
    def test_init_success(self, data_dirs, device, file_path, tensor_names, success, chunk_size):
        """Tests that initialization succeeds or fails based on file availability."""
        if success:
            _PassthroughDataset(
                file_sets_path=data_dirs[file_path],
                tensor_names=tensor_names,
                device=device,
                chunk_size=chunk_size,
            )
        else:
            with pytest.raises(AssertionError):
                _PassthroughDataset(
                    file_sets_path=data_dirs[file_path],
                    tensor_names=tensor_names,
                    device=device,
                    chunk_size=chunk_size,
                )

    @pytest.mark.parametrize(
        'file_path, tensor_names, expected_count',
        [
            ('traj_len', ['lengths'], 2),
            ('traj_len_abs', ['lengths', 'abstract'], 1),
        ],
    )
    def test_file_paths(self, data_dirs, file_path, tensor_names, expected_count):
        """Tests that _get_all_file_names discovers the correct file sets and paths."""
        dataset = _PassthroughDataset(
            file_sets_path=data_dirs[file_path],
            tensor_names=tensor_names,
            device='cpu',
            chunk_size=1,
        )
        assert len(dataset.file_sets) == expected_count
        expected_keys = set(tensor_names + ['trajectories'])
        for file_set in dataset.file_sets:
            assert set(file_set.keys()) == expected_keys
            for path in file_set.values():
                assert os.path.exists(path)

    @pytest.mark.parametrize(
        'file_path, tensor_names',
        [('traj_len', ['lengths']), ('traj_len_abs', ['lengths', 'abstract'])],
    )
    @pytest.mark.parametrize('chunk_size', [1, 2])
    def test_get_batches_per_epoch(self, data_dirs, file_path, tensor_names, chunk_size):
        """Tests if the get_batches_per_epoch() method is true to it's name"""
        fix_seed(self.seed)
        dataset = _PassthroughDataset(
            file_sets_path=data_dirs[file_path],
            tensor_names=tensor_names,
            device='cpu',
            chunk_size=chunk_size,
        )
        assert dataset.get_batches_per_epoch() == len(list(dataset))

    @pytest.mark.parametrize(
        'file_path, tensor_names, chunk_size, expected_iterations',
        [
            ('traj_len', ['lengths'], 1, 3),
            ('traj_len', ['lengths'], 2, 1),
            ('traj_len_abs', ['lengths', 'abstract'], 1, 2),
            ('traj_len_abs', ['lengths', 'abstract'], 2, 1),
        ],
    )
    def test_iter_count(self, data_dirs, file_path, tensor_names, chunk_size, expected_iterations):
        """Tests that the dataset yields the correct number of batches."""
        fix_seed(self.seed)
        dataset = _PassthroughDataset(
            file_sets_path=data_dirs[file_path],
            tensor_names=tensor_names,
            device='cpu',
            chunk_size=chunk_size,
        )
        assert len(list(dataset)) == expected_iterations

    @pytest.mark.parametrize(
        'file_path, tensor_names, chunk_size',
        [
            ('traj_len', ['lengths'], 1),
            ('traj_len_abs', ['lengths', 'abstract'], 2),
        ],
    )
    def test_iter_results(self, data_dirs, device, file_path, tensor_names, chunk_size):
        """Tests that concatenated iteration output matches the original source data."""
        fix_seed(self.seed)
        dataset = _PassthroughDataset(
            file_sets_path=data_dirs[file_path],
            tensor_names=tensor_names,
            device=device,
            chunk_size=chunk_size,
        )

        all_batches = list(dataset)
        yielded = {}
        for key in all_batches[0][0].keys():
            yielded[key] = torch.cat([b[0][key] for b in all_batches], dim=0)

        if file_path == 'traj_len':
            expected = {
                'trajectories': torch.cat(
                    [
                        torch.from_numpy(self.big_trajectories),
                        torch.from_numpy(self.small_trajectories),
                    ],
                    dim=0,
                ).to(device),
                'lengths': torch.cat(
                    [
                        torch.from_numpy(self.big_lengths),
                        torch.from_numpy(self.small_lengths),
                    ],
                    dim=0,
                ).to(device),
            }
        else:
            expected = {
                'trajectories': torch.from_numpy(self.abs_trajectories).to(device),
                'lengths': torch.from_numpy(self.abs_lengths).to(device),
                'abstract': torch.from_numpy(self.abs_abstract).to(device),
            }

        # sort all tensors, to be in the trajectories' ascending order
        yielded = _sort_dict_by_key(yielded, 'trajectories')
        expected = _sort_dict_by_key(expected, 'trajectories')
        for key in expected:
            assert_close(yielded[key], expected[key])
