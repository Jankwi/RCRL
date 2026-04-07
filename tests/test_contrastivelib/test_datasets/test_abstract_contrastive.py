import os

import joblib
import numpy as np
import pytest
import torch
from tensordict import TensorDict
from torch.testing import assert_close

from src.common import fix_seed
from src.contrastivelib.datasets.abstract_contrastive import AbstractContrastiveDataset
from src.contrastivelib.utils.special import SpecialStateManager


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
        fix_seed(self.seed)

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
        expected_keys = set(tensor_names + ['trajectories', '_prefix'])
        for file_set in dataset.file_sets:
            assert set(file_set.keys()) == expected_keys
            assert isinstance(file_set['_prefix'], str)
            for key, path in file_set.items():
                if key == '_prefix':
                    continue
                assert os.path.exists(path)

    @pytest.mark.parametrize(
        'file_path, k', [('traj_len', 1), ('traj_len_abs', 1), ('traj_len_abs', 2)]
    )
    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    @pytest.mark.parametrize('chunk_size', [1, 2])
    def test_get_random_starting_positions(self, data_dirs, file_path, k, device, chunk_size):
        """
        Tests that get_random_starting_positions returns starting states of k random trajectories
        from the same file.
        """
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('CUDA not available')

        dataset = _PassthroughDataset(
            file_sets_path=data_dirs[file_path],
            tensor_names=[],
            device=device,
            chunk_size=chunk_size,
        )
        starting = dataset.get_random_starting_positions(k=k)

        all_file_starts = [
            torch.from_numpy(joblib.load(file_set['trajectories']))[:, 0].to(device)
            for file_set in dataset.file_sets
        ]

        assert starting.shape == (k, *all_file_starts[0].shape[1:])
        assert any(
            all(any(torch.equal(s, fs) for fs in file_starts) for s in starting)
            for file_starts in all_file_starts
        )

    @pytest.mark.parametrize('file_path', ['traj_len_abs'])
    @pytest.mark.parametrize('with_spec_mgr', [False, True])
    @pytest.mark.parametrize('last_n', [1, None])
    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    @pytest.mark.parametrize('chunk_size', [1])
    def test_get_random_trajectory_ends(
        self, data_dirs, file_path, with_spec_mgr, last_n, chunk_size, device
    ):
        """Test the get_random_trajectory_ends method - for now just on whole trajectories"""
        assert file_path == 'traj_len_abs'
        trajectories, lengths = (
            torch.from_numpy(self.abs_trajectories).to(device),
            torch.from_numpy(self.abs_lengths).to(device),
        )
        spec_mgr = (
            SpecialStateManager(-1, trajectories[0][0].shape, device=device)
            if with_spec_mgr
            else None
        )
        last_n = last_n if last_n is not None else len(trajectories[0])

        lens = [int(lengths[i].cpu().item()) for i in range(len(trajectories))]
        used_indices = [
            (i, j)
            for i in range(len(trajectories))
            for j in range(max(0, lens[i] - last_n), lens[i])
        ]
        expected_states = torch.stack(
            ([spec_mgr.create_many(1).squeeze(0)] if with_spec_mgr else [])
            + [trajectories[i][j] for i, j in used_indices]
        )
        expected_labels = (['special'] if with_spec_mgr else []) + [
            f'trajectory_{i}' for (i, j) in used_indices
        ]

        dataset = _PassthroughDataset(
            file_sets_path=data_dirs[file_path],
            tensor_names=['lengths'],
            device=device,
            chunk_size=chunk_size,
            spec_mgr=spec_mgr,
        )

        states, labels = dataset.get_random_trajectory_ends(k=len(trajectories), last_n=last_n)
        print(f'{len(expected_labels)=} {len(labels)=}')
        assert labels == expected_labels
        assert_close(states, expected_states)

    @pytest.mark.parametrize(
        'file_path, tensor_names',
        [('traj_len', ['lengths']), ('traj_len_abs', ['lengths', 'abstract'])],
    )
    @pytest.mark.parametrize('chunk_size', [1, 2])
    def test_get_batches_per_epoch(self, data_dirs, file_path, tensor_names, chunk_size):
        """Tests if the get_batches_per_epoch() method is true to it's name"""
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
