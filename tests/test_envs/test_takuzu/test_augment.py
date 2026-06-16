import pytest
import torch
from torch.testing import assert_close

from src.common import fix_seed
from src.envs.takuzu.augment import (
    AbstractTakuzuAugmentation,
    AllBadPixelChoice,
    AllRandomPixelChoice,
    MinOneBadPixelChoice,
)
from src.envs.utils import sample_random_on


def random_empty_mask(m_boards, n, device):
    """Random boolean mask (True == empty cell), with at least one empty cell per board."""
    mask = torch.randint(0, 2, (m_boards, n, n), dtype=torch.bool, device=device)
    forced = torch.randint(0, n * n, (m_boards,), device=device)
    mask.view(m_boards, -1)[torch.arange(m_boards, device=device), forced] = True
    return mask


def make_boards(m_boards, n, device):
    """Returns (pivot_states, full_boards, empty_mask).

    full_boards are solved boards (values in {1, 2}); pivot_states equal full_boards except
    on empty cells, which are 0.
    """
    full_boards = torch.randint(1, 3, (m_boards, n, n), dtype=torch.uint8, device=device)
    empty_mask = random_empty_mask(m_boards, n, device)
    pivot_states = full_boards.clone()
    pivot_states[empty_mask] = 0
    return pivot_states, full_boards, empty_mask


def build_trajectories(pivot_states, full_boards, pivot_idx, lengths, T, device):
    """Embeds pivot_states at `pivot_idx` and full_boards at `lengths - 1` of zeroed trajs."""
    m, n, _ = pivot_states.shape
    traj = torch.zeros(m, T, n, n, dtype=pivot_states.dtype, device=device)
    batch_idx = torch.arange(m, device=device)
    traj[batch_idx, pivot_idx] = pivot_states
    traj[batch_idx, lengths - 1] = full_boards
    return traj


class _RecordingAugment(AbstractTakuzuAugmentation):
    """Mock augmentation that records the args handed to `_augment` and returns the sampled mask."""

    def _augment(self, pivot_state, sampled_mask, full_board):
        self.last_pivot = pivot_state
        self.last_full = full_board
        self.last_mask = sampled_mask
        return sampled_mask.float()


class TestSampleRandomOn:
    """Tests for sample_random_on function"""

    def setup_method(self, method):
        self.seed = 42
        self.atol = 0.02
        fix_seed(self.seed)

    @pytest.mark.parametrize('m_boards, samples', [(2, 10_000)])
    @pytest.mark.parametrize('n', [2, 4])
    @pytest.mark.parametrize('k', [1, 3])
    def test_sample_random_on(self, m_boards, n, samples, k, device_fx):
        mask = random_empty_mask(m_boards, n, device_fx)
        pick_ratio = (k / mask.sum(dim=(1, 2)).float()).clamp(max=1.0).view(-1, 1, 1)
        expected_distrib = pick_ratio * mask.float()

        mask_rep = mask.repeat_interleave(samples, dim=0)
        sampled, valid = sample_random_on(mask_rep, k)
        sampled_grouped = sampled.view(m_boards, samples, -1)

        distrib = torch.zeros_like(expected_distrib)
        for mask_idx in range(m_boards):
            for i in range(n):
                for j in range(n):
                    idx = i * n + j
                    valid_samples = (sampled_grouped[mask_idx] == idx).sum(dim=-1)
                    distrib[mask_idx, i, j] = valid_samples.float().mean()

        assert_close(distrib, expected_distrib, atol=self.atol, rtol=0.0)


class TestAbstractTakuzuAugmentation:
    """General tests for the new (trajectories, lengths, pivot_idx, temporal_dist) API,
    exercised through a mock subclass."""

    def setup_method(self, method):
        self.seed = 42
        self.atol = 0.02
        fix_seed(self.seed)

    @pytest.mark.parametrize('m_boards', [256])
    @pytest.mark.parametrize('n', [4])
    @pytest.mark.parametrize('T', [8])
    def test_pivot_and_full_extraction(self, m_boards, n, T, device_fx):
        """pivot_state is read at pivot_idx, full_board at lengths - 1, regardless of per-board
        indices."""
        pivot_states, full_boards, _ = make_boards(m_boards, n, device_fx)
        lengths = torch.randint(2, T + 1, (m_boards,), device=device_fx)
        # pivot_idx uniformly in [0, lengths - 2], strictly before the (solved) terminal frame
        pivot_idx = (torch.rand(m_boards, device=device_fx) * (lengths - 1).float()).long()
        traj = build_trajectories(pivot_states, full_boards, pivot_idx, lengths, T, device_fx)

        aug = _RecordingAugment()
        aug(traj, lengths, pivot_idx, temporal_dist=torch.ones_like(pivot_idx))

        assert_close(aug.last_pivot, pivot_states)
        assert_close(aug.last_full, full_boards)

    @pytest.mark.parametrize('m_boards', [512])
    @pytest.mark.parametrize('n', [4])
    def test_sampled_count_and_support(self, m_boards, n, device_fx):
        """Each board samples min(temporal_dist, empties) cells, all of them empty cells."""
        pivot_states, full_boards, empty_mask = make_boards(m_boards, n, device_fx)
        T, pivot_pos = 5, 1
        lengths = torch.full((m_boards,), T, dtype=torch.long, device=device_fx)
        pivot_idx = torch.full((m_boards,), pivot_pos, dtype=torch.long, device=device_fx)
        traj = build_trajectories(pivot_states, full_boards, pivot_idx, lengths, T, device_fx)

        temporal_dist = torch.randint(1, 4, (m_boards,), device=device_fx)
        empties = empty_mask.sum(dim=(1, 2))
        expected_count = torch.minimum(temporal_dist, empties)

        aug = _RecordingAugment()
        aug(traj, lengths, pivot_idx, temporal_dist)
        mask = aug.last_mask

        assert (mask.sum(dim=(1, 2)) == expected_count).all().item()
        # sampled cells are always empty cells
        assert (mask & ~empty_mask).sum().item() == 0

    @pytest.mark.parametrize('m_boards, reps', [(2, 10_000)])
    @pytest.mark.parametrize('n', [4])
    def test_sampling_distribution(self, m_boards, reps, n, device_fx):
        """Empty cells are sampled uniformly, each with prob min(t_dist, empties)/empties."""
        pivot_states, full_boards, empty_mask = make_boards(m_boards, n, device_fx)
        T, pivot_pos = 5, 1
        lengths = torch.full((m_boards,), T, dtype=torch.long, device=device_fx)
        pivot_idx = torch.full((m_boards,), pivot_pos, dtype=torch.long, device=device_fx)
        traj = build_trajectories(pivot_states, full_boards, pivot_idx, lengths, T, device_fx)

        temporal_dist = torch.randint(1, 4, (m_boards,), device=device_fx)
        empties = empty_mask.sum(dim=(1, 2)).float()
        pick_ratio = (temporal_dist.float() / empties).clamp(max=1.0).view(-1, 1, 1)
        expected_distrib = pick_ratio * empty_mask.float()

        tiled_traj = traj.repeat_interleave(reps, dim=0)
        tiled_lengths = lengths.repeat_interleave(reps)
        tiled_pivot_idx = pivot_idx.repeat_interleave(reps)
        tiled_tdist = temporal_dist.repeat_interleave(reps)

        aug = _RecordingAugment()
        mask = aug(tiled_traj, tiled_lengths, tiled_pivot_idx, tiled_tdist)
        distrib = mask.view(m_boards, reps, n, n).mean(dim=1)

        assert_close(distrib, expected_distrib, atol=self.atol, rtol=0.0)

    @pytest.mark.parametrize('m_boards', [256])
    @pytest.mark.parametrize('n', [4])
    def test_default_temporal_dist(self, m_boards, n, device_fx):
        """default_temporal_dist is used when temporal_dist is omitted."""
        pivot_states, full_boards, empty_mask = make_boards(m_boards, n, device_fx)
        T, pivot_pos = 5, 1
        lengths = torch.full((m_boards,), T, dtype=torch.long, device=device_fx)
        pivot_idx = torch.full((m_boards,), pivot_pos, dtype=torch.long, device=device_fx)
        traj = build_trajectories(pivot_states, full_boards, pivot_idx, lengths, T, device_fx)

        default = 2
        empties = empty_mask.sum(dim=(1, 2))
        expected_count = torch.minimum(torch.full_like(empties, default), empties)

        aug = _RecordingAugment(default_temporal_dist=default)
        aug(traj, lengths, pivot_idx)
        assert (aug.last_mask.sum(dim=(1, 2)) == expected_count).all().item()

    @pytest.mark.parametrize('n', [4])
    def test_missing_temporal_dist_raises(self, n, device_fx):
        """An augmentation with no default and no explicit temporal_dist asserts."""
        pivot_states, full_boards, _ = make_boards(2, n, device_fx)
        T, pivot_pos = 5, 1
        lengths = torch.full((2,), T, dtype=torch.long, device=device_fx)
        pivot_idx = torch.full((2,), pivot_pos, dtype=torch.long, device=device_fx)
        traj = build_trajectories(pivot_states, full_boards, pivot_idx, lengths, T, device_fx)

        aug = _RecordingAugment()
        with pytest.raises(AssertionError):
            aug(traj, lengths, pivot_idx)


class TestTakuzuAugmentations:
    """Tests for the concrete Takuzu augmentations."""

    def setup_method(self, method):
        self.seed = 42
        fix_seed(self.seed)

    def _run(self, aug_cls, m_boards, n, device):
        pivot_states, full_boards, empty_mask = make_boards(m_boards, n, device)
        T, pivot_pos = 5, 1
        lengths = torch.full((m_boards,), T, dtype=torch.long, device=device)
        pivot_idx = torch.full((m_boards,), pivot_pos, dtype=torch.long, device=device)
        traj = build_trajectories(pivot_states, full_boards, pivot_idx, lengths, T, device)
        temporal_dist = torch.randint(1, 4, (m_boards,), device=device)

        negative = aug_cls()(traj, lengths, pivot_idx, temporal_dist)

        # Empty cells hold 0, all fills are in {1, 2}, so the changed cells are exactly the sample.
        flip_mask = negative != pivot_states
        empties = empty_mask.sum(dim=(1, 2))
        expected_count = torch.minimum(temporal_dist, empties)

        # only empty cells are ever altered, and exactly min(t_dist, empties) of them
        assert (negative[~empty_mask] == pivot_states[~empty_mask]).all().item()
        assert (flip_mask.sum(dim=(1, 2)) == expected_count).all().item()
        return pivot_states, full_boards, negative, flip_mask

    @pytest.mark.parametrize('m_boards', [256])
    @pytest.mark.parametrize('n', [4])
    def test_all_bad(self, m_boards, n, device_fx):
        _, full_boards, negative, flip_mask = self._run(AllBadPixelChoice, m_boards, n, device_fx)
        # every altered cell holds the wrong value
        assert_close(negative[flip_mask], 3 - full_boards[flip_mask])

    @pytest.mark.parametrize('m_boards', [256])
    @pytest.mark.parametrize('n', [4])
    def test_all_random(self, m_boards, n, device_fx):
        _, _, negative, flip_mask = self._run(AllRandomPixelChoice, m_boards, n, device_fx)
        altered = negative[flip_mask]
        assert ((altered == 1) | (altered == 2)).all().item()

    @pytest.mark.parametrize('m_boards', [256])
    @pytest.mark.parametrize('n', [4])
    def test_min_one_bad(self, m_boards, n, device_fx):
        _, full_boards, negative, flip_mask = self._run(
            MinOneBadPixelChoice, m_boards, n, device_fx
        )
        altered = negative[flip_mask]
        assert ((altered == 1) | (altered == 2)).all().item()

        # every board with at least one altered cell has at least one wrong cell
        wrong = (negative == 3 - full_boards) & flip_mask
        has_flip = flip_mask.any(dim=(1, 2))
        assert (wrong.sum(dim=(1, 2))[has_flip] >= 1).all().item()
