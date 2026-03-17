import pytest
import torch
from torch.testing import assert_close

from src.contrastivelib.utils import fix_seed
from src.envs.storage import DeepStateStorage, ShallowStateStorage
from tests.test_envs.mock import MockVectorizedEnvironment


@pytest.fixture(params=[ShallowStateStorage, DeepStateStorage])
def storage_class(request):
    return request.param


class TestStateStoragesBasic:
    """Tests if universal AbstractStateStorage properties work on both subclasses"""

    def setup_method(self, method):
        self.max_idx = 20
        self.num_games = 5
        self.seed = 42

    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    def test_all_active_after_init(self, storage_class, device):
        """Verifies that all games are active immediately after initialization."""
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('CUDA not available')

        fix_seed(self.seed)
        env = MockVectorizedEnvironment(max_idx=self.max_idx, device=device)

        # Goals (col 0) = random in [1, max_idx], (col 1) = 0 -- no state is solved
        goals = torch.randint(1, self.max_idx + 1, (self.num_games, 1), device=device)
        positions = torch.zeros((self.num_games, 1), dtype=torch.long, device=device)
        starting_positions = torch.cat([goals, positions], dim=1)

        expected_active_idx = torch.arange(self.num_games, device=device)
        storage = storage_class(starting_positions=starting_positions, env=env, device=device)

        assert_close(storage.active_idx, expected_active_idx)

    @pytest.mark.parametrize('solved_indices', [[], [0], [1, 3], [0, 1, 2, 3, 4]])
    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    def test_solved_at_init(self, storage_class, device, solved_indices):
        """Verifies active count, active indices, and solved rate when some games start solved."""
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('CUDA not available')

        fix_seed(self.seed)
        env = MockVectorizedEnvironment(max_idx=self.max_idx, device=device)

        # Solved games: goal=0, position=0 → immediately solved
        # Unsolved games: goal=random positive, position=0
        unsolved_indices = [i for i in range(self.num_games) if i not in solved_indices]
        goals = torch.randint(1, self.max_idx + 1, (self.num_games,), device=device)
        goals[solved_indices] = 0
        positions = torch.zeros(self.num_games, dtype=torch.long, device=device)
        starting_positions = torch.stack([goals, positions], dim=1)

        expected_active_idx = torch.tensor(unsolved_indices, dtype=torch.long, device=device)
        expected_solved_rate = len(solved_indices) / self.num_games

        storage = storage_class(starting_positions=starting_positions, env=env, device=device)
        storage.active_states()

        assert_close(storage.active_idx, expected_active_idx)
        assert storage._get_solved_rate()['solved_rate'] == expected_solved_rate

    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    def test_multiple_moves(self, storage_class, device):
        """Verifies active tracking and solved rate across two rounds, with one illegal move."""
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('CUDA not available')

        # 3 games: goal=[2, 3, 10], position=[0, 0, 0]
        # Round 1: all move +2 → positions=[2, 2, 2] → game 0 solved (2==2)
        # Round 2: game 1 moves +1 → position=3, solved (3==3); game 2 makes illegal move, dropped
        # Final: solved_rate=2/3, no active games
        num_games = 3
        goals = torch.tensor([2, 3, 10], device=device)
        positions = torch.zeros(num_games, dtype=torch.long, device=device)
        starting_positions = torch.stack([goals, positions], dim=1)

        env = MockVectorizedEnvironment(max_idx=10, device=device)
        storage = storage_class(starting_positions=starting_positions, env=env, device=device)

        # --- Round 1: all three active, move +2, game 0 solved next ---
        expected_active_before_r1 = torch.tensor([0, 1, 2], device=device)
        active = storage.active_states()
        assert_close(storage.active_idx, expected_active_before_r1)

        new_states = active.clone()
        new_states[:, 1] += 2
        storage.process_chosen_moves(
            new_states, torch.ones(len(active), dtype=torch.bool, device=device)
        )

        # Round 2: game 0 solved, games 1 & 2 remain; game 1 moves +1, game 2 makes illegal move
        expected_active_after_r1 = torch.tensor([1, 2], device=device)
        active = storage.active_states()
        assert_close(storage.active_idx, expected_active_after_r1)
        assert storage._get_solved_rate()['solved_rate'] == 1 / num_games

        new_states = active.clone()
        new_states[:, 1] += 1  # game 1: pos→3; game 2: pos→3
        still_active = torch.tensor([True, False], device=device)  # game 2 rejected (illegal)
        storage.process_chosen_moves(new_states, still_active)

        # --- Final: game 1 solved, game 2 dropped — no active games remain ---
        expected_active_final = torch.tensor([], dtype=torch.long, device=device)
        storage.active_states()
        assert_close(storage.active_idx, expected_active_final)
        assert storage._get_solved_rate()['solved_rate'] == 2 / num_games
