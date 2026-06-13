# ruff: noqa: F401
import os
from typing import Dict, List, Tuple, Type

import gin
import matplotlib.pyplot as plt
import torch

from src.contrastivelib.datasets import AbstractContrastiveDataset, PassthroughDataset
from src.contrastivelib.datasets.utils import extract_answers
from src.envs import VectorizedEnvironment
from src.envs.explore import eval_no_search
from src.envs.maze.colorful import VectorizedColorfulMaze
from src.envs.maze.quarter import VectorizedQuarterMaze
from src.envs.maze.target import VectorizedTargetMaze
from src.envs.takuzu import VectorizedTakuzu
from src.envs.validation import (
    DataBasedValueEstimator,
    count_solved_trajectories,
    count_valid_transitions,
)
from src.envs.visual import get_trajectory_gifs
from src.jobs.utils.anomalies import AnomalyManager
from src.jobs.utils.dummy import DummySpecialStateManager


@gin.configurable
class ValidationJob:
    """Job for validating a dataset against its environment"""

    def __init__(
        self,
        dataset_path: str,
        env_cls: Type[VectorizedEnvironment],
        dataset_cls: Type[AbstractContrastiveDataset] = PassthroughDataset,
        chunk_size: int = 1024,
        n_gifs: int = 0,
        n_solved_rate: int = 64,
        n_invalid_renders: int = 5,
        n_unsolved_renders: int = 0,
        save_dir: str = '/validation-logs',
        pass_dummy_spec_mgr: bool = False,
        gin_config: Dict[str, str] = None,
    ) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset_cls(
            file_sets_path=dataset_path, chunk_size=chunk_size, device=self.device
        )
        extra_args = (
            {'spec_mgr': DummySpecialStateManager(device=self.device)}
            if pass_dummy_spec_mgr
            else {}
        )
        self.env = env_cls(device=self.device, **extra_args)
        self.n_gifs = n_gifs
        self.n_solved_rate = n_solved_rate
        self.needs_answers = self.env.requires_answers()

        self.save_dir = dataset_path + save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.anm_mgr = AnomalyManager(
            dir=f'{self.save_dir}/anomalies',
            budget={
                'invalid_transition': n_invalid_renders,
                'unsolved_trajectory': n_unsolved_renders,
            },
        )

        print('-----Initalized ValidationJob-----------')
        for name, value in gin_config.items():
            print(f'  {name}: {value}')
        print('---------------------------------------')

    def get_gifs(self) -> None:
        if self.n_gifs == 0:
            return

        tensors = self.dataset.get_random_trajectory_data(self.n_gifs)
        materials = get_trajectory_gifs(
            starting_positions=None,
            env=self.env,
            ve=None,
            max_len=None,
            tensors=tensors,
        )

        print('---------------------------------------')
        print(f'Saving {len(materials)} gifs to {self.save_dir}...')
        for i, frames in enumerate(materials):
            filename = f'{self.save_dir}/trajectory-{i}.gif'
            frames[0].save(filename, save_all=True, append_images=frames[1:], duration=200, loop=0)
        print('---------------------------------------')

    def get_sanity_check_solved_rate(self) -> None:
        tensors = self.dataset.get_random_trajectory_data(self.n_solved_rate)
        starting = tensors['trajectories'][:, 0]
        answers = (
            extract_answers(tensors['trajectories'], tensors['lengths'])
            if self.needs_answers
            else None
        )
        ve = DataBasedValueEstimator(
            tensors['trajectories'], tensors['lengths'], device=self.device
        )

        solved_rate = eval_no_search(starting, self.env, ve, answers=answers)['solved_rate']
        print('---------------------------------------')
        print(f'Sanity check {solved_rate=}')
        print('---------------------------------------')

    def _process_stats(self, stats: List[Tuple[int, int]]) -> Tuple[int, int, float]:
        total_first = sum(first for first, _ in stats)
        total_all = sum(second for _, second in stats)
        return total_first, total_all, total_first / total_all

    def get_traj_stats(self) -> None:
        min_len, max_len = None, None

        def update_len_stats(lengths: torch.Tensor) -> None:
            nonlocal min_len, max_len
            batch_min, batch_max = lengths.min().item(), lengths.max().item()
            min_len = batch_min if min_len is None else min(min_len, batch_min)
            max_len = batch_max if max_len is None else max(max_len, batch_max)

        valid, solved = [], []
        for tensors, _ in self.dataset:
            update_len_stats(tensors['lengths'])
            valid.append(count_valid_transitions(**tensors, env=self.env, anm_mgr=self.anm_mgr))
            solved.append(count_solved_trajectories(**tensors, env=self.env, anm_mgr=self.anm_mgr))

        print('---------------------------------------')
        print("---------Trajectories' Stats----------")
        valid_tr, all_tr, prc_tr = self._process_stats(valid)
        print(f'{valid_tr} out of {all_tr} transitions are valid ({prc_tr:.2%})')
        solved_tr, all_tr, prc_solved = self._process_stats(solved)
        print(f'{solved_tr} out of {all_tr} trajectories are solved ({prc_solved:.2%})')
        print(f'Trajectory lengths range from {min_len} to {max_len}')
        print('---------------------------------------')

    def execute(self) -> None:
        with self.dataset.preload_file_set():
            self.get_gifs()
            self.get_sanity_check_solved_rate()
            self.get_traj_stats()
