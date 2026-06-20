# ruff: noqa: F401
import os
from typing import Dict, List, Type

import gin
import torch

from src.contrastivelib.datasets import AbstractContrastiveDataset, PassthroughDataset
from src.envs import ValueEstimator, VectorizedEnvironment
from src.envs.explore import eval_no_search
from src.envs.maze.colorful import VectorizedColorfulMaze
from src.envs.maze.quarter import VectorizedQuarterMaze
from src.envs.maze.target import VectorizedTargetMaze
from src.jobs import AbstractJob
from src.jobs.utils import switch_path
from src.jobs.utils.dummy import DummySpecialStateManager


class _RandomValueEstimator(ValueEstimator):
    """A ValueEstimator that returns random values, for testing purposes"""

    def get_solved_distance(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        return torch.rand(states.shape[0], device=states.device)


@gin.configurable
class SolvedRateJob(AbstractJob):
    """Job for evaluating the model, and saving checkpoints and logging"""

    def __init__(
        self,
        n_boards: int,
        frames_back: List[int] = None,
        from_starting: bool = True,
        device: torch.device = None,
        dataset: AbstractContrastiveDataset = None,
        file_sets_path: str = None,
        env: VectorizedEnvironment = None,
        env_class: Type[VectorizedEnvironment] = None,
        ve: ValueEstimator = None,
        print_stats: bool = True,
        force_store_on_cpu: bool = False,
        gin_config: Dict[str, str] = None,
    ) -> None:
        file_sets_path = switch_path(file_sets_path)
        self.device = (
            device
            if device is not None
            else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.env = (
            env
            if env is not None
            else env_class(
                device=self.device, spec_mgr=DummySpecialStateManager(device=self.device)
            )
        )
        self.ve = ve if ve is not None else _RandomValueEstimator(self.device)
        dataset = (
            dataset
            if dataset is not None
            else PassthroughDataset(file_sets_path=file_sets_path, device=self.device)
        )

        self.print_stats = print_stats
        self.needs_answers = self.env.requires_answers()

        self.n_boards = n_boards
        with dataset.preload_file_set():
            self.store_device = torch.device('cpu') if force_store_on_cpu else self.device

            # Sample one shared set of trajectories so the global solved_rate and every
            # i-back score are computed on the same boards (apples-to-apples, monotonic).
            data = dataset.get_random_trajectory_data(k=self.n_boards)

            self.from_starting = from_starting
            self.starting_answers = None
            if self.from_starting:
                starting = dataset.get_starting(
                    data,
                    device=self.store_device,
                    append_answers=self.needs_answers,
                )
                if self.needs_answers:
                    self.starting, self.starting_answers = starting
                else:
                    self.starting = starting

            self.frames_back = frames_back
            self.starting_frames_answers = None
            if self.frames_back is not None:
                self.n, self.starting_frames = len(self.frames_back), None
                for i, fb in enumerate(self.frames_back):
                    frames = dataset.get_ith_back(
                        data,
                        i=fb,
                        device=self.store_device,
                        append_answers=self.needs_answers,
                        print_stats=self.print_stats,
                    )
                    if self.needs_answers:
                        frames, frame_answers = frames
                    if self.starting_frames is None:
                        self.starting_frames = torch.empty(
                            (self.n, *frames.shape), dtype=frames.dtype, device=self.store_device
                        )
                        if self.needs_answers:
                            self.starting_frames_answers = torch.empty(
                                (self.n, *frame_answers.shape),
                                dtype=frame_answers.dtype,
                                device=self.store_device,
                            )
                    self.starting_frames[i] = frames
                    if self.needs_answers:
                        self.starting_frames_answers[i] = frame_answers

    def solved_rate_scalar(self, states: torch.Tensor, answers: torch.Tensor = None) -> float:
        answers = answers.to(self.device) if answers is not None else None
        rate_dict = eval_no_search(states.to(self.device), self.env, self.ve, answers=answers)
        return rate_dict['solved_rate']

    def execute(self) -> Dict[str, float]:
        solved_rates = {}

        if self.from_starting:
            solved_rates['solved_rate'] = self.solved_rate_scalar(
                self.starting, self.starting_answers
            )
        if self.frames_back is not None:
            for i, fb in enumerate(self.frames_back):
                frame_answers = (
                    self.starting_frames_answers[i]
                    if self.starting_frames_answers is not None
                    else None
                )
                solved_rates[f'solved_rate_{fb}_back'] = self.solved_rate_scalar(
                    self.starting_frames[i], frame_answers
                )

        if self.print_stats:
            print('Solved rates:')
            for key, value in solved_rates.items():
                print(f'{key}: {value:.4f}')
        return solved_rates
