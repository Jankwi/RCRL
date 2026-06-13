import os
from typing import Dict

import gin
import matplotlib.pyplot as plt
import torch

from src.contrastivelib.datasets import AbstractContrastiveDataset
from src.contrastivelib.utils.special import SpecialStateManager
from src.envs import ValueEstimator, VectorizedEnvironment
from src.envs.explore import eval_no_search
from src.envs.storage import DeepStateStorage
from src.envs.visual import get_trajectory_gifs
from src.jobs import AbstractJob
from src.jobs.utils.logging import AbstractLogger


@gin.configurable
class TrajectoryVisualizationJob(AbstractJob):
    """
    Renders trajectory visualizations for a single environment.

    For each sampled trajectory it saves a gif of the rollout, renders the goal/solved
    state as a PNG (skipped automatically when the goal is a special token), and writes
    a human-readable `.log` dump of every board in order plus the goal state.
    """

    def __init__(
        self,
        env: VectorizedEnvironment,
        ve: ValueEstimator,
        device: torch.device,
        spec_mgr: SpecialStateManager = None,
        logger: AbstractLogger = None,
        n_gifs: int = None,
        default_n_gifs: int = 0,
        gif_max_len: int = None,
        render_goal: bool = True,
        write_log: bool = True,
        gif_duration: int = 200,
        subdir: str = None,
        gin_config: Dict[str, str] = None,
    ) -> None:
        self.env = env
        self.ve = ve
        self.device = device
        self.spec_mgr = spec_mgr
        self.logger = logger
        self.n_gifs = n_gifs if n_gifs is not None else default_n_gifs
        self.gif_max_len = gif_max_len
        self.render_goal = render_goal
        self.write_log = write_log
        self.gif_duration = gif_duration
        self.subdir = subdir or env.env_string()

    def _write_log(
        self,
        path: str,
        split: str,
        index: int,
        boards: torch.Tensor,
        goal: torch.Tensor,
        is_special: bool,
    ) -> None:
        lines = [
            f'Trajectory {index} ({split}) on {self.env.env_string()}',
            f'Number of boards: {len(boards)}',
            '',
        ]
        for j, board in enumerate(boards):
            lines.append(f'Board {j}:')
            lines.append(str(board.cpu().numpy()))
            lines.append('')

        lines.append('Goal state:')
        if is_special:
            val = self.spec_mgr.val if self.spec_mgr is not None else None
            shape = tuple(self.spec_mgr.shape) if self.spec_mgr is not None else None
            lines.append(f'[special state — token value={val}, shape={shape}]')
        else:
            lines.append(str(goal.cpu().numpy()))
        lines.append('')

        with open(path, 'w') as f:
            f.write('\n'.join(lines))

    def execute(
        self,
        dataset: AbstractContrastiveDataset,
        out_dir: str,
        split: str,
        epoch: int,
    ) -> Dict[str, float]:
        if self.n_gifs == 0:
            return {}

        needs = self.env.requires_answers()
        sampled = dataset.get_random_starting_positions(
            k=self.n_gifs, device=self.device, append_answers=needs
        )
        starting, answers = sampled if needs else (sampled, None)

        max_len = {'max_len': self.gif_max_len} if self.gif_max_len else {}
        tensors, _ = eval_no_search(
            starting, self.env, self.ve, DeepStateStorage, max_len=max_len, answers=answers
        )
        materials = get_trajectory_gifs(None, self.env, None, tensors=tensors)

        goals = tensors['goals']
        if self.spec_mgr is not None:
            is_special = self.spec_mgr.detect_special_states(goals)
        else:
            is_special = torch.zeros(len(goals), dtype=torch.bool)

        sub = f'{out_dir}/{self.subdir}'
        os.makedirs(sub, exist_ok=True)

        gif_paths: Dict[str, str] = {}
        goal_paths: Dict[str, str] = {}
        for i, frames in enumerate(materials):
            stem = f'{sub}/trajectory-{split}-{i}'
            key = f'{self.env.env_string()}_{split}_trajectory-{i}'

            gif_path = f'{stem}.gif'
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=self.gif_duration,
                loop=0,
            )
            gif_paths[f'{key}-gif'] = gif_path

            special_i = bool(is_special[i])
            if self.render_goal is not False and not special_i:
                goal_path = f'{stem}-goal.png'
                fig = self.env.render_state(goals[i].cpu().numpy())
                fig.savefig(goal_path, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)
                goal_paths[f'{key}-goal'] = goal_path

            if self.write_log:
                length = tensors[i]['lengths'].item()
                boards = tensors[i]['trajectories'][:length]
                self._write_log(f'{stem}.log', split, i, boards, goals[i], special_i)

        if self.logger is not None:
            self.logger.log_gifs(epoch, gif_paths)
            if goal_paths:
                self.logger.log_image(epoch, goal_paths, format='png')

        return {}
