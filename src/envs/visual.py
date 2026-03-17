import io
from typing import List

import matplotlib.pyplot as plt
import torch
from PIL import Image

from src.envs import ValueEstimator, VectorizedEnvironment
from src.envs.explore import eval_no_search
from src.envs.storage import DeepStateStorage


def get_trajectory_gifs(
    starting_positions: torch.Tensor,
    env: VectorizedEnvironment,
    ve: ValueEstimator,
    max_len: int = None,
) -> List[List[Image.Image]]:
    """For each position, returns a list of image frames"""
    n_traj = starting_positions.shape[0]
    max_len_kwargs = {'max_len': max_len} if max_len is not None else {}
    tensors, _ = eval_no_search(starting_positions, env, ve, DeepStateStorage, **max_len_kwargs)

    materials: List[List[Image.Image]] = []
    for i in range(n_traj):
        frames: List[Image.Image] = []

        traj = tensors[i]['trajectories']
        length = tensors[i]['lengths']
        for j in range(length.cpu().item()):
            buf = io.BytesIO()
            fig = env.render_state(traj[j].cpu().numpy())
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
            buf.seek(0)
            img = Image.open(buf)
            img.load()
            frames.append(img)
            plt.close(fig)

        materials.append(frames)
    return materials
