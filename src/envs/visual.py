import io
from typing import List

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.manifold import TSNE

from src.envs import ValueEstimator, VectorizedEnvironment
from src.envs.explore import eval_no_search
from src.envs.storage import DeepStateStorage

_TSNE_RANDOM_STATE = 42
_DISTINCT_COLORS = [
    '#e41a1c',
    '#377eb8',
    '#4daf4a',
    '#984ea3',
    '#ff7f00',
    '#ffff33',
    '#a65628',
    '#f781bf',
    '#999999',
    '#66c2a5',
    '#fc8d62',
    '#8da0cb',
    '#e78ac3',
    '#a6d854',
    '#ffd92f',
    '#e5c494',
    '#b3b3b3',
    '#8dd3c7',
    '#bebada',
    '#fb8072',
]
_PERPLEXITY = 5


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
    tensors = tensors.cpu()

    materials: List[List[Image.Image]] = []
    for i in range(n_traj):
        frames: List[Image.Image] = []

        traj = tensors[i]['trajectories'].numpy()
        length = tensors[i]['lengths']
        for j in range(length.item()):
            buf = io.BytesIO()
            fig = env.render_state(traj[j])
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
            buf.seek(0)
            img = Image.open(buf)
            img.load()
            frames.append(img)
            plt.close(fig)

        materials.append(frames)
    return materials


def get_embed_plots(embeds: torch.Tensor, labels: List[str]) -> matplotlib.figure.Figure:
    """Creates a t-SNE scatter plot for provided embeddings and per-point trajectory labels."""
    if len(embeds) != len(labels):
        raise ValueError('embeds and labels must have the same number of samples')

    embeds_np = embeds.detach().cpu().float().numpy()
    tsne = TSNE(n_components=2, random_state=_TSNE_RANDOM_STATE, perplexity=_PERPLEXITY)
    embeds_2d = tsne.fit_transform(embeds_np)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    unique_labels = list(dict.fromkeys(labels))

    for i, label in enumerate(unique_labels):
        indices = [j for j, current_label in enumerate(labels) if current_label == label]
        points = embeds_2d[indices]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            alpha=0.7,
            s=24,
            color=_DISTINCT_COLORS[i % len(_DISTINCT_COLORS)],
            label=label,
        )

    ax.set_title('Trajectory End-State Embeddings (t-SNE)')
    ax.grid(alpha=0.2)
    if unique_labels:
        ax.legend(loc='best', fontsize=8, ncol=2)

    fig.tight_layout()
    return fig
