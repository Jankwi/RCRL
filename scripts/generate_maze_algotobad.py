"""One-off: generate `algotobad` (haywire) trajectories + comparison gifs for the maze datasets.

For each real trajectory we keep the shared prefix up to a random pivot, then random-walk away from
the trajectory (opposite direction) until a dead end, reusing RandomDirectionMazeAugmentation one
step at a time. If the agent cannot move at the pivot, the bad trajectory is just a 2-frame
stay-in-place. Each bad trajectory is no longer than its good trajectory and uses the same width and
agent-less padding, so it is a drop-in.

Run: uv run python scripts/generate_maze_algotobad.py [--data_dir ...] [--device ...] [--n_vis 3]
"""

import argparse
import glob
import io
import os
import re
import sys
from typing import Tuple

import joblib
import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.maze.augment import RandomDirectionMazeAugmentation
from src.envs.maze.center import VectorizedCenterMaze
from src.envs.maze.utils import get_pos_indices

POS_MARKER = 2


def generate_algotobad(
    trajectories: torch.Tensor,
    lengths: torch.Tensor,
    grid_dim: int,
    aug: RandomDirectionMazeAugmentation,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """trajectories: (N,T,G,G) int8 on device; lengths: (N,) long. Returns (bad, bad_len)."""
    n, t = trajectories.shape[0], trajectories.shape[1]
    batch = torch.arange(n, device=device)

    # Random pivot in [0, length-2] so a real "next" state exists to walk opposite to.
    pivot = (torch.rand(n, device=device) * (lengths - 1).to(torch.float)).to(torch.long)
    pivot = torch.minimum(pivot, lengths - 2).clamp_min(0)

    bad = trajectories.clone()  # prefix [0..pivot] already correct; tail overwritten below
    bad_lengths = pivot + 1
    active = torch.ones(n, dtype=torch.bool, device=device)

    curr = trajectories[batch, pivot]  # state at pivot
    prev = trajectories[batch, pivot + 1]  # trajectory's next cell -> the one to avoid first

    dummy = torch.zeros(n, dtype=torch.long, device=device)
    ones = torch.ones(n, dtype=torch.long, device=device)

    for step in range(t - 1):
        write_idx = pivot + 1 + step
        can_write = active & (write_idx <= lengths - 1)  # stay within the good trajectory's length
        if not can_write.any():
            break

        two_frame = torch.stack([curr, prev], dim=1)  # (N,2,G,G): pivot=0, avoid cell at index 1
        new = aug(trajectories=two_frame, lengths=dummy, pivot_idx=dummy, temporal_dist=ones)

        moved = (
            get_pos_indices(new, grid_dim, POS_MARKER)
            != get_pos_indices(curr, grid_dim, POS_MARKER)
        ).any(dim=1)
        step_ok = can_write & moved

        bad[batch[step_ok], write_idx[step_ok]] = new[step_ok]
        bad_lengths[step_ok] = write_idx[step_ok] + 1

        active = active & moved & (write_idx + 1 <= lengths - 1)  # dead-end or reached good length
        prev, curr = curr, new

    # Nowhere to go at the pivot (no walk step taken): 2-frame stay-in-place at the pivot state.
    stuck = bad_lengths == pivot + 1
    pivot_state = trajectories[batch, pivot]
    bad[stuck, 0] = pivot_state[stuck]
    bad[stuck, 1] = pivot_state[stuck]
    bad_lengths[stuck] = 2

    # Fill the unused tail (>= bad_length) with the maze's agent-less pad board (never sampled).
    time_idx = torch.arange(t, device=device).unsqueeze(0)
    tail = time_idx >= bad_lengths.unsqueeze(1)
    bad[tail] = trajectories[:, -1:].expand(-1, t, -1, -1)[tail]

    return bad, bad_lengths


def _render(env: VectorizedCenterMaze, board: np.ndarray) -> Image.Image:
    fig = env.render_state(board)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def save_pair_gif(
    env: VectorizedCenterMaze,
    good: np.ndarray,
    good_len: int,
    bad: np.ndarray,
    bad_len: int,
    path: str,
) -> None:
    """Side-by-side good (left) | algotobad (right) gif, each held on its last real frame."""
    g = [_render(env, good[k]) for k in range(int(good_len))]
    b = [_render(env, bad[k]) for k in range(int(bad_len))]
    frames = []
    for k in range(max(len(g), len(b))):
        gi, bi = g[min(k, len(g) - 1)], b[min(k, len(b) - 1)]
        canvas = Image.new('RGB', (gi.width + bi.width, max(gi.height, bi.height)), 'white')
        canvas.paste(gi, (0, 0))
        canvas.paste(bi, (gi.width, 0))
        frames.append(canvas)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=300, loop=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/center-maze')
    parser.add_argument('--grid_dim', type=int, default=13)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_vis', type=int, default=3)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    aug = RandomDirectionMazeAugmentation()
    env = VectorizedCenterMaze(device=device, grid_dim=args.grid_dim)
    vis_done = False

    for split in ('train', 'valid', 'test'):
        split_dir = os.path.join(args.data_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for traj_path in sorted(glob.glob(os.path.join(split_dir, '*_trajectories.pkl'))):
            prefix = re.match(r'^(.*)_trajectories\.pkl$', os.path.basename(traj_path)).group(1)
            trajectories = torch.from_numpy(np.asarray(joblib.load(traj_path))).to(device)
            lengths = (
                torch.from_numpy(
                    np.asarray(joblib.load(os.path.join(split_dir, f'{prefix}_lengths.pkl')))
                )
                .to(torch.long)
                .to(device)
            )

            bad, bad_len = generate_algotobad(trajectories, lengths, args.grid_dim, aug, device)

            joblib.dump(
                bad.to(torch.int8).cpu().numpy(), os.path.join(split_dir, f'{prefix}_algotobad.pkl')
            )
            joblib.dump(
                bad_len.to(torch.int32).cpu().numpy(),
                os.path.join(split_dir, f'{prefix}_algotobad_lengths.pkl'),
            )
            print(
                f'{split}/{prefix}: bad_len min/mean/max = '
                f'{int(bad_len.min())}/{float(bad_len.float().mean()):.1f}/{int(bad_len.max())}'
            )

            if not vis_done and args.n_vis > 0:
                vis_dir = os.path.join(args.data_dir, 'algotobad_vis')
                os.makedirs(vis_dir, exist_ok=True)
                tr, ln, bd, bl = (
                    trajectories.cpu().numpy(),
                    lengths.cpu().numpy(),
                    bad.cpu().numpy(),
                    bad_len.cpu().numpy(),
                )
                for i in range(min(args.n_vis, len(tr))):
                    save_pair_gif(
                        env,
                        tr[i],
                        ln[i],
                        bd[i],
                        bl[i],
                        os.path.join(vis_dir, f'{prefix}_pair_{i}.gif'),
                    )
                print(f'wrote {min(args.n_vis, len(tr))} comparison gifs to {vis_dir}')
                vis_done = True


if __name__ == '__main__':
    main()
