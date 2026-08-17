"""Structural codec for the Takuzu and Center Maze trajectory datasets.

Both datasets are RL rollouts over a context that stays fixed for a whole episode, changing by a
single atomic edit per step. Storing the context once plus a per-step delta is ~400x smaller than
the raw frame stacks, which is what makes the data small enough to commit.

Every structural property the encoders exploit is asserted while encoding, so a dataset that
violates one fails loudly at compression time rather than silently reconstructing wrong frames on
someone else's machine.

Container: an *uncompressed* ``np.savez`` archive run through ``lzma``. Compressing the archive as
a whole rather than array by array lets xz see the shared prefix between the good and the
``algotobad`` step streams, which is worth ~20% on Takuzu.
"""

import hashlib
import io
import lzma
from typing import Dict, Tuple

import numpy as np

LZMA_PRESET = 6

TAKUZU_STEPS, TAKUZU_GRID = 49, 8
TAKUZU_CELLS = TAKUZU_GRID**2
TAKUZU_SUFFIXES = ('trajectories', 'algotobad', 'lengths')

MAZE_STEPS, MAZE_GRID = 36, 13
MAZE_CELLS = MAZE_GRID**2
MAZE_SUFFIXES = ('trajectories', 'algotobad', 'lengths', 'algotobad_lengths')
MAZE_FREE, MAZE_WALL, MAZE_AGENT = 0, 1, 2
MAZE_PAD_POS = 255  # sentinel for "no agent in this frame", outside the 0..168 cell range


# --------------------------------------------------------------------------- container


def pack(arrays: Dict[str, np.ndarray]) -> bytes:
    """Serialises the encoded arrays into a single xz-compressed npz blob."""
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return lzma.compress(buffer.getvalue(), preset=LZMA_PRESET)


def unpack(blob: bytes) -> Dict[str, np.ndarray]:
    """Inverse of `pack`."""
    with np.load(io.BytesIO(lzma.decompress(blob))) as archive:
        return {name: archive[name] for name in archive.files}


def content_digest(array: np.ndarray) -> str:
    """Hashes array *contents*, not pickle bytes, so it survives joblib/pickle version drift."""
    hasher = hashlib.sha256()
    hasher.update(f'{array.dtype.str}|{array.shape}|'.encode())
    hasher.update(np.ascontiguousarray(array).tobytes())
    return hasher.hexdigest()


# --------------------------------------------------------------------------- takuzu


def _takuzu_encode_steps(flat: np.ndarray) -> np.ndarray:
    """Encodes each step as one byte: 0 for no change, else 1 + cell * 2 + (value - 1)."""
    changed = flat[:, 1:] != flat[:, :-1]
    n_changed = changed.sum(axis=2)
    assert n_changed.max() <= 1, 'a Takuzu step changed more than one cell'

    cell = changed.argmax(axis=2)
    wrote = n_changed == 1
    value = np.take_along_axis(flat[:, 1:], cell[:, :, None], axis=2)[:, :, 0].astype(np.int64)
    assert np.isin(value[wrote], (1, 2)).all(), 'a Takuzu step wrote a value outside {1, 2}'

    code = np.where(wrote, 1 + cell.astype(np.int64) * 2 + (value - 1), 0)
    assert code.max() <= 128, 'Takuzu step code overflowed a byte'
    return code.astype(np.uint8)


def encode_takuzu(
    trajectories: np.ndarray, algotobad: np.ndarray, lengths: np.ndarray
) -> Dict[str, np.ndarray]:
    """Encodes one Takuzu file set. Frames are (N, 49, 8, 8) uint8 over {0, 1, 2}."""
    n = trajectories.shape[0]
    expected = (n, TAKUZU_STEPS, TAKUZU_GRID, TAKUZU_GRID)
    assert trajectories.shape == expected, f'unexpected trajectories shape {trajectories.shape}'
    assert algotobad.shape == expected, f'unexpected algotobad shape {algotobad.shape}'
    assert lengths.shape == (n,), f'unexpected lengths shape {lengths.shape}'

    good = trajectories.reshape(n, TAKUZU_STEPS, TAKUZU_CELLS)
    bad = algotobad.reshape(n, TAKUZU_STEPS, TAKUZU_CELLS)
    assert np.array_equal(good[:, 0], bad[:, 0]), 'good and bad trajectories differ at frame 0'

    # Two bits per cell for the shared start board.
    cells = good[:, 0].astype(np.uint8).reshape(-1, 1)
    bits = np.unpackbits(cells, axis=1, count=2, bitorder='little')
    return {
        'init': np.packbits(bits, bitorder='little').reshape(n, -1),
        'traj_steps': _takuzu_encode_steps(good),
        'algo_steps': _takuzu_encode_steps(bad),
        'lengths': lengths.astype(np.uint8),
    }


def _takuzu_rebuild(init: np.ndarray, code: np.ndarray) -> np.ndarray:
    n = init.shape[0]
    out = np.empty((n, TAKUZU_STEPS, TAKUZU_CELLS), np.uint8)
    out[:, 0] = init
    current = init.copy()

    for step in range(TAKUZU_STEPS - 1):
        encoded = code[:, step].astype(np.int64)
        rows = np.nonzero(encoded > 0)[0]
        payload = encoded[rows] - 1
        current[rows, payload // 2] = (payload % 2 + 1).astype(np.uint8)
        out[:, step + 1] = current

    return out.reshape(n, TAKUZU_STEPS, TAKUZU_GRID, TAKUZU_GRID)


def decode_takuzu(arrays: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (trajectories, algotobad, lengths)."""
    lengths = arrays['lengths']
    n = lengths.shape[0]

    bits = np.unpackbits(arrays['init'].reshape(-1), bitorder='little').reshape(-1, 2)
    init = (bits[:, 0] + 2 * bits[:, 1]).astype(np.uint8).reshape(n, TAKUZU_CELLS)

    return (
        _takuzu_rebuild(init, arrays['traj_steps']),
        _takuzu_rebuild(init, arrays['algo_steps']),
        lengths,
    )


# --------------------------------------------------------------------------- center maze


def _maze_encode_path(flat: np.ndarray, lengths: np.ndarray, walls: np.ndarray) -> np.ndarray:
    """Encodes the agent's flat cell index per frame, with a sentinel for padded frames."""
    valid = np.arange(MAZE_STEPS)[None, :] < lengths[:, None]
    assert (flat[~valid] == MAZE_FREE).all(), 'a padded maze frame was not all zero'

    agent = flat == MAZE_AGENT
    assert (agent.sum(axis=2)[valid] == 1).all(), 'a maze frame did not hold exactly one agent'
    assert ((flat == MAZE_WALL) == walls[:, None, :])[valid].all(), 'maze walls were not static'

    pos = agent.argmax(axis=2).astype(np.uint8)
    pos[~valid] = MAZE_PAD_POS
    return pos


def encode_maze(
    trajectories: np.ndarray,
    algotobad: np.ndarray,
    lengths: np.ndarray,
    algotobad_lengths: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Encodes one Center Maze file set. Frames are (N, 36, 13, 13) int8 over {0, 1, 2}."""
    n = trajectories.shape[0]
    expected = (n, MAZE_STEPS, MAZE_GRID, MAZE_GRID)
    assert trajectories.shape == expected, f'unexpected trajectories shape {trajectories.shape}'
    assert algotobad.shape == expected, f'unexpected algotobad shape {algotobad.shape}'
    for name, array in (('lengths', lengths), ('algotobad_lengths', algotobad_lengths)):
        assert array.shape == (n,), f'unexpected {name} shape {array.shape}'
        assert array.max() <= MAZE_STEPS, f'{name} exceeds the {MAZE_STEPS}-frame trajectory'

    good = trajectories.reshape(n, MAZE_STEPS, MAZE_CELLS)
    bad = algotobad.reshape(n, MAZE_STEPS, MAZE_CELLS)
    # Walls are fixed by the context, so both trajectories share the frame-0 mask.
    walls = good[:, 0] == MAZE_WALL

    return {
        'walls': np.packbits(walls, axis=1, bitorder='little'),
        'traj_pos': _maze_encode_path(good, lengths, walls),
        'algo_pos': _maze_encode_path(bad, algotobad_lengths, walls),
        'lengths': lengths.astype(np.uint8),
        'algo_lengths': algotobad_lengths.astype(np.uint8),
    }


def _maze_rebuild(walls: np.ndarray, pos: np.ndarray) -> np.ndarray:
    n = walls.shape[0]
    out = np.zeros((n, MAZE_STEPS, MAZE_CELLS), np.int8)

    # Written frame by frame; broadcasting walls across all 36 steps at once would cost ~200MB.
    for step in range(MAZE_STEPS):
        rows = np.nonzero(pos[:, step] != MAZE_PAD_POS)[0]
        out[rows, step] = walls[rows]
        out[rows, step, pos[rows, step].astype(np.int64)] = MAZE_AGENT

    return out.reshape(n, MAZE_STEPS, MAZE_GRID, MAZE_GRID)


def decode_maze(
    arrays: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (trajectories, algotobad, lengths, algotobad_lengths)."""
    walls = np.unpackbits(arrays['walls'], axis=1, count=MAZE_CELLS, bitorder='little').astype(
        np.int8
    )

    return (
        _maze_rebuild(walls, arrays['traj_pos']),
        _maze_rebuild(walls, arrays['algo_pos']),
        arrays['lengths'],
        arrays['algo_lengths'],
    )
