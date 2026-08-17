"""Reconstruct the Takuzu and Center Maze datasets from the archives committed in `data-tracked/`.

Reads `data-tracked/compressed-datasets/` and writes `data/`, in the exact layout, filenames, and
dtypes the dataset loaders expect. Needs no network access, no account, and no token.

Run: uv run python scripts/decompress_datasets.py [--out ...] [--only takuzu] [--verify]
(or simply `./decompress.sh`, which forwards the same flags)
"""

import argparse
import json
import os
import sys
from typing import Dict

import joblib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.dataset_codec import (
    MAZE_SUFFIXES,
    TAKUZU_SUFFIXES,
    content_digest,
    decode_maze,
    decode_takuzu,
    unpack,
)
from src.jobs.utils import switch_path


def env_root(env: str, out: str) -> str:
    """Where an environment's splits go.

    Without --out this mirrors how the configs resolve their `data/<name>` dataset paths, so
    $RCRL_DATA_PATH_SWITCH sends the data wherever a run would look for it.
    """
    return f'{out}/{env}' if out is not None else switch_path(f'data/{env}')


def restore_file_set(entry: Dict[str, object], source: str, out: str, verify: bool) -> int:
    """Decodes one archive and writes its .pkl files. Returns bytes written."""
    with open(f'{source}/{entry["archive"]}', 'rb') as handle:
        arrays = unpack(handle.read())

    if entry['kind'] == 'takuzu':
        rebuilt = dict(zip(TAKUZU_SUFFIXES, decode_takuzu(arrays)))
    else:
        rebuilt = dict(zip(MAZE_SUFFIXES, decode_maze(arrays)))

    split_dir = f'{env_root(entry["env"], out)}/{entry["split"]}'
    os.makedirs(split_dir, exist_ok=True)
    written = 0

    for suffix, spec in entry['outputs'].items():
        array = np.ascontiguousarray(
            rebuilt[suffix].astype(np.dtype(spec['dtype'])).reshape(spec['shape'])
        )
        if verify:
            digest = content_digest(array)
            assert digest == spec['sha256'], (
                f'checksum mismatch on {entry["prefix"]}_{suffix} '
                f'(expected {spec["sha256"][:12]}, got {digest[:12]})'
            )

        path = f'{split_dir}/{entry["prefix"]}_{suffix}.pkl'
        joblib.dump(torch.from_numpy(array) if spec['torch'] else array, path)
        written += os.path.getsize(path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='data-tracked/compressed-datasets')
    parser.add_argument(
        '--out',
        default=None,
        help='output directory (default: data/, honouring $RCRL_DATA_PATH_SWITCH)',
    )
    parser.add_argument('--only', choices=('takuzu', 'center-maze'), default=None)
    parser.add_argument(
        '--verify', action='store_true', help='check reconstructed tensors against the manifest'
    )
    args = parser.parse_args()

    with open(f'{args.source}/manifest.json') as handle:
        manifest = json.load(handle)

    entries = [e for e in manifest['file_sets'] if args.only in (None, e['env'])]
    if not entries:
        raise SystemExit(f'no file sets matched --only {args.only}')

    roots = sorted({env_root(e['env'], args.out) for e in entries})
    print(f'Decompressing {len(entries)} file sets into {", ".join(roots)} ...')
    total = 0
    for entry in entries:
        written = restore_file_set(entry, args.source, args.out, args.verify)
        total += written
        print(f'  {entry["env"]}/{entry["split"]}/{entry["prefix"]}: {written / 1e6:.1f} MB')

    checked = ' (checksums verified)' if args.verify else ''
    print(f'\nDone: {total / 1e6:.1f} MB written{checked}')


if __name__ == '__main__':
    main()
