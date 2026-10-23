"""Author-side: compress `data-original/` into the committed `data-tracked/compressed-datasets/`.

Reviewers never run this — they run `./decompress.sh`. It lives in the repo so the compressed
archives are reproducible and auditable rather than opaque blobs.

Every file set is decoded back in memory and compared against the source before anything is
written, so a bad archive can never reach the commit.

Run: uv run python scripts/compress_datasets.py [--source ...] [--out ...]
"""

import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, List

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.dataset_codec import (
    MAZE_SUFFIXES,
    TAKUZU_SUFFIXES,
    content_digest,
    decode_maze,
    decode_takuzu,
    encode_maze,
    encode_takuzu,
    pack,
)

# env -> (codec kind, split dirs, tensor suffixes). Center Maze has no test split.
DATASETS = {
    'takuzu': ('takuzu', ('train', 'valid', 'test'), TAKUZU_SUFFIXES),
    'center-maze': ('maze', ('train', 'valid'), MAZE_SUFFIXES),
}


def as_numpy(loaded: object) -> np.ndarray:
    """Source pickles hold torch tensors (Takuzu) or numpy arrays (Center Maze)."""
    return loaded.numpy() if hasattr(loaded, 'numpy') else loaded


def compress_file_set(kind: str, split_dir: str, prefix: str, suffixes: tuple) -> Dict[str, object]:
    """Encodes one file set, verifies the round-trip, and returns (blob, manifest entry)."""
    source = {s: as_numpy(joblib.load(f'{split_dir}/{prefix}_{s}.pkl')) for s in suffixes}

    if kind == 'takuzu':
        arrays = encode_takuzu(source['trajectories'], source['algotobad'], source['lengths'])
        rebuilt = dict(zip(TAKUZU_SUFFIXES, decode_takuzu(arrays)))
        emit_torch = True
    else:
        arrays = encode_maze(
            source['trajectories'],
            source['algotobad'],
            source['lengths'],
            source['algotobad_lengths'],
        )
        rebuilt = dict(zip(MAZE_SUFFIXES, decode_maze(arrays)))
        emit_torch = False

    for suffix, original in source.items():
        restored = rebuilt[suffix].astype(original.dtype).reshape(original.shape)
        assert np.array_equal(original, restored), f'round-trip mismatch on {prefix}_{suffix}'

    outputs = {
        suffix: {
            'dtype': str(original.dtype),
            'shape': list(original.shape),
            'torch': emit_torch,
            'sha256': content_digest(original),
        }
        for suffix, original in source.items()
    }
    return {'blob': pack(arrays), 'outputs': outputs, 'n': int(source['lengths'].shape[0])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='data-original')
    parser.add_argument('--out', default='data-tracked/compressed-datasets')
    args = parser.parse_args()

    entries: List[Dict[str, object]] = []
    total_raw = total_compressed = 0

    for env, (kind, splits, suffixes) in DATASETS.items():
        for split in splits:
            split_dir = f'{args.source}/{env}/{split}'
            if not os.path.isdir(split_dir):
                continue

            for path in sorted(glob.glob(f'{split_dir}/*_trajectories.pkl')):
                prefix = re.match(r'^(.*)_trajectories\.pkl$', os.path.basename(path)).group(1)
                raw = sum(os.path.getsize(f'{split_dir}/{prefix}_{s}.pkl') for s in suffixes)

                result = compress_file_set(kind, split_dir, prefix, suffixes)
                blob = result['blob']

                archive = f'{env}/{split}/{prefix}.npz.xz'
                os.makedirs(f'{args.out}/{env}/{split}', exist_ok=True)
                with open(f'{args.out}/{archive}', 'wb') as handle:
                    handle.write(blob)

                entries.append(
                    {
                        'env': env,
                        'split': split,
                        'prefix': prefix,
                        'kind': kind,
                        'archive': archive,
                        'n': result['n'],
                        'outputs': result['outputs'],
                    }
                )
                total_raw += raw
                total_compressed += len(blob)
                print(
                    f'{env}/{split}/{prefix}: {raw / 1e6:.1f} MB -> {len(blob) / 1e6:.2f} MB '
                    f'({raw / len(blob):.0f}x), round-trip exact'
                )

    with open(f'{args.out}/manifest.json', 'w') as handle:
        json.dump({'version': 1, 'file_sets': entries}, handle, indent=2)
        handle.write('\n')

    print(
        f'\n{len(entries)} file sets: {total_raw / 1e6:.1f} MB -> '
        f'{total_compressed / 1e6:.2f} MB ({total_raw / total_compressed:.0f}x)'
    )


if __name__ == '__main__':
    main()
