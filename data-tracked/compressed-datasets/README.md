# Compressed datasets

The Takuzu and Center Maze datasets, stored as structural deltas so they fit in the repository.
Run `./decompress.sh` from the repository root to reconstruct them into `data/`.

Raw, the two datasets are 5.1 GB of `.pkl` tensors. Here they are 13 MB — a ~400x reduction, with
byte-exact reconstruction. Decompression takes about 10 seconds and expands to ~2.7 GB.

## Layout

One `.npz.xz` archive per file set, plus `manifest.json` holding each file set's shapes, dtypes and
per-tensor SHA-256 content hashes (`./decompress.sh --verify` checks reconstructed tensors against
them).

```
takuzu/train/{blue,green,red,yellow}.npz.xz
takuzu/valid/purple.npz.xz
takuzu/test/orange.npz.xz
center-maze/train/train_chunk_{0,1,2,3}.npz.xz
center-maze/valid/valid_chunk_0.npz.xz
```

Each archive expands to the `{prefix}_{tensor}.pkl` files the dataset loaders glob for, under
`data/<environment>/<split>/`.

## Takuzu

8x8 puzzles solved one pixel at a time. `train` holds 2^17 (131072) puzzles across 4 file sets of
2^15, `valid` holds 2^13 (8192) in one file set, and `test` holds 2^14 (16384) in one file set.

For a file set of size M, all `torch.uint8`:

| tensor | shape | meaning |
| --- | --- | --- |
| `trajectories` | (M, 49, 8, 8) | frame `t` of an algorithmic solve of board `i`; `0` empty, `1`/`2` the two pixel values |
| `algotobad` | (M, 49, 8, 8) | same, but the agent starts making wrong pixel choices at a random point |
| `lengths` | (M,) | number of real frames; frames past this repeat the last one |

For each `i`, `trajectories[i]` and `algotobad[i]` share the same starting board and stay identical
until `algotobad` snaps. Exactly one pixel is filled per step, which is what the codec stores.

## Center Maze

13x13 mazes where the agent walks to the centre. `train` holds 2^17 (131072) mazes across 4 chunks
of 2^15, `valid` holds 2^13 (8192) in one chunk. There is no test split.

For a chunk of size M:

| tensor | shape | dtype | meaning |
| --- | --- | --- | --- |
| `trajectories` | (M, 36, 13, 13) | int8 | frame `t` of a solve of maze `i`; `0` free, `1` wall, `2` agent |
| `algotobad` | (M, 36, 13, 13) | int8 | shares a prefix with the solve, then random-walks away until a dead end |
| `lengths` | (M,) | int32 | number of real frames; frames past this are all zero |
| `algotobad_lengths` | (M,) | int32 | the bad trajectory's own length, always at most `lengths` |

Walls are fixed per maze and shared between the two trajectories; exactly one cell holds the agent
in each real frame, and it moves to a 4-adjacent cell per step.

The published dataset also carried a `wrong_choices` tensor. It is not referenced anywhere in this
codebase, so it is not shipped here.

## How the compression works

Both datasets are rollouts over a context that never changes mid-episode, with one atomic edit per
step, so the codec stores the context once plus a per-step delta:

- **Takuzu** — the shared starting board at 2 bits per cell, then one byte per step encoding which
  cell was filled and with which value.
- **Center Maze** — the wall mask as 169 packed bits, then one byte per step for the agent's cell
  index (with a sentinel for padded frames).

Each file set's arrays go into an uncompressed `np.savez` archive that is then run through `lzma`.
Compressing the archive as a whole rather than array by array lets xz exploit the shared prefix
between the good and the `algotobad` step streams.

`scripts/dataset_codec.py` holds the codec and asserts every structural property it relies on;
`scripts/compress_datasets.py` regenerates these archives from the raw data and verifies each
round-trip before writing.
