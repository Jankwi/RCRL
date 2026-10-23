#!/usr/bin/env bash
# Reconstructs the Takuzu and Center Maze datasets into data/ from the archives in data-tracked/.
# Accepts the same flags as scripts/decompress_datasets.py, e.g. ./decompress.sh --verify
set -euo pipefail

uv run python scripts/decompress_datasets.py "$@"
