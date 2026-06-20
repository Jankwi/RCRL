# CRTR-compass
Conditioning CRTR on abstract goals.

## Quickstart

### WANDB login
First, make sure that you are logged into Weights and Biases, by calling:
```bash
wandb init
```
Then enter your API key, and press enter.

### WANDB parameters
For security reasons wandb project and entity names, are not listed explicitly, but rather used as environment variables.
They can be set up by running:
```bash
export wandb_entity=your-wandb-entity
export wandb_project=your-wandb-project
```

### Dataset path switching
Dataset paths in configs use `data/` as a relative prefix. When running locally, place
datasets under `data/` in the repo root. On the server (where datasets live at a fixed
absolute location), set the following before dispatching jobs:
```bash
export RCRL_DATA_PATH_SWITCH=/absolute/path/to/your/datasets
```
Any `data/<name>` path in a config will be resolved to `$RCRL_DATA_PATH_SWITCH/<name>`
at runtime.

### Runner
Then, you can simply launch the runner with the desired config, by calling:
```bash
uv run python runner.py --config_file configs/train/maze-simple-vibe/vibe-train.gin
```
If you wish to override some of the variabkes from the config, then run it with gin bindings:
```bash
uv run python runner.py --config_file configs/train/maze-simple-vibe/vibe-train.gin --gin_bindings 'TrainJob.seed=1'
```

## Linting
Linting in this project is configured using ruff. To trigger checks, run:
```bash
uv run ruff check
```
If you wish to only trigger them in a specific directory (the checks are the most strict in ```src/```), run:
```bash
uv run ruff check src/
```
For formatting changes run:
```bash
uv run ruff format
```

## Testing
If you wish to execute all tests in a given directory run:
```
uv run python3 -m pytest tests/path/to/testing/directory
```
To deselect tests marked as slow, run:
```
uv run python -m pytest tests/path/to/testing/directory/ -m "not slow"
```