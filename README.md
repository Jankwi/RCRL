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

### Runner
Then, you can simply launch the runner with the desired config, by calling:
```bash
uv run python runner.py --config_file configs/train/maze-simple-vibe/vibe-train.gin
```
If you wish to override some of the variabkes from the config, then run it with gin bindings:
```bash
uv run python runner.py --config_file configs/train/maze-simple-vibe/vibe-train.gin --gin_bindings 'TrainJob.seed=1'
```