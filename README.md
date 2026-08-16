# Relative Contrastive Reinforcement Learning (RCRL)
This repository contains code for Relative Contrastive Reinforcement Learning (RCRL) experiments on 2 combinatorial environments: Takuzu and Center Maze.

If you just want to reproduce RCRL runs with abstract goals, ensure you have the [Prerequisites](#prerequisites) covered, and run:
```bash
# RCRL on Takuzu with the Compass Goal
uv run python runner.py --config_file configs/takuzu/compass/rcrl.gin
```

```bash
# RCRL on Center Maze with the Compass Goal
uv run python runner.py --config_file configs/center-maze/compass/rcrl.gin
```

## Prerequisites

### UV
To run the code, you need to have [uv installed](https://docs.astral.sh/uv/getting-started/installation/).
Then, you have to install all dependencies using:
```bash
uv sync
```

### Datasets
The datasets used for [Takuzu](https://huggingface.co/datasets/Jankwi/Takuzu) and [Center Maze](https://huggingface.co/datasets/Jankwi/Center-Maze) are both available on Hugging Face. The dataset cards contain information about their structure and references to generation processes. The least complex way to obtain the datasets is to download them straight from Hugging Face.

For that, a [Hugging Face account and access token](https://huggingface.co/docs/hub/security-tokens) are required, which you use to log in:
```bash
uv run hf auth login --token your_token_here
```

To download the Takuzu dataset run:
```bash
uv run hf download Jankwi/Takuzu --local-dir ./data/takuzu --repo-type dataset --include "*"
```

For the Center Maze dataset use:
```bash
uv run hf download Jankwi/Center-Maze --local-dir ./data/center-maze --repo-type dataset --include "*"
```

### WANDB
All training configs log metrics to [Weights and Biases](https://wandb.ai/) by default. To disable logging, set `MultipleLoggers.wandb` to `False` in the config.

If you wish to use wandb in this repository, you need to generate an [API key](https://docs.wandb.ai/models/quickstart). Then, run:
```bash
wandb init
```
and enter your wandb key when prompted.

You also need to specify wandb project and entity names. For security reasons, those are not listed explicitly, but rather used as environment variables. You need to set them by calling:
```bash
export WANDB_ENTITY=your-wandb-entity
export WANDB_PROJECT=your-wandb-project
```

## Running Experiments
### Configs
Each experiment is defined by a gin config, living somewhere in the ```configs/``` directory.  
The main training configs are grouped in the following order:
* **Environment** (one of ```takuzu/``` or ```center-maze/```)
* **Goal Type** (one of ```compass/``` (abstract goal) or ```explicit/``` (CRL sanity check & mean heuristic experiment))
* **Method** (```rcrl.gin``` (ours), or one of the baselines: ```crtr.gin``` and ```crtr-mixed.gin```)  

Our main results can be reproduced by running the training jobs for RCRL on the compass goal, using ```configs/takuzu/compass/rcrl.gin``` and ```configs/center-maze/compass/rcrl.gin```.

### Running Jobs
To launch a job defined by a config file, call the ```runner.py``` script, e.g.:
```bash
uv run python runner.py --config_file configs/takuzu/compass/rcrl.gin
```

If you wish to override some of the variables from the config, then run it with gin bindings:
```bash
uv run python runner.py --config_file configs/takuzu/compass/rcrl.gin --gin_bindings 'run.seed=1'
```

### Tips for Local Runs
If you just wish to check a method directionally, and not produce a full run, you can run it with the ```ez.sh``` script, which will override the config to run with small parameters, and without wandb logging.

For example, instead of calling:
```bash
uv run python runner.py --config_file configs/takuzu/compass/rcrl.gin
```
simply use
```bash
./ez.sh configs/takuzu/compass/rcrl.gin
```

The ```ez.sh``` is useful for checking approaches directionally, and the parameters are designed for the runs to fit on smaller GPUs.

### Tips for Server Runs
When working with the repo on a server, you may want the datasets to live in a different directory than ```data/```.  
This repo provides a mechanism to switch the dataset path at runtime, so that you can keep the datasets in a fixed location on the server, while still using relative paths in configs.

To do so, simply set the following environment variable:
```bash
export RCRL_DATA_PATH_SWITCH=/absolute/path/to/your/datasets
```
Any `data/<name>` path in a config will be resolved to `$RCRL_DATA_PATH_SWITCH/<name>`
at runtime.

## Development
### Linting
Linting in this project is configured using ruff. To trigger checks, run:
```bash
uv run ruff check
```
For formatting changes run:
```bash
uv run ruff format
```

### Testing
If you wish to execute all tests in a given directory run:
```
uv run python3 -m pytest tests/path/to/testing/directory
```
To deselect tests marked as slow, run:
```
uv run python -m pytest tests/path/to/testing/directory/ -m "not slow"
```