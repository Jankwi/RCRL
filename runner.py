import argparse
import os
import random

import gin

import wandb


@gin.configurable
def run(num_epochs: int, architecture: str):
    # Start a new wandb run to track this script.
    wandb_run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity=os.environ.get('wandb_entity'),
        # Set the wandb project where this run will be logged.
        project=os.environ.get('wandb_project'),
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": 0.02,
            "architecture": architecture,
            "dataset": "hello-world-message",
            "epochs": num_epochs,
        },
    )

    # Simulate training.
    offset = random.random() / 5
    for epoch in range(2, num_epochs):
        acc = 1 - 2**-epoch - random.random() / epoch - offset
        loss = 2**-epoch + random.random() / epoch + offset

        # Log metrics to wandb.
        wandb_run.log({"acc": acc, "loss": loss})

    # Finish the run and upload any remaining data.
    wandb_run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, help="Path to the config file")
    parser.add_argument(
        "--gin_bindings",
        nargs='*',
        default=[],
        metavar='BIND',
        help='Gin bindings like "run.seed=123" "train_job_baseline.lr=1e-4"'
    )

    args = parser.parse_args()
    gin.parse_config_files_and_bindings(
        config_files=[args.config_file],
        bindings=args.gin_bindings
    )

    print("==== Final Config (after overrides) ====")
    print(f"{gin.config_str()=}")

    print(f"{run()=}")
