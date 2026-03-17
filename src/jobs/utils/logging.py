import os
from abc import abstractmethod
from typing import Dict

import gin

import wandb


class AbstractLogger:
    def __init__(self, epochs: int, log_freq: int, run_params: dict = None) -> None:
        self.final_epoch = epochs - 1
        self.log_freq = log_freq

    def is_log_epoch(self, epoch: int) -> bool:
        return (epoch == 0) or (epoch % self.log_freq == 0) or (epoch == self.final_epoch)

    @abstractmethod
    def log_gifs(self, epoch: int, paths_dict: Dict[str, str]) -> None:
        pass

    @abstractmethod
    def log_state_dicts(self, epoch: int, paths_dict: Dict[str, str]) -> None:
        pass

    @abstractmethod
    def log_metrics(self, epoch: int, metrics_dict: Dict[str, float]) -> None:
        pass

    @abstractmethod
    def finish_run(self) -> None:
        pass


@gin.configurable
class StdOutLogger(AbstractLogger):
    """Logger class, writing to a file or stdout"""

    def __init__(self, epochs: int, log_freq: int, run_params: dict = {}, file: str = None) -> None:
        super().__init__(epochs=epochs, log_freq=log_freq)
        self.file = open(file, 'a') if file else None
        print('-----Starting stdout logging for run-----', file=self.file)
        for name, value in run_params.items():
            print(f'  {name}: {value}', file=self.file)

    def log_gifs(self, epoch: int, paths_dict: Dict[str, str]) -> None:
        print(f'-----Gifs for {epoch}-----', file=self.file)
        for name, value in paths_dict.items():
            print(f'{name}: {value}', file=self.file)

    def log_state_dicts(self, epoch: int, paths_dict: Dict[str, str]) -> None:
        print(f'-----State dicts for {epoch}-----', file=self.file)
        for name, value in paths_dict.items():
            print(f'{name}: {value}', file=self.file)

    def log_metrics(self, epoch: int, metrics_dict: Dict[str, float]) -> None:
        print(f'-----Metrics for epoch {epoch}-----', file=self.file)
        for name, value in metrics_dict.items():
            print(f'  {name}: {value}', file=self.file)

    def finish_run(self) -> None:
        print('-----Run finished-----', file=self.file)
        if self.file:
            self.file.close()


@gin.configurable
class WandbLogger(AbstractLogger):
    """Logger class, sending all telemetry data to Weights & Biases"""

    def __init__(self, epochs: int, log_freq: int, run_params: dict = {}) -> None:
        super().__init__(epochs=epochs, log_freq=log_freq)
        self.wandb_run = wandb.init(
            entity=os.environ.get('wandb_entity'),
            project=os.environ.get('wandb_project'),
            config=run_params,
        )
        self.log_later = dict()

    def log_gifs(self, epoch: int, paths_dict: Dict[str, str]) -> None:
        self.log_later = self.log_later | {
            f'{name}-epoch-{epoch}': wandb.Video(path, format='gif')
            for name, path in paths_dict.items()
        }

    def log_state_dicts(self, epoch: int, paths_dict: Dict[str, str]) -> None:
        """Wraps all state dicts in a wandb artifact, and sends it"""
        artifact = wandb.Artifact(
            name=f'checkpoint-{epoch}-{self.wandb_run.name}',
            type='model',
            metadata={
                'epoch': epoch,
                'wandb_name': self.wandb_run.name,
                'wandb_id': self.wandb_run.id,
            },
        )
        for name, path in paths_dict.items():
            artifact.add_file(local_path=path, name=name)
        self.wandb_run.log_artifact(artifact)

    def log_metrics(self, epoch: int, metrics_dict: Dict[str, float]) -> None:
        self.wandb_run.log(metrics_dict | {'epoch': epoch} | self.log_later)
        self.log_later.clear()

    def finish_run(self) -> None:
        self.wandb_run.finish()


@gin.configurable
class MultipleLoggers(AbstractLogger):
    """Logger class, comprising of multiple sub-loggers"""

    def __init__(
        self,
        epochs: int,
        log_freq: int,
        run_params: dict,
        stdout: bool = False,
        wandb: bool = False,
    ) -> None:
        super().__init__(epochs=epochs, log_freq=log_freq)
        self.loggers = []
        if stdout:
            self.loggers.append(
                StdOutLogger(epochs=epochs, log_freq=log_freq, run_params=run_params)
            )
        if wandb:
            self.loggers.append(
                WandbLogger(epochs=epochs, log_freq=log_freq, run_params=run_params)
            )

    def log_gifs(self, epoch: int, paths_dict: Dict[str, str]) -> None:
        for logger in self.loggers:
            logger.log_gifs(epoch, paths_dict)

    def log_state_dicts(self, epoch: int, paths_dict: Dict[str, str]) -> None:
        for logger in self.loggers:
            logger.log_state_dicts(epoch, paths_dict)

    def log_metrics(self, epoch: int, metrics_dict: Dict[str, float]) -> None:
        for logger in self.loggers:
            logger.log_metrics(epoch, metrics_dict)

    def finish_run(self) -> None:
        for logger in self.loggers:
            logger.finish_run()
