import functools
import os
import tempfile
from abc import abstractmethod
from typing import Callable, Dict, Set

import gin

import wandb


class AbstractLogger:
    def __init__(
        self,
        epochs: int,
        default_log_freq: int,
        run_params: dict = None,
        advanced_frequencies: Dict[str, int] = {},
        skip_first_metrics: Set[str] = set(),
    ) -> None:
        self.final_epoch = epochs - 1
        self.default_log_freq = default_log_freq
        self.advanced_frequencies = advanced_frequencies
        self.skip_first_metrics = skip_first_metrics

    def _check_epoch(self, epoch: int, freq: int, skip_first: bool = False) -> bool:
        if skip_first and epoch == 0:
            return False
        return (epoch == 0) or (epoch % freq == 0) or (epoch == self.final_epoch)

    def is_log_epoch(self, epoch: int) -> bool:
        return self._check_epoch(epoch, self.default_log_freq)

    def is_metric_epoch(self, epoch: int, metric: str) -> bool:
        freq = self.advanced_frequencies.get(metric, self.default_log_freq)
        skip_first = metric in self.skip_first_metrics
        return self._check_epoch(epoch, freq, skip_first=skip_first)

    @abstractmethod
    def log_gifs(self, epoch: int, paths_dict: Dict[str, str]) -> None:
        pass

    @abstractmethod
    def log_image(self, epoch: int, paths_dict: Dict[str, str], format: str) -> None:
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


def metric_action(name: str) -> Callable[[Callable], Callable]:
    """Decorator that gates method execution based on per-metric logging frequency."""

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(self, epoch: int, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
            if not self.logger.is_metric_epoch(epoch, name):
                return {}
            return fn(self, epoch, *args, **kwargs)

        return wrapper

    return decorator


@gin.configurable
class StdOutLogger(AbstractLogger):
    """Logger class, writing to a file or stdout"""

    def __init__(
        self,
        epochs: int,
        default_log_freq: int,
        run_params: dict = {},
        file: str = None,
        advanced_frequencies: Dict[str, int] = {},
        skip_first_metrics: Set[str] = set(),
    ) -> None:
        super().__init__(
            epochs=epochs,
            default_log_freq=default_log_freq,
            advanced_frequencies=advanced_frequencies,
            skip_first_metrics=skip_first_metrics,
        )
        self.file = open(file, 'a') if file else None
        print('\n-----Starting stdout logging for run-----', file=self.file)
        for name, value in run_params.items():
            print(f'  {name}: {value}', file=self.file)

    def log_gifs(self, epoch: int, paths_dict: Dict[str, str]) -> None:
        print(f'\n-----Gifs for {epoch}-----', file=self.file)
        for name, value in paths_dict.items():
            print(f'{name}: {value}', file=self.file)

    def log_image(self, epoch: int, paths_dict: Dict[str, str], format: str) -> None:
        print(f'\n-----Images ({format}) for {epoch}-----', file=self.file)
        for name, value in paths_dict.items():
            print(f'{name}: {value}', file=self.file)

    def log_state_dicts(self, epoch: int, paths_dict: Dict[str, str]) -> None:
        print(f'\n-----State dicts for {epoch}-----', file=self.file)
        for name, value in paths_dict.items():
            print(f'{name}: {value}', file=self.file)

    def log_metrics(self, epoch: int, metrics_dict: Dict[str, float]) -> None:
        print(f'\n-----Metrics for epoch {epoch}-----', file=self.file)
        for name, value in metrics_dict.items():
            print(f'  {name}: {value}', file=self.file)

    def finish_run(self) -> None:
        print('\n-----Run finished-----', file=self.file)
        if self.file:
            self.file.close()


@gin.configurable
class WandbLogger(AbstractLogger):
    """Logger class, sending all telemetry data to Weights & Biases"""

    def __init__(
        self,
        epochs: int,
        default_log_freq: int,
        run_params: dict = {},
        advanced_frequencies: Dict[str, int] = {},
        skip_first_metrics: Set[str] = set(),
    ) -> None:
        super().__init__(
            epochs=epochs,
            default_log_freq=default_log_freq,
            advanced_frequencies=advanced_frequencies,
            skip_first_metrics=skip_first_metrics,
        )
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

    def log_image(self, epoch: int, paths_dict: Dict[str, str], format: str) -> None:
        self.log_later = self.log_later | {
            f'{name}-epoch-{epoch}': wandb.Image(path) for name, path in paths_dict.items()
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
class BestValOsLogger(AbstractLogger):
    """Logger that tracks the best value of a specified metric and writes it to a temp file."""

    def __init__(
        self,
        epochs: int,
        default_log_freq: int,
        metric_name: str,
        run_params: dict = {},
        advanced_frequencies: Dict[str, int] = {},
        skip_first_metrics: Set[str] = set(),
    ) -> None:
        super().__init__(
            epochs=epochs,
            default_log_freq=default_log_freq,
            advanced_frequencies=advanced_frequencies,
            skip_first_metrics=skip_first_metrics,
        )
        self.metric_name = metric_name
        self.best_val = float('-inf')

    def log_gifs(self, epoch: int, paths_dict: Dict[str, str]) -> None:
        pass

    def log_image(self, epoch: int, paths_dict: Dict[str, str], format: str) -> None:
        pass

    def log_state_dicts(self, epoch: int, paths_dict: Dict[str, str]) -> None:
        pass

    def log_metrics(self, epoch: int, metrics_dict: Dict[str, float]) -> None:
        if self.metric_name in metrics_dict:
            val = metrics_dict[self.metric_name]
            if val > self.best_val:
                self.best_val = val

    @staticmethod
    def result_path(metric_name: str) -> str:
        return os.path.join(tempfile.gettempdir(), f'{metric_name}_max_val.txt')

    def finish_run(self) -> None:
        with open(self.result_path(self.metric_name), 'w') as f:
            f.write(str(self.best_val))


@gin.configurable
class MultipleLoggers(AbstractLogger):
    """Logger class, comprising of multiple sub-loggers"""

    def __init__(
        self,
        epochs: int,
        default_log_freq: int,
        run_params: dict,
        stdout: bool = True,
        wandb: bool = True,
        best_val_os: bool = False,
        advanced_frequencies: Dict[str, int] = {},
        skip_first_metrics: Set[str] = set(),
    ) -> None:
        super().__init__(
            epochs=epochs,
            default_log_freq=default_log_freq,
            advanced_frequencies=advanced_frequencies,
            skip_first_metrics=skip_first_metrics,
        )
        self.loggers: AbstractLogger = []
        if stdout:
            self.loggers.append(
                StdOutLogger(
                    epochs=epochs,
                    default_log_freq=default_log_freq,
                    run_params=run_params,
                    advanced_frequencies=advanced_frequencies,
                    skip_first_metrics=skip_first_metrics,
                )
            )
        if wandb:
            self.loggers.append(
                WandbLogger(
                    epochs=epochs,
                    default_log_freq=default_log_freq,
                    run_params=run_params,
                    advanced_frequencies=advanced_frequencies,
                    skip_first_metrics=skip_first_metrics,
                )
            )
        if best_val_os:
            self.loggers.append(
                BestValOsLogger(
                    epochs=epochs,
                    default_log_freq=default_log_freq,
                    advanced_frequencies=advanced_frequencies,
                    skip_first_metrics=skip_first_metrics,
                )
            )

    def log_gifs(self, epoch: int, paths_dict: Dict[str, str]) -> None:
        for logger in self.loggers:
            logger.log_gifs(epoch, paths_dict)

    def log_image(self, epoch: int, paths_dict: Dict[str, str], format: str) -> None:
        for logger in self.loggers:
            logger.log_image(epoch, paths_dict, format)

    def log_state_dicts(self, epoch: int, paths_dict: Dict[str, str]) -> None:
        for logger in self.loggers:
            logger.log_state_dicts(epoch, paths_dict)

    def log_metrics(self, epoch: int, metrics_dict: Dict[str, float]) -> None:
        for logger in self.loggers:
            logger.log_metrics(epoch, metrics_dict)

    def finish_run(self) -> None:
        for logger in self.loggers:
            logger.finish_run()
