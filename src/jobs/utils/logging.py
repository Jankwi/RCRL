import os
from abc import abstractmethod
from typing import Dict, Set

import gin

import wandb


class AbstractLogger:
    @abstractmethod
    def __init__(self, run_params: dict) -> None:
        pass

    @abstractmethod
    def log_metrics(self, epoch_no: int, metrics_dict: Dict[str, float]) -> None:
        pass

    @abstractmethod
    def finish_run(self) -> None:
        pass


@gin.configurable
class StdOutLogger(AbstractLogger):
    """Logger class, writing everything to standard output"""

    def __init__(self, run_params: dict) -> None:
        print('-----Starting stdout logging for run-----')
        for name, value in run_params.items():
            print(f'  {name}: {value}')

    def log_metrics(self, epoch_no: int, metrics_dict: Dict[str, float]) -> None:
        print(f'-----Metrics for epoch {epoch_no}-----')
        for name, value in metrics_dict.items():
            print(f'  {name}: {value}')

    def finish_run(self) -> None:
        print('-----Run finished-----')


@gin.configurable
class WandbLogger(AbstractLogger):
    """Logger class, sending all telemetry data to Weights & Biases"""

    def __init__(self, run_params: dict) -> None:
        self.wandb_run = wandb.init(
            entity=os.environ.get('wandb_entity'),
            project=os.environ.get('wandb_project'),
            config=run_params,
        )

    def log_metrics(self, epoch_no: int, metrics_dict: Dict[str, float]) -> None:
        self.wandb_run.log(metrics_dict)

    def finish_run(self) -> None:
        self.wandb_run.finish()


class MultipleLoggers(AbstractLogger):
    """Logger class, comprising of multiple sub-loggers"""

    def __init__(self, run_params: dict, logger_names: Set[str]) -> None:
        self.loggers = []
        if 'stdout' in logger_names:
            self.loggers.append(StdOutLogger(run_params))
        if 'wandb' in logger_names:
            self.loggers.append(WandbLogger(run_params))

    def log_metrics(self, epoch_no: int, metrics_dict: Dict[str, float]) -> None:
        for logger in self.loggers:
            logger.log_metrics(epoch_no, metrics_dict)

    def finish_run(self) -> None:
        for logger in self.loggers:
            logger.finish_run()
