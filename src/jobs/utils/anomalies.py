import os
from contextlib import contextmanager
from typing import Dict, Iterator

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class AnomalyManager:
    def __init__(self, dir: str, budget: Dict[str, int]) -> None:
        self.dir = dir
        self.actively_logging = False
        self.budget = budget
        self.used = {key: 0 for key in budget}

    def has_budget(self, anm_type: str) -> bool:
        return self.used[anm_type] < self.budget[anm_type]

    @contextmanager
    def open_anomaly_logging(self, anm_type: str) -> Iterator[None]:
        """Context for logging images under the same anomaly"""
        self.actively_logging = True
        self.anm_type = anm_type
        self.log_dir = f'{self.dir}/{anm_type}-{self.used[anm_type]}'

        os.makedirs(self.log_dir, exist_ok=True)
        print(f'Logging anomaly {self.used[anm_type]} to {self.log_dir}')

        try:
            yield
        finally:
            self.actively_logging = False
            self.used[anm_type] += 1

    def log_figure(self, fig: Figure, name: str) -> None:
        """Save a figure within an active logging context"""
        assert self.actively_logging, 'log_figure must be called within a log_anomaly context'
        print(f'Logging figure {name} for anomaly {self.used[self.anm_type]} to {self.log_dir}')
        fig.savefig(os.path.join(self.log_dir, f'{name}.png'))
        plt.close(fig)
