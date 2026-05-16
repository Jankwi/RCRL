import argparse
from typing import Dict, Type

import gin

from src.common import fix_seed
from src.jobs import AbstractJob
from src.jobs.timing_job import *  # noqa: F403
from src.jobs.train_job import *  # noqa: F403
from src.jobs.validation_job import *  # noqa: F403


def gin_config_to_dict(config_str: str) -> Dict[str, str]:
    result = {}
    for line in config_str.splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            key, _, value = line.partition(' = ')
            result[key] = value
    return result


@gin.configurable
def run(seed: int, job_class: Type[AbstractJob], gin_config: Dict[str, str]):
    fix_seed(seed)
    job = job_class(gin_config=gin_config)
    job.execute()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=True, help='Path to the config file')
    parser.add_argument(
        '--gin_bindings',
        nargs='*',
        default=[],
        metavar='BIND',
        help='Gin bindings like "run.Trainjob.seed=1"',
    )

    args = parser.parse_args()
    gin.parse_config_files_and_bindings(config_files=[args.config_file], bindings=args.gin_bindings)

    print('====Starting run====')
    run(gin_config=gin_config_to_dict(gin.config_str()))
