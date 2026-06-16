import argparse
from typing import Dict, Type

import gin

from src.common import fix_seed
from src.jobs import AbstractJob
from src.jobs.augment_preview_job import *  # noqa: F403
from src.jobs.solved_rate_job import *  # noqa: F403
from src.jobs.timing_job import *  # noqa: F403
from src.jobs.train_job import *  # noqa: F403
from src.jobs.validation_job import *  # noqa: F403


def read_bindings_files(bindings_files: list) -> list:
    bindings = []
    for bindings_file in bindings_files:
        with open(bindings_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    bindings.append(line)
    return bindings


def gin_config_to_dict(config_str: str) -> Dict[str, str]:
    result = {}
    lines = iter(config_str.splitlines())
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            while line.endswith('\\'):
                line = line.rstrip('\\') + next(lines).strip()
            key, _, value = line.partition(' = ')
            result[key] = value
    return result


@gin.configurable
def run(
    seed: int,
    job_class: Type[AbstractJob],
    gin_config: Dict[str, str],
    name: str = 'no name set',
    family: str = 'undeclared',
    tags: Dict[str, str] = {},
):
    fix_seed(seed)
    for tag_key, tag_value in tags.items():
        gin_config[f'tags.{tag_key}'] = tag_value
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
        help='Gin bindings like "run.TrainJob.seed=1"',
    )
    parser.add_argument(
        '--binding_files',
        nargs='*',
        default=[],
        metavar='FILE',
        help='Path(s) to gin file(s); each non-empty, non-comment line is treated as a binding',
    )
    parser.add_argument(
        '--only_parse',
        action='store_true',
        default=False,
        help="Don't run the job, just check if config parses OK",
    )

    args = parser.parse_args()
    bindings = read_bindings_files(args.binding_files) + args.gin_bindings
    gin.parse_config_files_and_bindings(config_files=[args.config_file], bindings=bindings)

    if not args.only_parse:
        print('====Starting run====')
        run(gin_config=gin_config_to_dict(gin.config_str()))
