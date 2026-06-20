import os


def switch_path(path: str) -> str:
    prefix = os.environ.get('RCRL_DATA_PATH_SWITCH')
    if prefix is not None and path is not None and path.startswith('data/'):
        return prefix.rstrip('/') + '/' + path[len('data/') :]
    return path
