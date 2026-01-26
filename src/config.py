# read config file (and a path helper)

import pathlib
import yaml

# load config
def load_config():
    """
    Go to the repo root and load config/config.yaml

    Returns:
        dict: the loaded config
    """
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "config" / "config.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

# help function to generate path
def repo_path(*parts):
    """
    generate path relative to repo root
    Args:
        *parts: parts of the path relative to repo root
    Returns:
        pathlib.Path: the resolved path
    """
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    return repo_root.joinpath(*parts).resolve()