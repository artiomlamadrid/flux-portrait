import logging
from functools import lru_cache
from pathlib import Path

import toml
import yaml
from dotenv import load_dotenv


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def load_env() -> None:
    load_dotenv(get_project_root() / ".env")


def load_toml(path: str | Path) -> dict:
    return toml.load(path)


def load_yaml(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}
