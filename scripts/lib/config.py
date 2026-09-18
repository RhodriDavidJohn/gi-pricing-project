import argparse
import logging
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_DIR / "config.yaml"


def load_config(config_path=DEFAULT_CONFIG):
    """Read the YAML config and resolve paths relative to the config file's folder."""
    config_path = Path(config_path).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    project_dir = config_path.parent
    for key, value in cfg["paths"].items():
        path = Path(value).expanduser()
        cfg["paths"][key] = path if path.is_absolute() else project_dir / path

    return cfg


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def config_from_command_line(description):
    """Shared command line for the step scripts: an optional --config path."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="path to config.yaml")
    args = parser.parse_args()

    setup_logging()
    return load_config(args.config)
