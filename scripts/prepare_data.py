#!/usr/bin/env python3
"""Step 1: clean the raw freMTPL2 data and save one row per policy.

Usage: python scripts/prepare_data.py [--config config.yaml]
"""

from lib.config import config_from_command_line
from lib.data import prepare_data


def main(cfg):
    prepare_data(cfg)


if __name__ == "__main__":
    main(config_from_command_line(__doc__))
