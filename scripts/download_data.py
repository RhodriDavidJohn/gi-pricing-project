#!/usr/bin/env python3
"""
Step 0: download the freMTPL2freq (policy, exposure, claim counts) [id 41214]
and freMTPL2sev (claim amounts) [id 41215] datasets from OpenML
via sklearn.datasets.fetch_openml

Usage: python scripts/download_data.py [--config config.yaml] [--force]

Files are saved to `paths.raw_dir`. Files that already exist are kept unless --force is given.
"""

import argparse
import logging
from typing import Dict

import pandas as pd
from sklearn.datasets import fetch_openml

from lib.config import DEFAULT_CONFIG, load_config, setup_logging
from lib.data import FREQ_FILE, SEV_FILE

logger = logging.getLogger(__name__)

OPENML_IDS = {"freq": 41214, "sev": 41215}
FILE_NAMES = {"freq": FREQ_FILE, "sev": SEV_FILE}


def load_data(names=tuple(OPENML_IDS)):
    logger.info("Loading data...")
    return {name: fetch_openml(data_id=OPENML_IDS[name], as_frame="auto").frame for name in names}


def data_info(data: Dict[str, pd.DataFrame]):
    logger.info("=== Data Information ===")

    for name, df in data.items():
        logger.info("  %s", FILE_NAMES[name].removesuffix(".csv"))
        logger.info(f"      Shape:       {df.shape}")
        logger.info(f"      Columns:     {df.columns.to_list()}")
        logger.info(f"      Data Types:  \n{df.dtypes}")
        logger.info(f"      Missingness: \n{df.isna().sum()}")

    logger.info("=" * 25)


def save_data(df: pd.DataFrame, file_path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)


def main(cfg, force=False):
    raw_dir = cfg["paths"]["raw_dir"]
    paths = {name: raw_dir / file_name for name, file_name in FILE_NAMES.items()}

    to_download = [name for name, path in paths.items() if force or not path.exists()]
    if not to_download:
        logger.info("Raw data already in %s - skipping download (use --force to refresh)", raw_dir)
        return paths

    data = load_data(to_download)

    data_info(data)

    for name, df in data.items():
        save_data(df, paths[name])
        logger.info("Saved %s", paths[name])

    logger.info("Data download complete")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    parser.add_argument("--force", action="store_true", help="download even if the files exist")
    args = parser.parse_args()

    setup_logging()
    main(load_config(args.config), force=args.force)
