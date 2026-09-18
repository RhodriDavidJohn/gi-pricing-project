#!/usr/bin/env python3
"""Run the pipeline steps in order:
download -> prepare -> select_glm -> select_xgb -> train_final -> report

Usage:
    python scripts/run_pipeline.py                              # all steps
                                                                # (download skips files that exist)
    python scripts/run_pipeline.py --steps select_glm select_xgb  # some steps
    python scripts/run_pipeline.py --steps train_final --config live_config.yaml
"""

import argparse
import logging
from pathlib import Path

import download_data
import prepare_data
import report
import select_glm
import select_xgb
import train_final
from lib.config import DEFAULT_CONFIG, load_config, setup_logging

logger = logging.getLogger(__name__)

STEPS = {
    "download": download_data.main,
    "prepare": prepare_data.main,
    "select_glm": select_glm.main,
    "select_xgb": select_xgb.main,
    "train_final": train_final.main,
    "report": report.main,
}


def run(cfg, steps=tuple(STEPS)):
    results = {}
    for step in STEPS:  # always run in pipeline order
        if step in steps:
            logger.info("---------- %s ----------", step)
            results[step] = STEPS[step](cfg)
    return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="path to config.yaml")
    parser.add_argument("--steps", nargs="+", choices=list(STEPS), default=list(STEPS))
    args = parser.parse_args()

    setup_logging()
    run(load_config(args.config), args.steps)


if __name__ == "__main__":
    main()
