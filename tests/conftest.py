"""Shared fixtures: a small synthetic dataset with the freMTPL2 columns."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from lib.config import load_config

PROJECT_DIR = Path(__file__).resolve().parents[1]


def make_raw_data(n: int = 6000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    df_frq = pd.DataFrame({
        "IDpol": np.arange(1, n + 1, dtype=float),
        "Exposure": rng.uniform(0.05, 1.0, n).round(2),
        "Area": rng.choice(list("ABCDEF"), n),
        "VehPower": rng.integers(4, 15, n),
        "VehAge": rng.integers(0, 20, n),
        "DrivAge": rng.integers(18, 90, n),
        "BonusMalus": rng.integers(50, 150, n),
        "VehBrand": rng.choice(["B1", "B2", "B12"], n),
        "VehGas": rng.choice(["'Regular'", "'Diesel'"], n),
        "Density": np.exp(rng.normal(6, 2, n)).round().clip(1),
        "Region": rng.choice(["R11", "R24", "R82"], n),
    })
    rate = np.exp(-2.2 + 0.01 * (df_frq["BonusMalus"] - 100) + 0.4 * (df_frq["DrivAge"] <= 25))
    df_frq["ClaimNb"] = rng.poisson(rate * df_frq["Exposure"])

    claim_ids = np.repeat(df_frq["IDpol"], df_frq["ClaimNb"])
    df_sev = pd.DataFrame({
        "IDpol": claim_ids.to_numpy(),
        "ClaimAmount": rng.gamma(shape=1.5, scale=1200, size=len(claim_ids)).round(2),
    })
    return df_frq, df_sev


@pytest.fixture(scope="session")
def raw_data():
    return make_raw_data()


@pytest.fixture
def cfg(tmp_path, raw_data):
    """The real config, pointed at a temporary project with synthetic raw data."""
    df_frq, df_sev = raw_data
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    df_frq.to_csv(raw_dir / "freMTPL2freq.csv", index=False)
    df_sev.to_csv(raw_dir / "freMTPL2sev.csv", index=False)

    with open(PROJECT_DIR / "config.yaml") as f:
        raw_cfg = yaml.safe_load(f)
    # keep XGBoost quick
    raw_cfg["xgb"]["n_estimators_max"] = 30
    raw_cfg["xgb"]["early_stopping_rounds"] = 5
    raw_cfg["xgb"]["param_grid"] = {"max_depth": [2, 3], "learning_rate": [0.1]}

    with open(tmp_path / "config.yaml", "w") as f:
        yaml.safe_dump(raw_cfg, f)

    return load_config(tmp_path / "config.yaml")


@pytest.fixture
def splits(policy_data, cfg):
    from lib.split import split_policy_data

    return split_policy_data(policy_data, cfg)


@pytest.fixture
def policy_data(cfg, raw_data):
    from lib.data import build_features

    df, _ = build_features(*raw_data, cfg)
    return df
