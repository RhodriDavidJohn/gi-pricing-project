import logging

import numpy as np
import pandas as pd
import pytest

from lib.data import (
    band,
    band_labels,
    build_policy_data,
    check_claim_counts,
    clean_frequency_data,
    feature_info,
)
from lib.split import train_val_test_df_split


RAW_ROWS = pd.DataFrame({
    "ClaimNb": [0, 6],
    "Exposure": [0.5, 1.4],
    "VehPower": [7, 12],
    "VehAge": [0, 10],
    "DrivAge": [25, 70],
    "BonusMalus": [100, 190],
    "VehBrand": ["B12", "B1"],
    "VehGas": ["'Regular'", "'Diesel'"],
    "Density": [299, 300],
    "Area": ["'A'", "'D'"],
    "Region": ["R24", "R11"],
})


@pytest.fixture
def info(cfg):
    return feature_info(RAW_ROWS, pd.DataFrame({"ClaimAmount": [100.0, 200.0, 5000.0]}), cfg["features"])


def test_clean_frequency_data_derived_columns(cfg, info):
    out = clean_frequency_data(RAW_ROWS, cfg["features"], info)

    assert out["YoungDrivAge"].tolist() == [1, 0]
    assert out["NewVehAge"].tolist() == [1, 0]
    assert out["PosClaims"].tolist() == [0, 1]
    assert out["LogDensity"].tolist() == pytest.approx(np.log([299, 300]))
    # density is used on the log scale only
    assert "Density" not in out.columns


def test_low_exposure_policies_are_dropped(cfg, info):
    rows = pd.concat([RAW_ROWS, RAW_ROWS.assign(Exposure=[0.003, 0.01])], ignore_index=True)

    out = clean_frequency_data(rows, {**cfg["features"], "min_exposure": 0.02}, info)
    assert out["Exposure"].tolist() == [0.5, 1.0]

    kept = clean_frequency_data(rows, {**cfg["features"], "min_exposure": None}, info)
    assert len(kept) == 4


def test_case_study_features(cfg, info):
    out = clean_frequency_data(RAW_ROWS, cfg["features"], info)

    # caps
    assert out["ClaimNb"].tolist() == [0, 4]
    assert out["Exposure"].tolist() == [0.5, 1.0]
    assert out["BonusMalusCapped"].tolist() == [100, 150]
    assert out["BonusMalus"].tolist() == [100, 190]

    # groups, as categories with every level (not just the ones in the data)
    assert out["DrivAgeBand"].astype(str).tolist() == ["21-25", "51-70"]
    assert out["VehAgeBand"].astype(str).tolist() == ["0", "1-10"]
    assert out["VehPowerBand"].astype(str).tolist() == ["7", "9+"]
    assert list(out["DrivAgeBand"].cat.categories) == ["18-20", "21-25", "26-30", "31-40", "41-50", "51-70", "71+"]

    # quotes removed, text as categories, Area as an ordered code
    assert out["VehGas"].astype(str).tolist() == ["Regular", "Diesel"]
    assert out["AreaCode"].tolist() == [1, 2]
    assert list(out["Region"].cat.categories) == ["R11", "R24"]


def test_single_row_keeps_all_levels(cfg, info):
    # a single quote must be encoded with the same levels as the training data
    out = clean_frequency_data(RAW_ROWS.iloc[[0]], cfg["features"], info)
    assert list(out["VehBrand"].cat.categories) == ["B1", "B12"]
    assert list(out["Area"].cat.categories) == ["A", "D"]


def test_band_labels():
    values = pd.Series([10, 18, 20, 21, 25, 90])
    assert band(values, [18, 21, 26]).tolist() == ["18-20", "18-20", "18-20", "21-25", "21-25", "26+"]
    assert band_labels([0, 1, 11]) == ["0", "1-10", "11+"]


def test_claim_cap(info):
    assert info["claim_cap"] == pytest.approx(np.percentile([100, 200, 5000], 99.5))

    df_frq = pd.DataFrame({"IDpol": [1, 2]})
    df_sev = pd.DataFrame({"IDpol": [1, 1, 2], "ClaimAmount": [100.0, 900.0, 50.0]})
    out = build_policy_data(df_frq, df_sev, claim_cap=500)

    assert out["MeanClaimAmount"].tolist() == [500, 50]
    assert out["MeanClaimAmountCapped"].tolist() == [300, 50]
    # actual cost is never capped
    assert out["TotalClaimAmount"].tolist() == [1000, 50]


def test_build_policy_data_totals_and_counts():
    df_frq = pd.DataFrame({"IDpol": [1, 2, 3], "ClaimNb": [2, 0, 1]})
    df_sev = pd.DataFrame({"IDpol": [1, 1, 3], "ClaimAmount": [100.0, 300.0, 50.0]})

    out = build_policy_data(df_frq, df_sev)

    assert out["MeanClaimAmount"].tolist()[0] == 200
    assert out["TotalClaimAmount"].tolist() == [400, 0, 50]
    assert out["NbSevClaims"].tolist() == [2, 0, 1]
    assert out["sev_flag"].tolist() == [1, 0, 1]


def test_check_claim_counts_logs_disagreements(caplog):
    df = pd.DataFrame({
        "ClaimNb": [1, 0, 2, 1],        # claim with no amount, amount with no claim, counts differ
        "NbSevClaims": [0, 1, 1, 1],
    })

    with caplog.at_level(logging.INFO, logger="lib.data"):
        check_claim_counts(df)

    assert "1 policies with claims but no amounts" in caplog.text
    assert "1 with amounts but no claims" in caplog.text
    assert "1 where the counts differ" in caplog.text


def test_split_is_disjoint_and_stratified():
    df = pd.DataFrame({"x": range(1000), "PosClaims": [1] * 100 + [0] * 900})
    df_train, df_val, df_test = train_val_test_df_split(df, "PosClaims", 0.2, 0.2)

    assert (len(df_train), len(df_val), len(df_test)) == (600, 200, 200)
    assert set(df_train.index).isdisjoint(df_test.index)
    assert set(df_val.index).isdisjoint(df_test.index)
    for part in (df_train, df_val, df_test):
        assert part["PosClaims"].mean() == pytest.approx(0.1)
