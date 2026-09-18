import numpy as np
import pandas as pd
import pytest

from lib import analysis, charts


@pytest.fixture
def scored(policy_data):
    rng = np.random.default_rng(3)
    df = policy_data.copy()
    good = df["BonusMalus"] / 100 * 200
    return df, {"GLM": good.to_numpy(), "XGBoost": (good * rng.uniform(0.7, 1.3, len(df))).to_numpy()}


def test_band_factor_labels():
    bands = analysis.band_factor(pd.Series([18, 20, 21, 25, 90]), [18, 21, 26])
    assert bands.astype(str).tolist() == ["18-20", "18-20", "21-25", "21-25", "26+"]


def test_factor_table_totals(scored):
    df, rates = scored
    table = analysis.factor_table(df, "DrivAge", [18, 30, 50], rates)

    assert table["Exposure"].sum() == pytest.approx(df["Exposure"].sum())
    assert table["Actual"].sum() == pytest.approx(df["TotalClaimAmount"].sum())
    assert table["GLMAE"].iloc[0] == pytest.approx(table["Actual"].iloc[0] / table["GLM"].iloc[0])

    # stored-transformed factors come back on their original scale
    assert set(analysis.factor_table(df, "VehGas", None, rates).index) == {"Diesel", "Regular"}
    density = analysis.factor_values(df, "Density")
    assert density.min() >= 1


def test_lift_and_double_lift_tables(scored):
    df, rates = scored
    lift = analysis.lift_table(df["TotalClaimAmount"], rates["GLM"], df["Exposure"], 5)
    assert list(lift.index) == [1, 2, 3, 4, 5]
    # equal exposure bands
    assert lift["Exposure"].max() / lift["Exposure"].min() < 1.1
    assert lift["Predicted"].sum() == pytest.approx(np.sum(rates["GLM"] * df["Exposure"]))

    double = analysis.double_lift_table(
        df["TotalClaimAmount"], rates["GLM"], rates["XGBoost"], df["Exposure"], 5
    )
    assert double["RatioMin"].is_monotonic_increasing
    # indices average to 1 when weighted by exposure
    for col in ["ActualIndex", "AIndex", "BIndex"]:
        assert np.average(double[col], weights=double["Exposure"]) == pytest.approx(1)


def test_dislocation():
    table = analysis.dislocation_table([100, 100, 100, 100], [70, 96, 108, 150])
    assert table["Policies"].tolist() == [1, 0, 0, 1, 1, 0, 1]
    assert table["Share"].sum() == pytest.approx(1)

    summary = analysis.dislocation_summary([100, 100, 100, 100], [70, 96, 108, 150])
    assert summary["share_over_10pct"] == 0.5
    assert summary["share_over_25pct"] == 0.5


def test_claim_size_summary():
    summary = analysis.claim_size_summary(np.arange(1, 101), large_percentile=90)
    assert summary["large_share_of_claims"] == pytest.approx(0.1)
    assert summary["large_share_of_cost"] == pytest.approx(sum(range(91, 101)) / sum(range(1, 101)))


def test_charts_render(scored, tmp_path):
    df, rates = scored
    table = analysis.factor_table(df, "VehBrand", None, rates)
    for ordered in (True, False):
        fig = charts.factor_chart(table, "VehBrand", list(rates), ordered)
        charts.save_figure(fig, tmp_path / f"factor_{ordered}.png")
        assert (tmp_path / f"factor_{ordered}.png").stat().st_size > 1000
