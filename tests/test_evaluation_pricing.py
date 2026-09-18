import numpy as np
import pandas as pd
import pytest

from lib.evaluation import evaluate_pure_premium, gini_index
from lib.pricing import build_tariff_sheet, minimum_premium_from_percentile, rebase_factor


def test_gini_orders_predictions_sensibly():
    rng = np.random.default_rng(0)
    actual = rng.gamma(1, 100, 5000)
    exposure = np.ones_like(actual)

    assert gini_index(actual, actual, exposure) > 0.4
    assert abs(gini_index(actual, rng.random(5000), exposure)) < 0.05
    assert gini_index(actual, -actual, exposure) < 0


def test_evaluate_pure_premium():
    metrics, deciles = evaluate_pure_premium(
        actual_cost=[50, 50, 0, 0], premium_rate=[100, 200, 10, 20], exposure=[0.5] * 4, n_deciles=2
    )
    assert metrics["total_earned_premium"] == 165
    assert metrics["loss_ratio"] == pytest.approx(100 / 165)
    assert deciles["Actual_Cost"].sum() == 100


def test_evaluate_pure_premium_rejects_nan():
    # a NaN premium used to be skipped silently by pandas sums
    with pytest.raises(ValueError):
        evaluate_pure_premium([1, 1], [1, np.nan], [1, 1])


def test_rebase_factor_gives_loss_ratio_of_one():
    factor = rebase_factor([30, 30], [100, 100], [0.5, 0.5])
    assert factor == pytest.approx(0.6)


def test_tariff_sheet_loading_and_minimum_premium():
    df = pd.DataFrame({"IDpol": [1, 2]})
    tariff = build_tariff_sheet(df, [10.0, 1200.0], monthly_loading=0.06, minimum_premium=50)

    assert tariff["Annual_Pure_Premium"].tolist() == [50, 1200]
    assert tariff["Monthly_Pure_Premium"].tolist() == [4.42, 106]
    assert tariff["Implied_Annual_Via_Monthly"].tolist() == [53, 1272]


def test_minimum_premium_from_percentile():
    assert minimum_premium_from_percentile(np.arange(1, 101), 5) == pytest.approx(5.95)
