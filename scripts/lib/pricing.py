"""Pure premium rebasing and tariff sheet."""

import numpy as np
import pandas as pd

from lib.evaluation import evaluate_pure_premium

TEST_PREDICTIONS_FILE = "test_predictions.csv"


def rebase_factor(actual_cost, premium_rate, exposure):
    """Factor that brings the portfolio loss ratio to 1.

    Calculate this on the validation set and apply it to the test set - calculating it
    on the test set makes the test loss ratio exactly 1 by construction.
    """
    earned_premium = np.asarray(premium_rate) * np.asarray(exposure)
    return float(np.sum(actual_cost) / np.sum(earned_premium))


def minimum_premium_from_percentile(premium_rate, percentile):
    """Floor for annual premiums, set at a percentile of the (rebased) premiums."""
    return float(np.percentile(premium_rate, percentile))


def build_tariff_sheet(df, annual_premium_rate, monthly_loading, minimum_premium=None):
    """Annual and monthly premium options per policy."""
    annual = np.asarray(annual_premium_rate, dtype=float)
    if minimum_premium is not None:
        annual = np.maximum(annual, minimum_premium)

    monthly = (annual / 12) * (1 + monthly_loading)

    df_tariff_sheet = pd.DataFrame({
        "IDpol": df["IDpol"].to_numpy(),
        "Annual_Pure_Premium": annual,
        "Monthly_Pure_Premium": monthly,
        "Implied_Annual_Via_Monthly": monthly * 12,
    })

    return df_tariff_sheet.round(2)


def evaluate_premium_options(predict_rate, df_val, df_test, cfg, out_dir):
    """Rebase on validation, build the tariff sheet and evaluate each premium option on test.

    predict_rate: function taking a DataFrame and returning the annual pure premium rate.
    """
    pricing_cfg = cfg["pricing"]

    # rebase so the portfolio loss ratio is 1 on the validation set
    val_rate = predict_rate(df_val)
    factor = rebase_factor(df_val["TotalClaimAmount"], val_rate, df_val["Exposure"])

    # minimum premium from the validation set, so the test set isn't used to set prices
    percentile = pricing_cfg["minimum_premium_percentile"]
    minimum_premium = None
    if percentile is not None:
        minimum_premium = minimum_premium_from_percentile(val_rate * factor, percentile)

    predicted_rate = predict_rate(df_test)
    adjusted_rate = predicted_rate * factor

    df_tariff_sheet = build_tariff_sheet(
        df_test, adjusted_rate, pricing_cfg["monthly_loading"], minimum_premium
    )
    df_tariff_sheet.to_csv(out_dir / "tariff_sheet.csv", index=False)

    # test set predictions, used by report.py
    pd.DataFrame({
        "IDpol": df_test["IDpol"].to_numpy(),
        "PredictedRate": predicted_rate,
        "PremiumRate": adjusted_rate,
    }).to_csv(out_dir / TEST_PREDICTIONS_FILE, index=False)

    premium_options = {
        "predicted": predicted_rate,
        "rebased": adjusted_rate,
        # rebased, with the minimum premium applied (the annual tariff)
        "annual_tariff": df_tariff_sheet["Annual_Pure_Premium"].to_numpy(),
        # worst case for profit: everyone pays monthly
        "all_monthly": df_tariff_sheet["Implied_Annual_Via_Monthly"].to_numpy(),
    }

    metrics = {"rebase_factor": factor, "minimum_premium": minimum_premium}
    if minimum_premium is not None:
        metrics["share_raised_to_minimum"] = float(np.mean(adjusted_rate < minimum_premium))

    for option, rate in premium_options.items():
        metrics[option], decile_summary = evaluate_pure_premium(
            df_test["TotalClaimAmount"], rate, df_test["Exposure"], pricing_cfg["n_deciles"]
        )
        decile_summary.to_csv(out_dir / f"deciles_{option}.csv")

    return metrics
