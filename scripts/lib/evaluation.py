"""Model evaluation metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_gamma_deviance,
    mean_poisson_deviance,
    mean_squared_error,
    precision_recall_curve,
)

from lib.data import SEVERITY_TARGET


def lorenz_curve(actual, predicted, exposure):
    """Cumulative share of exposure and of actual, from lowest to highest predicted risk."""
    order = np.argsort(np.asarray(predicted), kind="stable")
    actual = np.asarray(actual, dtype=float)[order]
    exposure = np.asarray(exposure, dtype=float)[order]

    # cumulative shares, starting the curve at (0, 0)
    cum_exposure = np.concatenate([[0], exposure.cumsum() / exposure.sum()])
    cum_actual = np.concatenate([[0], actual.cumsum() / actual.sum()])
    return cum_exposure, cum_actual


def gini_index(actual, predicted, exposure):
    """Actuarial Gini index: 1 - 2 * area under the Lorenz curve."""
    cum_exposure, cum_actual = lorenz_curve(actual, predicted, exposure)
    return float(1 - 2 * np.trapezoid(cum_actual, cum_exposure))


def poisson_deviance(claims, expected_claims):
    return float(mean_poisson_deviance(claims, expected_claims))


def gamma_deviance(actual, predicted, weights):
    # gamma deviance needs strictly positive amounts
    actual = np.clip(np.asarray(actual, dtype=float), 1e-6, None)
    return float(mean_gamma_deviance(actual, predicted, sample_weight=weights))


def evaluate_frequency(expected_claims, annual_frequency, df_test):
    """Poisson deviance and MSE on claim counts, and Gini on the predicted annual frequency.

    expected_claims: predicted claims over each policy's exposure.
    """
    return {
        "poisson_deviance": poisson_deviance(df_test["ClaimNb"], expected_claims),
        "mse": float(mean_squared_error(df_test["ClaimNb"], expected_claims)),
        "gini": gini_index(df_test["ClaimNb"], annual_frequency, df_test["Exposure"]),
    }


def evaluate_severity(predictions, df_sev_test):
    """Compare severity models. `predictions` maps model name -> predicted mean claim amount."""
    # models are fitted to capped claim amounts, so they are compared with capped amounts
    actual = df_sev_test[SEVERITY_TARGET].to_numpy()
    weights = df_sev_test["NbSevClaims"].to_numpy()

    metrics = {}
    for name, pred in predictions.items():
        metrics[name] = {
            "Gamma Deviance": gamma_deviance(actual, pred, weights),
            "MAE": mean_absolute_error(actual, pred, sample_weight=weights),
            "RMSE": np.sqrt(mean_squared_error(actual, pred, sample_weight=weights)),
            "Total Predicted Portfolio Cost": np.sum(pred * weights),
            "Actual Portfolio Cost (capped claims)": np.sum(actual * weights),
        }

    return pd.DataFrame(metrics)


def evaluate_classifier(y_true, proba):
    precision, recall, _ = precision_recall_curve(y_true, proba)
    return {
        "log_loss": float(log_loss(y_true, proba)),
        "brier_score": float(brier_score_loss(y_true, proba)),
        "pr_auc_average_precision": float(average_precision_score(y_true, proba)),
        "pr_auc_trapezoidal": float(auc(recall, precision)),
    }


def evaluate_pure_premium(actual_cost, premium_rate, exposure, n_deciles=10):
    """Portfolio loss ratio, Gini and decile lift table for an annual premium rate."""
    df_eval = pd.DataFrame({
        "Exposure": np.asarray(exposure),
        "Actual_Cost": np.asarray(actual_cost),
        "Premium_Rate": np.asarray(premium_rate),
    })
    if df_eval["Premium_Rate"].isna().any():
        raise ValueError("Premium rate contains NaN values")

    # premium earned for the time each policy was active
    df_eval["Earned_Pure_Premium"] = df_eval["Premium_Rate"] * df_eval["Exposure"]

    total_actual_loss = df_eval["Actual_Cost"].sum()
    total_earned_premium = df_eval["Earned_Pure_Premium"].sum()

    metrics = {
        "total_actual_loss": float(total_actual_loss),
        "total_earned_premium": float(total_earned_premium),
        "loss_ratio": float(total_actual_loss / total_earned_premium),
        "gini": gini_index(df_eval["Actual_Cost"], df_eval["Premium_Rate"], df_eval["Exposure"]),
    }

    # group policies into equal buckets based on predicted premium rate
    df_eval["Decile"] = pd.qcut(df_eval["Premium_Rate"], q=n_deciles, labels=False, duplicates="drop")
    decile_summary = df_eval.groupby("Decile").agg(
        Exposure=("Exposure", "sum"),
        Actual_Cost=("Actual_Cost", "sum"),
        Earned_Pure_Premium=("Earned_Pure_Premium", "sum"),
    )

    # observed vs predicted burning cost per unit of exposure
    decile_summary["Observed_Burning_Cost"] = decile_summary["Actual_Cost"] / decile_summary["Exposure"]
    decile_summary["Predicted_Burning_Cost"] = (
        decile_summary["Earned_Pure_Premium"] / decile_summary["Exposure"]
    )

    return metrics, decile_summary
