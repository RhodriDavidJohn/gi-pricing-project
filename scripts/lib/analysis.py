"""Tables behind the report charts. All take test set actuals and annual premium rates."""

import numpy as np
import pandas as pd

from lib.data import band, band_labels

# ----------------------------------------------------
# Rating factors
# ----------------------------------------------------


def factor_values(df, factor):
    """A customer detail as it appears on the policy data (Density is stored as its log)."""
    if factor == "Density":
        return np.exp(df["LogDensity"]).round()
    return df[factor]


def band_factor(values, edges=None):
    """Band a factor with readable labels ("18-20", "71+"). Without edges, each value is its own band."""
    if edges is None:
        return values
    return pd.Categorical(band(values, edges), categories=band_labels(edges))


def factor_table(df, factor, edges, rates):
    """Actual vs predicted burning cost (cost per year of exposure) by band of one factor.

    rates: dict of model name -> annual premium rate for each row of df.
    """
    data = pd.DataFrame({
        "Band": band_factor(factor_values(df, factor), edges),
        "Exposure": df["Exposure"].to_numpy(),
        "Claims": df["ClaimNb"].to_numpy(),
        "Actual": df["TotalClaimAmount"].to_numpy(),
    })
    for name, rate in rates.items():
        data[name] = np.asarray(rate) * data["Exposure"]

    table = data.groupby("Band", observed=True).sum()
    table["ClaimFrequency"] = table["Claims"] / table["Exposure"]
    for col in ["Actual", *rates]:
        table[f"{col}BurningCost"] = table[col] / table["Exposure"]
    for name in rates:
        table[f"{name}AE"] = table["Actual"] / table[name]

    table.index = table.index.astype(str)
    return table


# ----------------------------------------------------
# Ranking
# ----------------------------------------------------


def lorenz_table(curves, n_points=201):
    """Lorenz curves at evenly spaced exposure shares (small enough to commit and chart)."""
    grid = np.linspace(0, 1, n_points)
    table = pd.DataFrame({"ExposureShare": grid})
    for name, (cum_exposure, cum_actual) in curves.items():
        table[name] = np.interp(grid, cum_exposure, cum_actual)
    return table


def lift_table(actual, rate, exposure, n_bins=10):
    """Actual vs predicted burning cost by band of predicted rate (equal exposure bands)."""
    data = pd.DataFrame({
        "Rate": np.asarray(rate),
        "Exposure": np.asarray(exposure),
        "Actual": np.asarray(actual),
    }).sort_values("Rate", kind="stable")
    data["Predicted"] = data["Rate"] * data["Exposure"]
    data["Band"] = _exposure_bands(data["Exposure"], n_bins)

    table = data.groupby("Band")[["Exposure", "Actual", "Predicted"]].sum()
    table["ActualBurningCost"] = table["Actual"] / table["Exposure"]
    table["PredictedBurningCost"] = table["Predicted"] / table["Exposure"]
    return table


def double_lift_table(actual, rate_a, rate_b, exposure, n_bins=10):
    """Bands of the ratio rate_b / rate_a (equal exposure). In each band, actual cost and each
    model's premium are shown relative to the overall average burning cost.

    Where the models disagree most (the outer bands), the model whose line follows the
    actual line more closely is the better one.
    """
    data = pd.DataFrame({
        "Ratio": np.asarray(rate_b) / np.asarray(rate_a),
        "Exposure": np.asarray(exposure),
        "Actual": np.asarray(actual),
        "A": np.asarray(rate_a) * np.asarray(exposure),
        "B": np.asarray(rate_b) * np.asarray(exposure),
    }).sort_values("Ratio", kind="stable")
    data["Band"] = _exposure_bands(data["Exposure"], n_bins)

    table = data.groupby("Band").agg(
        Exposure=("Exposure", "sum"),
        Actual=("Actual", "sum"),
        A=("A", "sum"),
        B=("B", "sum"),
        RatioMin=("Ratio", "min"),
        RatioMax=("Ratio", "max"),
    )
    for col in ["Actual", "A", "B"]:
        average = table[col].sum() / table["Exposure"].sum()
        table[f"{col}Index"] = table[col] / table["Exposure"] / average
    return table


def _exposure_bands(sorted_exposure, n_bins):
    """Band number for rows already sorted by the ranking variable, so each band has
    (about) the same exposure."""
    cum_share = sorted_exposure.cumsum() / sorted_exposure.sum()
    return np.minimum((cum_share * n_bins).to_numpy().astype(int), n_bins - 1) + 1


# ----------------------------------------------------
# Price changes and claims
# ----------------------------------------------------

CHANGE_EDGES = [-np.inf, -0.25, -0.10, -0.05, 0.05, 0.10, 0.25, np.inf]
CHANGE_LABELS = ["< -25%", "-25% to -10%", "-10% to -5%", "-5% to +5%", "+5% to +10%", "+10% to +25%", "> +25%"]


def dislocation_table(rate_from, rate_to):
    """Share of policies by % change in premium when moving from one model to another."""
    change = np.asarray(rate_to) / np.asarray(rate_from) - 1
    bands = pd.cut(change, CHANGE_EDGES, labels=CHANGE_LABELS, right=False)

    table = pd.Series(bands).value_counts(sort=False).rename("Policies").to_frame()
    table["Share"] = table["Policies"] / table["Policies"].sum()
    return table


def dislocation_summary(rate_from, rate_to):
    change = np.asarray(rate_to) / np.asarray(rate_from) - 1
    return {
        "median_change": float(np.median(change)),
        "share_up_over_10pct": float(np.mean(change > 0.10)),
        "share_down_over_10pct": float(np.mean(change < -0.10)),
        "share_over_10pct": float(np.mean(np.abs(change) > 0.10)),
        "share_over_25pct": float(np.mean(np.abs(change) > 0.25)),
    }


def claim_size_summary(claim_amounts, large_percentile=99):
    amounts = np.sort(np.asarray(claim_amounts, dtype=float))
    threshold = float(np.percentile(amounts, large_percentile))
    large = amounts[amounts > threshold]
    return {
        "n_claims": int(len(amounts)),
        "mean": float(amounts.mean()),
        "median": float(np.median(amounts)),
        "max": float(amounts.max()),
        "large_percentile": large_percentile,
        "large_threshold": threshold,
        "large_share_of_claims": float(len(large) / len(amounts)),
        "large_share_of_cost": float(large.sum() / amounts.sum()),
    }
