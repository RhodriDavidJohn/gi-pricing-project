"""XGBoost frequency and severity models.

Two frequency models:
- simple: one Poisson model on ClaimNb with log(Exposure) as an offset (base_margin).
  The offset is exact here because the model is for the total claim count.
- hurdle: expected claims = P(claims > 0) * E[claims | claims > 0]. Exposure is a feature in
  both parts rather than an offset: the chance of a claim isn't proportional to exposure, and a
  policy that has claimed has at least one claim whatever its exposure.

Both frequency classes give `predict_expected_claims(df)` (over each policy's exposure) and
`predict_frequency(df)` (annual), so they can be swapped in the pricing model.

All models use early stopping on an eval set: the validation set during selection, the
holdout set for the final models.
"""

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import ParameterGrid
from xgboost import XGBClassifier, XGBRegressor

from lib.data import SEVERITY_TARGET
from lib.split import severity_rows


def log_exposure(df):
    return np.log(df["Exposure"].clip(lower=1e-6)).to_numpy()


def n_trees(model):
    """Number of trees used for prediction (the best iteration when early stopping was used)."""
    best_iteration = getattr(model, "best_iteration", None)
    return model.n_estimators if best_iteration is None else best_iteration + 1


# ----------------------------------------------------
# Simple frequency model
# ----------------------------------------------------
class SimpleFrequency:
    """The offset is log(Exposure) + log(average claim rate).

    XGBoost ignores its own starting value (base_score) when an offset is given, so the
    portfolio's average claim rate goes into the offset instead - otherwise the trees would
    start from 1 claim per year and spend many rounds just reaching the average.
    """

    def __init__(self, model, features, log_base_rate):
        self.model = model
        self.features = features
        self.log_base_rate = log_base_rate

    def offset(self, df, annual=False):
        log_exp = np.zeros(len(df)) if annual else log_exposure(df)
        return log_exp + self.log_base_rate

    def predict_expected_claims(self, df):
        return self.model.predict(df[self.features], base_margin=self.offset(df))

    def predict_frequency(self, df):
        return self.model.predict(df[self.features], base_margin=self.offset(df, annual=True))


def fit_simple_frequency(df_fit, df_eval, features, params, random_state=42):
    """Poisson model on ClaimNb with a log(Exposure) offset."""
    log_base_rate = float(np.log(df_fit["ClaimNb"].sum() / df_fit["Exposure"].sum()))
    frequency = SimpleFrequency(None, features, log_base_rate)

    model = XGBRegressor(**params, random_state=random_state)
    model.fit(
        df_fit[features], df_fit["ClaimNb"],
        base_margin=frequency.offset(df_fit),
        eval_set=[(df_eval[features], df_eval["ClaimNb"])],
        base_margin_eval_set=[frequency.offset(df_eval)],
        verbose=False,
    )
    frequency.model = model
    return frequency


# ----------------------------------------------------
# Hurdle frequency model
# ----------------------------------------------------
def with_exposure_constraint(params):
    """Hurdle model settings: more exposure can never mean fewer expected claims."""
    constraints = {**params.get("monotone_constraints", {}), "Exposure": 1}
    return {**params, "monotone_constraints": constraints}


def hurdle_X(df, features, annual=False):
    """Hurdle model inputs: the rating features plus Exposure (set to 1 for annual predictions)."""
    X = df[features].copy()
    X["Exposure"] = 1.0 if annual else df["Exposure"]
    return X


class HurdleFrequency:
    def __init__(self, classifier, count_model, features):
        self.classifier = classifier  # calibrated
        self.count_model = count_model
        self.features = features

    @property
    def raw_classifier(self):
        return self.classifier.estimator.estimator

    def predict_claim_probability(self, df, annual=False):
        return self.classifier.predict_proba(hurdle_X(df, self.features, annual))[:, 1]

    def predict_claims_if_claim(self, df, annual=False):
        """E[claims | claims > 0], always at least 1."""
        return 1 + self.count_model.predict(hurdle_X(df, self.features, annual))

    def predict_expected_claims(self, df, annual=False):
        return self.predict_claim_probability(df, annual) * self.predict_claims_if_claim(df, annual)

    def predict_frequency(self, df):
        return self.predict_expected_claims(df, annual=True)


def fit_hurdle_classifier(df_fit, df_eval, features, params, random_state=42):
    """P(claims > 0), with probabilities calibrated on the eval set. Returns the calibrated model."""
    raw_clf = XGBClassifier(**with_exposure_constraint(params), random_state=random_state)
    raw_clf.fit(
        hurdle_X(df_fit, features), df_fit["PosClaims"],
        eval_set=[(hurdle_X(df_eval, features), df_eval["PosClaims"])],
        verbose=False,
    )

    calibrated_clf = CalibratedClassifierCV(FrozenEstimator(raw_clf))
    calibrated_clf.fit(hurdle_X(df_eval, features), df_eval["PosClaims"])
    return calibrated_clf


def fit_hurdle_count(df_fit, df_eval, features, params, random_state=42):
    """Extra claims beyond the first, for policies with at least one claim.

    Modelling ClaimNb - 1 (zero or more) keeps the prediction of ClaimNb at 1 or above.
    """
    fit_pos = df_fit[df_fit["PosClaims"] == 1]
    eval_pos = df_eval[df_eval["PosClaims"] == 1]

    model = XGBRegressor(**with_exposure_constraint(params), random_state=random_state)
    model.fit(
        hurdle_X(fit_pos, features), fit_pos["ClaimNb"] - 1,
        eval_set=[(hurdle_X(eval_pos, features), eval_pos["ClaimNb"] - 1)],
        verbose=False,
    )
    return model


def save_calibration_plot(hurdle, df_test, path):
    fig = Figure(figsize=(12, 6))
    axes = fig.subplots(ncols=2)

    X_test, y_test = hurdle_X(df_test, hurdle.features), df_test["PosClaims"]
    CalibrationDisplay.from_estimator(
        hurdle.raw_classifier, X_test, y_test, n_bins=10, ax=axes[0], name="Raw XGB"
    )
    CalibrationDisplay.from_estimator(
        hurdle.classifier, X_test, y_test, n_bins=10, ax=axes[1], name="Calibrated XGB"
    )

    fig.savefig(path, bbox_inches="tight")


# ----------------------------------------------------
# Severity
# ----------------------------------------------------
class SeverityXGB:
    def __init__(self, model, features):
        self.model = model
        self.features = features

    def predict_severity(self, df):
        return self.model.predict(df[self.features])


def fit_severity(df_fit, df_eval, features, params, random_state=42):
    """(Capped) mean claim amount, weighted by the number of claims per policy."""
    fit_sev, eval_sev = severity_rows(df_fit), severity_rows(df_eval)

    model = XGBRegressor(**params, random_state=random_state)
    model.fit(
        fit_sev[features], fit_sev[SEVERITY_TARGET],
        sample_weight=fit_sev["NbSevClaims"],
        eval_set=[(eval_sev[features], eval_sev[SEVERITY_TARGET])],
        sample_weight_eval_set=[eval_sev["NbSevClaims"]],
        verbose=False,
    )
    return SeverityXGB(model, features)


# ----------------------------------------------------
# Tuning
# ----------------------------------------------------
def model_params(xgb_cfg, model_key, tuned_params=None, n_estimators=None):
    """Full XGBoost settings: shared + fixed settings for the model + tuned parameters + tree settings."""
    return {
        **xgb_cfg.get("common", {}),
        **xgb_cfg[model_key],
        **(tuned_params or {}),
        "n_estimators": n_estimators or xgb_cfg["n_estimators_max"],
        "early_stopping_rounds": xgb_cfg["early_stopping_rounds"],
    }


def grid_search(fit_and_score, param_grid):
    """Try every parameter combination; a lower score is better.

    fit_and_score(params) returns (fitted model, score, number of trees used).
    Returns the best parameters, the best fitted model and a results table.
    """
    rows = []
    best_params, best_model, best_score = None, None, np.inf

    for params in ParameterGrid(param_grid):
        model, score, trees = fit_and_score(params)
        rows.append({**params, "n_trees": trees, "score": score})

        if score < best_score:
            best_params, best_model, best_score = params, model, score

    return best_params, best_model, pd.DataFrame(rows).sort_values("score")
