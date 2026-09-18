#!/usr/bin/env python3
"""Step 3: choose the XGBoost frequency and severity models.

Usage: python scripts/select_xgb.py [--config config.yaml]

1. Each model is tuned over `xgb.param_grid`, with early stopping on the validation set, and the
   settings with the lowest validation deviance (log loss for the hurdle classifier) are kept.
2. The simple and hurdle frequency models are compared by validation Poisson deviance.
3. The frequency x severity pair with the highest validation pure premium Gini is chosen.
4. The chosen pair is reported on the test set.

Reports and the chosen specification (selected.yaml) are written to <output_dir>/selection/xgb/.
"""

import json
import logging

from lib import gbm
from lib.config import config_from_command_line
from lib.data import SEVERITY_TARGET, load_policy_data
from lib.evaluation import (
    evaluate_classifier,
    evaluate_frequency,
    evaluate_severity,
    gamma_deviance,
    poisson_deviance,
)
from lib.pricing import evaluate_premium_options
from lib.selection import choose_combination, save_selection, selection_dir
from lib.split import severity_rows, split_policy_data

logger = logging.getLogger(__name__)


def main(cfg):
    xgb_cfg = cfg["xgb"]
    features = xgb_cfg["features"]
    grid = xgb_cfg["param_grid"]
    seed = cfg["random_state"]

    out_dir = selection_dir(cfg, "xgb")
    df_train, df_val, df_test = split_policy_data(load_policy_data(cfg), cfg)
    df_val_pos = df_val[df_val["PosClaims"] == 1]
    df_sev_val = severity_rows(df_val)

    tuned, trees, metrics = {}, {}, {}

    def tune(model_key, fit_and_score):
        params, model, df_grid = gbm.grid_search(fit_and_score, grid)
        df_grid.to_csv(out_dir / f"grid_{model_key}.csv", index=False)
        tuned[model_key] = params
        trees[model_key] = int(df_grid.iloc[0]["n_trees"])
        logger.info("%s: best %s (%d trees)", model_key, params, trees[model_key])
        return model

    # ---------------- simple frequency ----------------
    def fit_simple(params):
        model = gbm.fit_simple_frequency(
            df_train, df_val, features, gbm.model_params(xgb_cfg, "simple_count", params), seed
        )
        score = poisson_deviance(df_val["ClaimNb"], model.predict_expected_claims(df_val))
        return model, score, gbm.n_trees(model.model)

    simple = tune("simple_count", fit_simple)

    # ---------------- hurdle frequency ----------------
    def fit_classifier(params):
        clf = gbm.fit_hurdle_classifier(
            df_train, df_val, features, gbm.model_params(xgb_cfg, "hurdle_classifier", params), seed
        )
        proba = clf.predict_proba(gbm.hurdle_X(df_val, features))[:, 1]
        score = evaluate_classifier(df_val["PosClaims"], proba)["log_loss"]
        return clf, score, gbm.n_trees(clf.estimator.estimator)

    def fit_count(params):
        model = gbm.fit_hurdle_count(
            df_train, df_val, features, gbm.model_params(xgb_cfg, "hurdle_count", params), seed
        )
        pred = model.predict(gbm.hurdle_X(df_val_pos, features))
        score = poisson_deviance(df_val_pos["ClaimNb"] - 1, pred)
        return model, score, gbm.n_trees(model)

    hurdle = gbm.HurdleFrequency(
        tune("hurdle_classifier", fit_classifier), tune("hurdle_count", fit_count), features
    )
    gbm.save_calibration_plot(hurdle, df_test, out_dir / "hurdle_calibration_test.png")

    # simplest first, so it wins ties
    freq_models = {"simple": simple, "hurdle": hurdle}
    metrics["frequency"] = {
        name: evaluate_frequency(
            model.predict_expected_claims(df_val), model.predict_frequency(df_val), df_val
        )
        for name, model in freq_models.items()
    }
    metrics["hurdle_classifier_test"] = evaluate_classifier(
        df_test["PosClaims"], hurdle.predict_claim_probability(df_test)
    )

    # ---------------- severity ----------------
    def fit_sev(params):
        model = gbm.fit_severity(
            df_train, df_val, features, gbm.model_params(xgb_cfg, "severity", params), seed
        )
        score = gamma_deviance(
            df_sev_val[SEVERITY_TARGET], model.predict_severity(df_sev_val), df_sev_val["NbSevClaims"]
        )
        return model, score, gbm.n_trees(model.model)

    severity = tune("severity", fit_sev)

    df_severity = evaluate_severity({"xgb": severity.predict_severity(df_sev_val)}, df_sev_val)
    df_severity.to_csv(out_dir / "severity_val.csv")

    # ---------------- choose the pair ----------------
    freq_type, _, df_pairs = choose_combination(
        {name: model.predict_frequency for name, model in freq_models.items()},
        {"xgb": severity.predict_severity},
        df_val,
    )
    df_pairs.to_csv(out_dir / "pair_comparison_val.csv", index=False)
    logger.info("Chosen XGBoost frequency model: %s", freq_type)

    # ---------------- test report ----------------
    def predict_rate(df):
        return freq_models[freq_type].predict_frequency(df) * severity.predict_severity(df)

    metrics["test_pure_premium"] = evaluate_premium_options(predict_rate, df_val, df_test, cfg, out_dir)

    spec = {
        "frequency_type": freq_type,
        "tuned_params": tuned,
        # trees used at the chosen settings (for reference - the final fit uses early stopping)
        "n_trees": trees,
    }
    metrics["selected"] = spec
    save_selection(spec, cfg, "xgb")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("XGBoost test pure premium:\n%s", json.dumps(metrics["test_pure_premium"], indent=2))

    return metrics


if __name__ == "__main__":
    main(config_from_command_line(__doc__))
