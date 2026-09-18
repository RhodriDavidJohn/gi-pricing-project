#!/usr/bin/env python3
"""Step 4: train the selected GLM and XGBoost models for live use.

Usage: python scripts/train_final.py [--config config.yaml]

Reads the raw data in `paths.raw_dir` directly, so it can be pointed at new data with a
different config. The models chosen by select_glm.py / select_xgb.py are fitted on all the
data except a holdout (`final.holdout_size`), which is used for XGBoost early stopping and
calibration, the rebase factor and the minimum premium.

Writes <model_dir>/<name>.joblib (load with lib.pricing_model.load_pricing_model) and
<model_dir>/<name>_card.json (what was trained, on what, and how it did on the holdout).
"""

import json
import logging
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split

from lib import gbm, glm
from lib.config import config_from_command_line
from lib.data import build_features, load_raw_data
from lib.evaluation import evaluate_pure_premium
from lib.pricing_model import GLMPricingModel, XGBPricingModel, describe_inputs
from lib.selection import load_selection

logger = logging.getLogger(__name__)


def train_glm(df_fit, df_holdout, spec, cfg, common):
    freq_spec, sev_spec = spec["frequency"], spec["severity"]

    nb_alpha = None
    if freq_spec["family"] == "negative_binomial":
        nb_alpha = cfg["glm"]["nb_alpha"]
        if nb_alpha is None:
            nb_alpha = glm.estimate_nb_alpha(df_fit, freq_spec["terms"])
    freq_results = glm.fit_frequency_glm(df_fit, freq_spec["terms"], freq_spec["family"], nb_alpha)

    if sev_spec["family"] == "gamma":
        sev_results = glm.fit_gamma_glm(df_fit, sev_spec["terms"])
        smearing = 1.0
    else:
        sev_results = glm.fit_log_normal_wls(df_fit, sev_spec["terms"])
        smearing = glm.smearing_factor(sev_results)

    model = GLMPricingModel(
        freq_params=freq_results.params,
        freq_terms=freq_results.terms,
        sev_params=sev_results.params,
        sev_terms=sev_results.terms,
        sev_family=sev_spec["family"],
        smearing_factor=smearing,
        name="glm",
        **common,
    )
    details = {"nb_alpha": nb_alpha, "frequency_aic": freq_results.aic, "severity_aic": sev_results.aic}
    return model, details


def train_xgb(df_fit, df_holdout, spec, cfg, common):
    xgb_cfg = cfg["xgb"]
    features = xgb_cfg["features"]
    seed = cfg["random_state"]
    tuned = spec["tuned_params"]

    def params(model_key):
        return gbm.model_params(xgb_cfg, model_key, tuned[model_key])

    if spec["frequency_type"] == "hurdle":
        frequency = gbm.HurdleFrequency(
            gbm.fit_hurdle_classifier(df_fit, df_holdout, features, params("hurdle_classifier"), seed),
            gbm.fit_hurdle_count(df_fit, df_holdout, features, params("hurdle_count"), seed),
            features,
        )
        trees = {
            "hurdle_classifier": gbm.n_trees(frequency.raw_classifier),
            "hurdle_count": gbm.n_trees(frequency.count_model),
        }
    else:
        frequency = gbm.fit_simple_frequency(df_fit, df_holdout, features, params("simple_count"), seed)
        trees = {"simple_count": gbm.n_trees(frequency.model)}

    severity = gbm.fit_severity(df_fit, df_holdout, features, params("severity"), seed)
    trees["severity"] = gbm.n_trees(severity.model)

    model = XGBPricingModel(
        frequency_model=frequency,
        severity_model=severity,
        frequency_type=spec["frequency_type"],
        name="xgb",
        **common,
    )
    return model, {"n_trees": trees}


TRAINERS = {"glm": train_glm, "xgb": train_xgb}


def main(cfg):
    df_frq_raw, df_sev_raw = load_raw_data(cfg["paths"]["raw_dir"])
    df, info = build_features(df_frq_raw, df_sev_raw, cfg)

    df_fit, df_holdout = train_test_split(
        df,
        test_size=cfg["final"]["holdout_size"],
        stratify=df["PosClaims"],
        random_state=cfg["random_state"],
    )
    logger.info("Final training: fit=%d, holdout=%d", len(df_fit), len(df_holdout))

    model_dir = cfg["paths"]["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)

    common = {
        "feature_cfg": cfg["features"],
        "feature_info": info,
        "input_fields": describe_inputs(df_frq_raw),
        "monthly_loading": cfg["pricing"]["monthly_loading"],
    }

    cards = {}
    for name in cfg["final"]["models"]:
        spec = load_selection(cfg, name)
        model, details = TRAINERS[name](df_fit, df_holdout, spec, cfg, common)
        model.calibrate(df_holdout, cfg["pricing"]["minimum_premium_percentile"])

        # holdout check (the rebase factor was set on this data, so the loss ratio is 1)
        holdout_rate = model.pure_premium(df_holdout)
        holdout_metrics, _ = evaluate_pure_premium(
            df_holdout["TotalClaimAmount"], holdout_rate, df_holdout["Exposure"]
        )

        model.save(model_dir / f"{name}.joblib")
        if hasattr(model, "relativities"):
            model.relativities().to_csv(model_dir / f"{name}_relativities.csv", index=False)

        cards[name] = {
            "model": name,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "raw_data": str(cfg["paths"]["raw_dir"]),
            "large_claim_cap": info["claim_cap"],
            "n_fit": len(df_fit),
            "n_holdout": len(df_holdout),
            "selection": spec,
            **details,
            "rebase_factor": model.rebase_factor,
            "minimum_premium": model.minimum_premium,
            "holdout": holdout_metrics,
            "input_fields": model.input_fields,
            "example_quote": model.quote(model.default_customer()).iloc[0].to_dict(),
        }
        with open(model_dir / f"{name}_card.json", "w") as f:
            json.dump(cards[name], f, indent=2, default=_to_json)

        logger.info("Saved %s model to %s", name, model_dir / f"{name}.joblib")

    return cards


def _to_json(value):
    """numpy / pandas values that json can't write directly."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value.item() if hasattr(value, "item") else str(value)


if __name__ == "__main__":
    main(config_from_command_line(__doc__))
