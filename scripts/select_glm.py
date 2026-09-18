#!/usr/bin/env python3
"""Step 2: choose the GLM frequency and severity models.

Usage: python scripts/select_glm.py [--config config.yaml]

1. Interactions are chosen for each model by forward selection on AIC or BIC (training set).
2. Frequency families (Poisson, negative binomial) and severity families (Gamma, log-normal)
   are compared by validation deviance.
3. The frequency x severity pair with the highest validation pure premium Gini is chosen.
4. The chosen pair is reported on the test set.

Reports and the chosen specification (selected.yaml) are written to <output_dir>/selection/glm/.
"""

import json
import logging

from lib import glm
from lib.config import config_from_command_line
from lib.data import load_policy_data
from lib.evaluation import evaluate_frequency, evaluate_severity
from lib.pricing import evaluate_premium_options
from lib.selection import choose_combination, save_selection, selection_dir
from lib.split import severity_rows, split_policy_data

logger = logging.getLogger(__name__)


def main(cfg):
    glm_cfg = cfg["glm"]
    covariates = glm_cfg["covariates"]
    interactions = list(glm_cfg["interactions_to_test"])
    selection = {"min_improvement": glm_cfg["min_improvement"], "criterion": glm_cfg["criterion"]}

    out_dir = selection_dir(cfg, "glm")
    df_train, df_val, df_test = split_policy_data(load_policy_data(cfg), cfg)
    df_sev_val = severity_rows(df_val)
    metrics = {}

    # ---------------- frequency ----------------
    # interactions chosen with the Poisson model; the negative binomial uses the same terms
    freq_terms, freq_history = glm.select_interactions(
        glm.fit_frequency_glm, df_train, covariates, interactions, **selection
    )
    freq_history.to_csv(out_dir / "interaction_selection_frequency.csv", index=False)

    nb_alpha = glm_cfg["nb_alpha"]
    if nb_alpha is None:
        nb_alpha = glm.estimate_nb_alpha(df_train, freq_terms)

    freq_results = {
        "poisson": glm.fit_frequency_glm(df_train, freq_terms, "poisson"),
        "negative_binomial": glm.fit_frequency_glm(df_train, freq_terms, "negative_binomial", nb_alpha),
    }
    freq_predictors = {
        family: (lambda df, results=results: glm.predict_frequency(results, df))
        for family, results in freq_results.items()
    }
    metrics["frequency"] = {}
    for family, predict in freq_predictors.items():
        annual_frequency = predict(df_val)
        expected_claims = annual_frequency * df_val["Exposure"].to_numpy()
        metrics["frequency"][family] = evaluate_frequency(expected_claims, annual_frequency, df_val)
    metrics["frequency"]["nb_alpha"] = nb_alpha

    # ---------------- severity ----------------
    gamma_terms, gamma_history = glm.select_interactions(
        glm.fit_gamma_glm, df_train, covariates, interactions, **selection
    )
    ln_terms, ln_history = glm.select_interactions(
        glm.fit_log_normal_wls, df_train, covariates, interactions, **selection
    )
    gamma_history.to_csv(out_dir / "interaction_selection_gamma.csv", index=False)
    ln_history.to_csv(out_dir / "interaction_selection_log_normal.csv", index=False)

    gamma_results = glm.fit_gamma_glm(df_train, gamma_terms)
    ln_results = glm.fit_log_normal_wls(df_train, ln_terms)

    sev_terms = {"gamma": gamma_terms, "log_normal": ln_terms}
    sev_predictors = {
        "gamma": lambda df: glm.predict_gamma_severity(gamma_results, df),
        "log_normal": lambda df: glm.predict_log_normal_severity(ln_results, df),
    }

    df_severity = evaluate_severity(
        {family: predict(df_sev_val) for family, predict in sev_predictors.items()}, df_sev_val
    )
    df_severity.to_csv(out_dir / "severity_comparison_val.csv")
    logger.info("Severity models on validation:\n%s", df_severity.round(2).to_string())

    # ---------------- choose the pair ----------------
    freq_family, sev_family, df_pairs = choose_combination(freq_predictors, sev_predictors, df_val)
    df_pairs.to_csv(out_dir / "pair_comparison_val.csv", index=False)
    logger.info("Chosen GLM: %s frequency x %s severity", freq_family, sev_family)

    (out_dir / "frequency_summary.txt").write_text(str(freq_results[freq_family].summary()))
    sev_summary = gamma_results if sev_family == "gamma" else ln_results
    (out_dir / "severity_summary.txt").write_text(str(sev_summary.summary()))

    # ---------------- test report ----------------
    def predict_rate(df):
        return freq_predictors[freq_family](df) * sev_predictors[sev_family](df)

    metrics["test_pure_premium"] = evaluate_premium_options(predict_rate, df_val, df_test, cfg, out_dir)

    spec = {
        "frequency": {"family": freq_family, "terms": freq_terms},
        "severity": {"family": sev_family, "terms": sev_terms[sev_family]},
    }
    metrics["selected"] = spec
    save_selection(spec, cfg, "glm")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("GLM test pure premium:\n%s", json.dumps(metrics["test_pure_premium"], indent=2))

    return metrics


if __name__ == "__main__":
    main(config_from_command_line(__doc__))
