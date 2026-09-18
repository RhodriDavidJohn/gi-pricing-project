#!/usr/bin/env python3
"""Step 6: charts and a results write-up comparing the selected GLM and XGBoost models.

Usage: python scripts/report.py [--config config.yaml]

Uses the test set predictions saved by select_glm.py / select_xgb.py (the same policies for
both models) and, for the price response charts, the final models from train_final.py.

Writes to <report_dir>:
    results.md      the write-up - every number in it is calculated from this run
    figures/*.png   charts used in results.md (and the README)
    tables/*.csv    the numbers behind the charts
"""

import json
import logging
import shutil
from datetime import date

import pandas as pd
import yaml

from lib import analysis, charts
from lib.config import config_from_command_line
from lib.data import FEATURE_INFO_FILE, INPUT_COLUMNS, load_policy_data, load_raw_data
from lib.evaluation import gini_index, lorenz_curve
from lib.pricing import TEST_PREDICTIONS_FILE
from lib.pricing_model import load_pricing_model
from lib.selection import SELECTED_FILE
from lib.split import split_policy_data

logger = logging.getLogger(__name__)

MODELS = {"glm": "GLM", "xgb": "XGBoost"}


# ----------------------------------------------------
# Loading
# ----------------------------------------------------
def load_selection_outputs(cfg):
    """Metrics, chosen specification and test predictions for each model."""
    outputs = {}
    for key, name in MODELS.items():
        sel_dir = cfg["paths"]["output_dir"] / "selection" / key
        if not (sel_dir / TEST_PREDICTIONS_FILE).exists():
            raise FileNotFoundError(f"No test predictions in {sel_dir} - run select_{key}.py first")

        outputs[name] = {
            "dir": sel_dir,
            "metrics": json.loads((sel_dir / "metrics.json").read_text()),
            "spec": yaml.safe_load((sel_dir / SELECTED_FILE).read_text()),
            "predictions": pd.read_csv(sel_dir / TEST_PREDICTIONS_FILE),
        }
    return outputs


def load_final_models(cfg):
    models = {}
    for key, name in MODELS.items():
        path = cfg["paths"]["model_dir"] / f"{key}.joblib"
        if path.exists():
            models[name] = load_pricing_model(path)
    return models


def test_set_with_predictions(cfg, outputs):
    _, _, df_test = split_policy_data(load_policy_data(cfg), cfg)
    df_test = df_test.reset_index(drop=True)

    for name, output in outputs.items():
        preds = output["predictions"][["IDpol", "PremiumRate"]].rename(columns={"PremiumRate": name})
        df_test = df_test.merge(preds, on="IDpol", how="left", validate="one_to_one")
        if df_test[name].isna().any():
            raise ValueError(
                f"{name} test predictions don't match the current test set - re-run the selection scripts"
            )
    return df_test


# ----------------------------------------------------
# Markdown helpers
# ----------------------------------------------------
def md_table(df, index=True):
    df = df.reset_index() if index else df
    lines = [
        "| " + " | ".join(str(c) for c in df.columns) + " |",
        "|" + "|".join("---" for _ in df.columns) + "|",
    ]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def pct(value, decimals=1):
    return f"{value:.{decimals}%}"


def money(value):
    return f"{value:,.0f}"


def figure(file_name, alt):
    return f"![{alt}](figures/{file_name})"


FAMILY_NAMES = {
    "poisson": "Poisson",
    "negative_binomial": "negative binomial",
    "gamma": "Gamma",
    "log_normal": "log-normal",
    "hurdle": "hurdle",
    "simple": "simple",
    "xgb": "XGBoost",
}


def cleaning_note(df_frq_raw, cfg):
    """One bullet for the policies removed for having almost no exposure (none if that's off)."""
    min_exposure = cfg["features"].get("min_exposure")
    if not min_exposure:
        return []
    dropped = df_frq_raw["Exposure"] < min_exposure
    return [
        f"- **Cleaning:** {dropped.sum():,} policies ({pct(dropped.mean())}) with less than "
        f"{min_exposure} years of exposure were removed before modelling."
    ]


def large_claim_note(cfg):
    percentile = cfg["features"].get("large_claim_percentile")
    if percentile is None:
        return ("- Large claims aren't capped or modelled separately, so severity results and test loss "
                "ratios are sensitive to a few big claims.")
    info = json.loads((cfg["paths"]["processed_dir"] / FEATURE_INFO_FILE).read_text())
    return (f"- Severity models are fitted to claim amounts capped at the {percentile}th percentile "
            f"({money(info['claim_cap'])}); the cost above the cap is spread over all policies by the "
            "rebase factor rather than priced by risk. Test loss ratios use uncapped claims, so a few "
            "big claims still move them.")


def interactions(terms, cfg):
    extra = [t for t in terms if t not in cfg["glm"]["covariates"]]
    return ", ".join(f"`{t}`" for t in extra) if extra else "none"


# ----------------------------------------------------
# Report
# ----------------------------------------------------
def main(cfg):
    report_cfg = cfg["report"]
    report_dir = cfg["paths"]["report_dir"]
    fig_dir, table_dir = report_dir / "figures", report_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    outputs = load_selection_outputs(cfg)
    df_test = test_set_with_predictions(cfg, outputs)
    final_models = load_final_models(cfg)
    df_frq_raw, df_sev_raw = load_raw_data(cfg["paths"]["raw_dir"])

    names = list(MODELS.values())
    rates = {name: df_test[name].to_numpy() for name in names}
    actual, exposure = df_test["TotalClaimAmount"], df_test["Exposure"]
    n_bands = report_cfg["n_bands"]

    def save(fig, file_name):
        charts.save_figure(fig, fig_dir / file_name)
        logger.info("Saved %s", fig_dir / file_name)

    # ---------------- ranking ----------------
    ginis = {name: gini_index(actual, rates[name], exposure) for name in names}
    curves = {name: lorenz_curve(actual, rates[name], exposure) for name in names}
    save(charts.lorenz_chart(curves, ginis), "lorenz.png")
    analysis.lorenz_table(curves).to_csv(table_dir / "lorenz.csv", index=False)

    lift_tables = {name: analysis.lift_table(actual, rates[name], exposure, n_bands) for name in names}
    for name, table in lift_tables.items():
        table.to_csv(table_dir / f"lift_{name}.csv")
    save(charts.lift_chart(lift_tables), "lift.png")

    double_lift = analysis.double_lift_table(actual, rates["GLM"], rates["XGBoost"], exposure, n_bands)
    double_lift.to_csv(table_dir / "double_lift.csv")
    save(charts.double_lift_chart(double_lift), "double_lift.png")

    # ---------------- rating factors ----------------
    factor_rows = []
    factors_done = []
    for factor, edges in report_cfg["factor_bands"].items():
        try:
            table = analysis.factor_table(df_test, factor, edges, rates)
        except KeyError:
            logger.warning("Skipping %s - not in the data", factor)
            continue

        table.to_csv(table_dir / f"ae_{factor}.csv")
        ordered = edges is not None or pd.api.types.is_numeric_dtype(analysis.factor_values(df_test, factor))
        ordered = ordered or factor in report_cfg["ordered_categories"]
        save(charts.factor_chart(table, factor, names, ordered), f"ae_{factor}.png")
        factors_done.append(factor)

        # largest actual vs expected gap in bands with at least 5% of the exposure
        big = table[table["Exposure"] / table["Exposure"].sum() >= 0.05]
        row = {"Factor": factor, "Used by models": "yes" if factor in INPUT_COLUMNS else "**no**"}
        for name in names:
            gap = (big[f"{name}AE"] - 1).abs()
            band = gap.idxmax()
            row[f"{name}: largest A/E gap"] = f"{big.loc[band, f'{name}AE']:.2f} ({band})"
        factor_rows.append(row)
    df_factors = pd.DataFrame(factor_rows)

    # ---------------- price changes ----------------
    dislocation = analysis.dislocation_table(rates["GLM"], rates["XGBoost"])
    dislocation.to_csv(table_dir / "dislocation.csv")
    dislocation_summary = analysis.dislocation_summary(rates["GLM"], rates["XGBoost"])
    save(charts.dislocation_chart(dislocation), "dislocation.png")

    # ---------------- final models: price responses and explanations ----------------
    explanations = {}
    if len(final_models) == len(MODELS):
        default = final_models["GLM"].default_customer()
        price_curves = {
            detail: {name: model.price_curve(default, detail) for name, model in final_models.items()}
            for detail in INPUT_COLUMNS
        }
        save(charts.price_curves_chart(price_curves, INPUT_COLUMNS), "price_curves.png")

        for example, details in report_cfg["example_customers"].items():
            customer = {**default, **(details or {})}
            explanations[example] = {}
            for key, name in MODELS.items():
                explanation = final_models[name].explain(customer)
                explanation["effects"].to_csv(table_dir / f"explain_{example}_{key}.csv")
                save(charts.effects_chart(explanation, customer, name), f"explain_{example}_{key}.png")
                explanations[example][name] = (customer, explanation)
    else:
        logger.warning("Final models not found in %s - skipping price response charts",
                       cfg["paths"]["model_dir"])

    # ---------------- claims ----------------
    claims = analysis.claim_size_summary(df_sev_raw["ClaimAmount"], report_cfg["large_claim_percentile"])
    save(charts.claim_size_chart(df_sev_raw["ClaimAmount"], claims), "claim_sizes.png")

    calibration_plot = outputs["XGBoost"]["dir"] / "hurdle_calibration_test.png"
    if outputs["XGBoost"]["spec"]["frequency_type"] == "hurdle" and calibration_plot.exists():
        shutil.copy(calibration_plot, fig_dir / "hurdle_calibration.png")

    # ---------------- write-up ----------------
    text = write_results(
        cfg, outputs, df_frq_raw, df_sev_raw, df_test, ginis, double_lift, df_factors,
        factors_done, dislocation, dislocation_summary, explanations, claims,
        has_price_curves=len(final_models) == len(MODELS),
        has_calibration=(fig_dir / "hurdle_calibration.png").exists(),
    )
    (report_dir / "results.md").write_text(text)
    logger.info("Wrote %s", report_dir / "results.md")

    return {"gini": ginis, "dislocation": dislocation_summary, "claims": claims}


def write_results(cfg, outputs, df_frq_raw, df_sev_raw, df_test, ginis, double_lift, df_factors,
                  factors_done, dislocation, dislocation_summary, explanations, claims,
                  has_price_curves, has_calibration):
    glm_out, xgb_out = outputs["GLM"], outputs["XGBoost"]
    glm_spec, xgb_spec = glm_out["spec"], xgb_out["spec"]
    test_pp = {name: out["metrics"]["test_pure_premium"] for name, out in outputs.items()}

    # headline comparisons
    better = max(ginis, key=ginis.get)
    worse = min(ginis, key=ginis.get)
    gini_gap = ginis[better] - ginis[worse]

    outer = double_lift.loc[[double_lift.index.min(), double_lift.index.max()]]
    gap_glm = (outer["AIndex"] - outer["ActualIndex"]).abs().mean()
    gap_xgb = (outer["BIndex"] - outer["ActualIndex"]).abs().mean()
    closer = "GLM" if gap_glm < gap_xgb else "XGBoost"

    total_exposure = df_frq_raw["Exposure"].sum()
    total_cost = df_sev_raw["ClaimAmount"].sum()

    lines = [
        "# Results",
        "",
        f"_Generated by `scripts/report.py` on {date.today():%d %B %Y}. Every number below is "
        "calculated from this run; the Discussion section is for the author's interpretation._",
        "",
        "## Summary",
        "",
        f"- **Data:** {len(df_frq_raw):,} policies, {total_exposure:,.0f} policy years, "
        f"{df_frq_raw['ClaimNb'].sum():,.0f} claims and {money(total_cost)} of claims cost "
        f"({df_frq_raw['ClaimNb'].sum() / total_exposure:.3f} claims and "
        f"{money(total_cost / total_exposure)} of cost per policy year).",
        *cleaning_note(df_frq_raw, cfg),
        f"- **Test set:** {len(df_test):,} policies ({pct(cfg['split']['test_size'], 0)}), "
        "used only for the results below.",
        f"- **Selected GLM:** {FAMILY_NAMES[glm_spec['frequency']['family']]} frequency × "
        f"{FAMILY_NAMES[glm_spec['severity']['family']]} severity. Interactions: frequency "
        f"{interactions(glm_spec['frequency']['terms'], cfg)}; severity "
        f"{interactions(glm_spec['severity']['terms'], cfg)}.",
        f"- **Selected XGBoost:** {xgb_spec['frequency_type']} frequency × severity with "
        f"`{cfg['xgb']['severity']['objective']}` objective.",
        f"- **Ranking:** {better} has the higher test Gini ({ginis[better]:.4f} vs "
        f"{ginis[worse]:.4f}, a difference of {gini_gap:.4f}).",
        f"- **Where the models disagree** (outer double lift bands), the {closer} premium is closer to "
        f"actual experience (average gap {100 * min(gap_glm, gap_xgb):.1f} vs "
        f"{100 * max(gap_glm, gap_xgb):.1f} percentage points of average cost).",
        f"- **Loss ratio on test** after rebasing on the validation set: GLM "
        f"{test_pp['GLM']['rebased']['loss_ratio']:.3f}, XGBoost {test_pp['XGBoost']['rebased']['loss_ratio']:.3f}.",
        f"- **Switching from GLM to XGBoost** would change the premium by more than 10% for "
        f"{pct(dislocation_summary['share_over_10pct'])} of test policies "
        f"({pct(dislocation_summary['share_up_over_10pct'])} up, "
        f"{pct(dislocation_summary['share_down_over_10pct'])} down).",
        f"- **Large claims:** the largest {100 - claims['large_percentile']}% of claims "
        f"(over {money(claims['large_threshold'])}) make up {pct(claims['large_share_of_cost'])} of claims cost.",
        "",
        "## Model comparison (test set)",
        "",
    ]

    rows = {
        "Gini": {name: f"{ginis[name]:.4f}" for name in ginis},
        "Rebase factor (from validation)": {n: f"{test_pp[n]['rebase_factor']:.3f}" for n in test_pp},
        "Loss ratio - model premium": {n: f"{test_pp[n]['predicted']['loss_ratio']:.3f}" for n in test_pp},
        "Loss ratio - rebased": {n: f"{test_pp[n]['rebased']['loss_ratio']:.3f}" for n in test_pp},
        "Loss ratio - with minimum premium": {
            n: f"{test_pp[n]['annual_tariff']['loss_ratio']:.3f}" for n in test_pp
        },
        "Loss ratio - everyone pays monthly": {
            n: f"{test_pp[n]['all_monthly']['loss_ratio']:.3f}" for n in test_pp
        },
        "Minimum premium": {
            n: money(test_pp[n]["minimum_premium"]) if test_pp[n]["minimum_premium"] else "-" for n in test_pp
        },
        "Policies raised to minimum": {
            n: pct(test_pp[n].get("share_raised_to_minimum", 0)) for n in test_pp
        },
    }
    lines += [
        md_table(pd.DataFrame(rows).T.rename_axis("Metric")),
        "",
        "Loss ratio = actual claims cost ÷ pure premium earned. The rebase factor and minimum "
        "premium were set on the validation set, so the test loss ratios aren't 1 by construction.",
        "",
        "## Ranking risks",
        "",
        figure("lorenz.png", "Lorenz curves"),
        "",
        figure("lift.png", "Lift charts"),
        "",
        "### Double lift",
        "",
        figure("double_lift.png", "Double lift chart"),
        "",
        f"In the band where XGBoost is lowest relative to the GLM (ratio "
        f"{double_lift['RatioMin'].iloc[0]:.2f}-{double_lift['RatioMax'].iloc[0]:.2f}), actual cost was "
        f"{pct(double_lift['ActualIndex'].iloc[0], 0)} of average against GLM {pct(double_lift['AIndex'].iloc[0], 0)} "
        f"and XGBoost {pct(double_lift['BIndex'].iloc[0], 0)}. In the band where XGBoost is highest (ratio "
        f"{double_lift['RatioMin'].iloc[-1]:.2f}-{double_lift['RatioMax'].iloc[-1]:.2f}), actual cost was "
        f"{pct(double_lift['ActualIndex'].iloc[-1], 0)} against GLM {pct(double_lift['AIndex'].iloc[-1], 0)} "
        f"and XGBoost {pct(double_lift['BIndex'].iloc[-1], 0)}.",
        "",
        "## Component models (validation set)",
        "",
    ]

    freq_rows = []
    for name, out in outputs.items():
        for model, values in out["metrics"]["frequency"].items():
            if isinstance(values, dict):
                freq_rows.append({
                    "Model": name,
                    "Frequency model": FAMILY_NAMES[model],
                    "Poisson deviance": f"{values['poisson_deviance']:.5f}",
                    "Gini (claim counts)": f"{values['gini']:.3f}",
                })
    nb_alpha = glm_out["metrics"]["frequency"].get("nb_alpha")
    lines += [md_table(pd.DataFrame(freq_rows), index=False), ""]
    if nb_alpha is not None:
        lines += [f"Estimated negative binomial alpha: {nb_alpha:.3g} (0 would mean no overdispersion).", ""]

    pairs = pd.concat(
        [pd.read_csv(out["dir"] / "pair_comparison_val.csv").assign(Model=name) for name, out in outputs.items()]
    )
    pairs = pd.DataFrame({
        "Model": pairs["Model"],
        "Frequency": pairs["frequency"].map(FAMILY_NAMES),
        "Severity": pairs["severity"].map(FAMILY_NAMES),
        "Validation Gini": pairs["val_gini"].map("{:.4f}".format),
    })
    lines += [
        "Lower deviance is better. The frequency × severity pair with the highest validation Gini "
        "(pure premium) was chosen for each model:",
        "",
        md_table(pairs, index=False),
        "",
    ]
    if has_calibration:
        lines += [figure("hurdle_calibration.png", "Hurdle classifier calibration"), ""]

    lines += [
        "## Actual vs expected by rating factor (test set)",
        "",
        "A/E = actual cost ÷ predicted cost; 1.00 is a perfect fit. The table shows the band with the "
        "largest gap among bands holding at least 5% of the exposure. Factors marked **no** aren't "
        "model inputs, so a clear pattern there suggests a missing rating factor.",
        "",
        md_table(df_factors, index=False),
        "",
    ]
    for factor in factors_done:
        lines += [figure(f"ae_{factor}.png", f"Actual vs expected by {factor}"), ""]

    lines += [
        "## Price changes: GLM to XGBoost (test set)",
        "",
        figure("dislocation.png", "Price change distribution"),
        "",
        f"Median change {dislocation_summary['median_change']:+.1%}; "
        f"{pct(dislocation_summary['share_over_25pct'])} of policies would change by more than 25%.",
        "",
        md_table(
            dislocation.assign(
                Policies=dislocation["Policies"].map("{:,}".format), Share=dislocation["Share"].map(pct)
            ).rename_axis("Change")
        ),
        "",
    ]

    if has_price_curves:
        lines += [
            "## How the price responds",
            "",
            figure("price_curves.png", "Price response curves"),
            "",
        ]
        lines += [
            "Each example below breaks a customer's pure premium down against the typical customer "
            "(the most common or median value of each detail). The bars are Shapley values: each "
            "detail's average effect over every mix of the two customers' details, so they add up "
            "exactly to the difference in premium.",
            "",
        ]
        for example, by_model in explanations.items():
            customer, first_explanation = next(iter(by_model.values()))
            reference = first_explanation["reference"]
            changed = {k: v for k, v in customer.items() if v != reference[k]}
            lines += [
                f"### Example: {example.replace('_', ' ')}",
                "",
                "Details different from the typical customer: "
                + (", ".join(f"{k} = {v}" for k, v in changed.items()) or "none") + ".",
                "",
                "| | |",
                "|---|---|",
                "| " + " | ".join(
                    figure(f"explain_{example}_{key}.png", f"{name} explanation - {example}")
                    for key, name in MODELS.items()
                ) + " |",
                "",
            ]
            for name, (_, explanation) in by_model.items():
                effects = explanation["effects"]
                top = effects["PremiumImpact"].abs().sort_values(ascending=False).index[:2]
                drivers = " and ".join(
                    f"{detail} ({effects.loc[detail, 'PremiumImpact']:+,.0f})"
                    for detail in top if effects.loc[detail, "PremiumImpact"] != 0
                )
                lines.append(
                    f"- {name}: pure premium {money(explanation['pure_premium'])} against "
                    f"{money(explanation['base_premium'])} for the typical customer"
                    + (f"; largest effects {drivers}." if drivers else ".")
                )
            lines.append("")

    lines += [
        "## Claim sizes",
        "",
        figure("claim_sizes.png", "Claim size distribution"),
        "",
        f"{claims['n_claims']:,} claims: median {money(claims['median'])}, mean {money(claims['mean'])}, "
        f"largest {money(claims['max'])}. Claims above the {claims['large_percentile']}th percentile "
        f"({money(claims['large_threshold'])}) are {pct(claims['large_share_of_claims'])} of claims and "
        f"{pct(claims['large_share_of_cost'])} of cost.",
        "",
        "## Discussion",
        "",
        "<!-- Author's interpretation. Points worth covering:",
        "  - Which model would you put into production, and why (Gini gain vs interpretability)?",
        "  - What do the double lift and dislocation results mean for customers and for the business?",
        "  - Which rating factors show A/E patterns, and what would you change?",
        "  - How do large claims affect the severity models and the results? -->",
        "",
        "_To be written._",
        "",
    ]

    unused = [f"`{f}`" for f in factors_done if f not in INPUT_COLUMNS]
    raw_in_xgb = [c for c in INPUT_COLUMNS if c in cfg["xgb"]["features"]]
    lines += [
        "## Limitations",
        "",
        "- Prices are pure premiums (expected claims cost); expenses, commission and profit aren't included.",
        large_claim_note(cfg),
        "- Results come from a single train/validation/test split, without confidence intervals.",
        f"- Customer details used by the models: {', '.join(INPUT_COLUMNS)}."
        + (f" Not used: {', '.join(unused)}." if unused else ""),
    ]
    if not raw_in_xgb:
        lines.append(
            "- The XGBoost models use the engineered features only (e.g. the young driver flag), not raw "
            "values such as `DrivAge`, so their prices change in steps."
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main(config_from_command_line(__doc__))
