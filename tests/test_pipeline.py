import json
import subprocess
import sys

import pandas as pd
import pytest
import yaml

import run_pipeline
from lib.config import PROJECT_DIR


def test_full_pipeline(cfg, tmp_path):
    results = run_pipeline.run(cfg)
    output_dir, model_dir = cfg["paths"]["output_dir"], cfg["paths"]["model_dir"]

    # selection reports and chosen specs
    for name in ["glm", "xgb"]:
        sel_dir = output_dir / "selection" / name
        spec = yaml.safe_load((sel_dir / "selected.yaml").read_text())
        assert spec == results[f"select_{name}"]["selected"]

        pairs = pd.read_csv(sel_dir / "pair_comparison_val.csv")
        assert pairs["val_gini"].round(4).is_monotonic_decreasing

        # test set results: rebased on validation, so the loss ratio is near (not exactly) 1
        test_results = results[f"select_{name}"]["test_pure_premium"]
        assert 0.5 < test_results["rebased"]["loss_ratio"] < 2

    xgb_spec = results["select_xgb"]["selected"]
    assert xgb_spec["frequency_type"] in ("hurdle", "simple")
    assert set(xgb_spec["tuned_params"]) == {"simple_count", "hurdle_classifier", "hurdle_count", "severity"}
    assert (output_dir / "selection" / "xgb" / "hurdle_calibration_test.png").exists()
    assert (model_dir / "glm_relativities.csv").exists()

    # final models and cards
    for name in ["glm", "xgb"]:
        card = json.loads((model_dir / f"{name}_card.json").read_text())
        assert card["large_claim_cap"] > 0
        assert card["n_fit"] == 5400 and card["n_holdout"] == 600
        # the rebase factor was set on the holdout (XGBoost predicts in float32, hence rel=1e-6)
        assert card["holdout"]["loss_ratio"] == pytest.approx(1, rel=1e-6)
        assert card["example_quote"]["AnnualPremium"] >= card["minimum_premium"]

    # report: figures, tables and a write-up with no unfilled values
    report_dir = cfg["paths"]["report_dir"]
    results = (report_dir / "results.md").read_text()
    for figure in ["lorenz", "lift", "double_lift", "dislocation", "price_curves", "claim_sizes",
                   "ae_DrivAge", "ae_Region", "explain_young_driver_glm", "explain_young_driver_xgb"]:
        assert (report_dir / "figures" / f"{figure}.png").exists()
        assert f"figures/{figure}.png" in results
    assert "nan" not in results.lower()
    assert "capped at the 99.5th percentile" in results

    # quote.py from the terminal, using the trained models
    config_path = tmp_path / "config.yaml"
    for args in (
        ["--model", "glm", "--DrivAge", "22", "--change", "DrivAge=40", "--explain", "--curve", "Density"],
        ["--model", "xgb", "--VehGas", "Diesel", "--Region", "R24", "--curve", "VehGas", "--explain"],
    ):
        result = subprocess.run(
            [sys.executable, str(PROJECT_DIR / "scripts" / "quote.py"), "--config", str(config_path), *args],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "AnnualPremium" in result.stdout


def test_choose_combination_prefers_first_listed_on_ties(policy_data):
    from lib.selection import choose_combination

    def frequency(df):
        return df["BonusMalus"].to_numpy() / 100

    def severity(df):
        return 1000.0

    freq_name, _, table = choose_combination(
        {"poisson": frequency, "negative_binomial": lambda df: frequency(df) * (1 + 1e-9)},
        {"gamma": severity},
        policy_data,
    )
    assert freq_name == "poisson"
    assert len(table) == 2
