"""Loads the pipeline outputs the app shows. Everything read here is small and can be committed.

    models/{glm,xgb}.joblib, *_card.json, glm_relativities.csv      (train_final.py)
    outputs/selection/{glm,xgb}/*.csv, metrics.json, selected.yaml   (select_glm.py, select_xgb.py)
    reports/tables/*.csv                                             (report.py)

Set GI_PRICING_CONFIG to use a config other than the project's config.yaml.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from lib.config import load_config  # noqa: E402
from lib.pricing_model import load_pricing_model  # noqa: E402

MODELS = {"glm": "GLM", "xgb": "XGBoost"}
FAMILY_NAMES = {
    "poisson": "Poisson",
    "negative_binomial": "Negative binomial",
    "gamma": "Gamma",
    "log_normal": "Log-normal",
    "hurdle": "Hurdle",
    "simple": "Simple (Poisson)",
    "xgb": "XGBoost Gamma",
}

# customer details: friendly name and help text
DETAIL_LABELS = {
    "DrivAge": ("Driver age", "Age of the main driver, in years."),
    "BonusMalus": ("Bonus-malus", "French no-claims scale: 50 is the best record, new drivers start at "
                   "100, above 100 means past claims."),
    "VehAge": ("Vehicle age", "Age of the car, in years."),
    "VehPower": ("Vehicle power", "Power rating of the car."),
    "VehBrand": ("Vehicle brand", "Brand code (brands are anonymised in the data)."),
    "VehGas": ("Fuel", "Diesel or regular petrol."),
    "Density": ("Population density", "People per km² where the driver lives."),
    "Area": ("Area code", "A to F; an area code that follows population density."),
    "Region": ("Region", "Region of France (anonymised codes)."),
}


def detail_label(detail):
    return DETAIL_LABELS.get(detail, (detail, None))[0]


@st.cache_resource
def config():
    return load_config(os.environ.get("GI_PRICING_CONFIG", PROJECT_DIR / "config.yaml"))


def selection_dir(key):
    return config()["paths"]["output_dir"] / "selection" / key


def required_files():
    paths = config()["paths"]
    files = []
    for key in MODELS:
        files += [
            paths["model_dir"] / f"{key}.joblib",
            paths["model_dir"] / f"{key}_card.json",
            selection_dir(key) / "metrics.json",
            selection_dir(key) / "selected.yaml",
        ]
    files += [paths["report_dir"] / "tables" / "lorenz.csv"]
    return files


def check_outputs():
    """Stop the page with instructions if the pipeline hasn't been run."""
    missing = [f for f in required_files() if not f.exists()]
    if missing:
        st.error(
            "The pipeline outputs this app needs are missing. Run `python scripts/run_pipeline.py` "
            "from the project folder first.\n\nMissing: "
            + ", ".join(f"`{f.relative_to(PROJECT_DIR) if f.is_relative_to(PROJECT_DIR) else f}`" for f in missing)
        )
        st.stop()


@st.cache_resource
def final_model(key):
    return load_pricing_model(config()["paths"]["model_dir"] / f"{key}.joblib")


@st.cache_data
def model_card(key):
    return json.loads((config()["paths"]["model_dir"] / f"{key}_card.json").read_text())


@st.cache_data
def relativities():
    path = config()["paths"]["model_dir"] / "glm_relativities.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def selection_metrics(key):
    return json.loads((selection_dir(key) / "metrics.json").read_text())


@st.cache_data
def selected_spec(key):
    return yaml.safe_load((selection_dir(key) / "selected.yaml").read_text())


@st.cache_data
def selection_table(key, name):
    path = selection_dir(key) / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


def selection_file(key, name):
    path = selection_dir(key) / name
    return path if path.exists() else None


@st.cache_data
def report_table(name, index_col=None):
    path = config()["paths"]["report_dir"] / "tables" / f"{name}.csv"
    return pd.read_csv(path, index_col=index_col) if path.exists() else None


def factor_names():
    """Rating factors with an actual vs expected table, in config order."""
    return [f for f in config()["report"]["factor_bands"] if report_table(f"ae_{f}") is not None]


# ----------------------------------------------------
# Model calculations (cached - customers are passed as tuples of (detail, value) pairs)
# ----------------------------------------------------
def as_key(customer):
    return tuple(sorted(customer.items()))


@st.cache_data
def price_curve(key, customer_key, detail):
    model = final_model(key)
    return model.price_curve(dict(customer_key), detail)


@st.cache_data
def explain(key, customer_key):
    return final_model(key).explain(dict(customer_key))


@st.cache_data
def quote(key, customer_key):
    return final_model(key).quote(dict(customer_key)).iloc[0]


@st.cache_data
def price_spread(customer_key):
    """For each detail and model: highest ÷ lowest premium across the detail's values - 1."""
    rows = []
    for key, name in MODELS.items():
        model = final_model(key)
        for detail in model.input_fields:
            premium = price_curve(key, customer_key, detail)["PurePremium"]
            rows.append({"Detail": detail, "Series": name, "Spread": premium.max() / premium.min() - 1})
    return pd.DataFrame(rows)


def glm_factor_relativities(factor):
    """Relativities for one GLM rating factor (main effects only), with the base level as 1."""
    table = relativities()
    prefix = f"{factor} = "
    rows = table[table["Term"].str.startswith(prefix) & ~table["Term"].str.contains(" × ")].copy()
    if rows.empty:
        return None
    rows["Level"] = rows["Term"].str.extract(r"= (.*) \(vs")[0]
    base = rows["Term"].str.extract(r"\(vs (.*)\)$")[0].iloc[0]
    base_rows = pd.DataFrame({"Model": ["frequency", "severity"], "Level": base, "Relativity": 1.0})
    out = pd.concat([base_rows, rows[["Model", "Level", "Relativity"]]], ignore_index=True)

    # only models that use the factor
    used = set(rows["Model"])
    out = out[out["Model"].isin(used)]
    out["Model"] = out["Model"].str.capitalize()

    levels = final_model("glm").feature_info["levels"].get(factor)
    if levels:
        out["Level"] = pd.Categorical(out["Level"], categories=levels, ordered=True)
        out = out.sort_values(["Level", "Model"])
        out["Level"] = out["Level"].astype(str)
    return out
