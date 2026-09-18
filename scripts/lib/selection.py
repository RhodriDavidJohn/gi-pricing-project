"""Shared helpers for the model selection scripts."""

import itertools

import pandas as pd
import yaml

from lib.evaluation import gini_index

SELECTED_FILE = "selected.yaml"


def selection_dir(cfg, model_name):
    path = cfg["paths"]["output_dir"] / "selection" / model_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def choose_combination(freq_predictors, sev_predictors, df_val):
    """Pick the frequency x severity pair with the highest pure premium Gini on validation.

    freq_predictors / sev_predictors map a name to a function returning annual frequency /
    mean claim amount for a DataFrame, simplest model first: ties (to 4 decimal places) go to
    the pair listed first. Returns (frequency name, severity name, results table).
    """
    rows = []
    for (freq_name, predict_freq), (sev_name, predict_sev) in itertools.product(
        freq_predictors.items(), sev_predictors.items()
    ):
        rate = predict_freq(df_val) * predict_sev(df_val)
        rows.append({
            "frequency": freq_name,
            "severity": sev_name,
            "val_gini": gini_index(df_val["TotalClaimAmount"], rate, df_val["Exposure"]),
        })

    table = pd.DataFrame(rows)
    order = table["val_gini"].round(4).sort_values(ascending=False, kind="stable").index
    table = table.loc[order].reset_index(drop=True)
    return table.loc[0, "frequency"], table.loc[0, "severity"], table


def save_selection(spec, cfg, model_name):
    path = selection_dir(cfg, model_name) / SELECTED_FILE
    with open(path, "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False)
    return path


def load_selection(cfg, model_name):
    path = cfg["paths"]["output_dir"] / "selection" / model_name / SELECTED_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run select_{model_name}.py first")
    with open(path) as f:
        return yaml.safe_load(f)
