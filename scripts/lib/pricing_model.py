"""Trained pricing models that quote customers - the interface for the (future) Streamlit app.

    model = load_pricing_model("models/glm.joblib")
    model.input_fields                                   # what to ask the customer
    model.quote({"DrivAge": 30, "VehAge": 2, ...})       # one row per customer
    model.price_curve(customer, "DrivAge")               # how the price moves with one detail
    model.explain(customer)                              # effect of each detail on the price

Prices are annual pure premiums (expected claims cost) after rebasing, with the minimum
premium applied. They don't include expenses, commission or profit.
"""

import re
from math import factorial

import joblib
import numpy as np
import pandas as pd

from lib import glm
from lib.data import CATEGORICAL_INPUTS, INPUT_COLUMNS, clean_frequency_data, clean_text
from lib.pricing import minimum_premium_from_percentile, rebase_factor


def describe_inputs(df_frq_raw):
    """Allowed values and defaults for each customer detail, taken from the training data."""
    fields = {}
    for col in INPUT_COLUMNS:
        values = df_frq_raw[col]
        if col in CATEGORICAL_INPUTS:
            values = clean_text(values)
            fields[col] = {
                "type": "category",
                "options": sorted(values.unique().tolist()),
                "default": values.mode()[0],
            }
        else:
            is_integer = bool((values % 1 == 0).all())
            cast = int if is_integer else float
            fields[col] = {
                "type": "integer" if is_integer else "number",
                "min": cast(values.min()),
                "max": cast(values.max()),
                "default": cast(values.median()),
            }
    return fields


class PricingModel:
    """Shared quoting logic. Subclasses provide predict_frequency and predict_severity."""

    def __init__(self, name, feature_cfg, feature_info, input_fields, monthly_loading):
        self.name = name
        self.feature_cfg = feature_cfg
        self.feature_info = feature_info  # category levels, so every quote is encoded like the training data
        self.input_fields = input_fields
        self.monthly_loading = monthly_loading
        self.rebase_factor = 1.0
        self.minimum_premium = None

    # -------- to be provided by subclasses --------
    def predict_frequency(self, df):
        raise NotImplementedError

    def predict_severity(self, df):
        raise NotImplementedError

    # -------- shared --------
    def default_customer(self):
        return {col: field["default"] for col, field in self.input_fields.items()}

    def prepare(self, customers):
        """Turn customer details (a dict or DataFrame) into model features for a one-year policy."""
        df = pd.DataFrame([customers]) if isinstance(customers, dict) else customers.copy()

        missing = set(INPUT_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing customer details: {sorted(missing)}")

        df = clean_frequency_data(df[INPUT_COLUMNS], self.feature_cfg, self.feature_info)
        df["Exposure"] = 1.0
        return df.reset_index(drop=True)

    def pure_premium(self, df, rebased=True):
        """Annual pure premium for prepared (feature-engineered) data."""
        rate = self.predict_frequency(df) * self.predict_severity(df)
        return rate * self.rebase_factor if rebased else rate

    def calibrate(self, df_holdout, minimum_premium_percentile=None):
        """Set the rebase factor and minimum premium from data the model wasn't fitted on."""
        rate = self.pure_premium(df_holdout, rebased=False)
        self.rebase_factor = rebase_factor(df_holdout["TotalClaimAmount"], rate, df_holdout["Exposure"])

        if minimum_premium_percentile is not None:
            self.minimum_premium = minimum_premium_from_percentile(
                rate * self.rebase_factor, minimum_premium_percentile
            )
        return self

    def quote(self, customers):
        df = self.prepare(customers)

        frequency = self.predict_frequency(df)
        severity = self.predict_severity(df)
        pure_premium = frequency * severity * self.rebase_factor

        annual = pure_premium
        if self.minimum_premium is not None:
            annual = np.maximum(pure_premium, self.minimum_premium)

        return pd.DataFrame({
            "AnnualFrequency": frequency,
            "AverageClaimCost": severity,
            "PurePremium": pure_premium,
            "AnnualPremium": annual,
            "MonthlyPremium": annual / 12 * (1 + self.monthly_loading),
            "MinimumPremiumApplied": annual > pure_premium,
        })

    def curve_values(self, field, n_points=60):
        """Values to try for one detail: every option, or a grid across its range
        (log-spaced when the range covers several orders of magnitude, e.g. Density)."""
        spec = self.input_fields[field]
        if spec["type"] == "category":
            return spec["options"]

        low, high = spec["min"], spec["max"]
        if low > 0 and high / low > 100:
            values = np.geomspace(low, high, n_points)
        else:
            values = np.linspace(low, high, n_points)

        return np.unique(values.round().astype(int)) if spec["type"] == "integer" else values

    def price_curve(self, customer, field, values=None):
        """Quote the same customer for each value of one detail, e.g. DrivAge from 18 to 90.
        Uses curve_values(field) when no values are given."""
        if values is None:
            values = self.curve_values(field)
        customers = pd.DataFrame([{**customer, field: value} for value in values])
        curve = self.quote(customers)
        curve.insert(0, field, list(values))
        return curve

    def explain(self, customer, reference=None):
        """Break a customer's pure premium down by customer detail, relative to a reference
        customer (default: the typical customer from default_customer()).

        Each detail's effect is its exact Shapley value for log(pure premium): the model is quoted
        for every mix of this customer's and the reference customer's details, and each detail
        gets its average contribution over all those mixes. This works for any model, and the
        effects add up exactly: reference premium × product of the factors = this premium.
        Details that match the reference have no effect.

        Returns a dict with:
        - base_premium: the reference customer's pure premium
        - pure_premium: this customer's pure premium
        - reference: the reference customer
        - effects: one row per detail - its value and the reference value, the log-scale effect on
          frequency and on severity, the total log effect, the multiplicative factor, and a
          currency amount (the premium difference shared out in proportion to the log effects)
        """
        reference = self.default_customer() if reference is None else reference
        customer = {**reference, **customer}
        players = [col for col in INPUT_COLUMNS if customer[col] != reference[col]]
        n = len(players)

        # every mix of customer and reference values (bit i set = player i takes the customer's value)
        masks = np.arange(2 ** n)
        mixes = [
            {**reference, **{p: customer[p] for i, p in enumerate(players) if mask >> i & 1}}
            for mask in masks
        ]
        quotes = self.quote(pd.DataFrame(mixes))
        log_frequency = np.log(quotes["AnnualFrequency"].to_numpy())
        log_severity = np.log(quotes["AverageClaimCost"].to_numpy())

        sizes = np.array([bin(mask).count("1") for mask in masks])
        weights = np.array([factorial(k) * factorial(n - k - 1) / factorial(n) for k in range(n)])

        effects = pd.DataFrame(
            0.0,
            index=pd.Index(INPUT_COLUMNS, name="Detail"),
            columns=["FrequencyEffect", "SeverityEffect"],
        )
        for i, player in enumerate(players):
            without = masks[(masks >> i & 1) == 0]
            with_player = without | (1 << i)
            w = weights[sizes[without]]
            effects.loc[player, "FrequencyEffect"] = np.sum(w * (log_frequency[with_player] - log_frequency[without]))
            effects.loc[player, "SeverityEffect"] = np.sum(w * (log_severity[with_player] - log_severity[without]))

        effects.insert(0, "Value", [customer[col] for col in INPUT_COLUMNS])
        effects.insert(1, "Reference", [reference[col] for col in INPUT_COLUMNS])
        effects["LogEffect"] = effects["FrequencyEffect"] + effects["SeverityEffect"]
        effects["Factor"] = np.exp(effects["LogEffect"])

        base_premium = float(quotes["PurePremium"].iloc[0])
        pure_premium = float(quotes["PurePremium"].iloc[-1])

        total_log = effects["LogEffect"].sum()
        share = effects["LogEffect"] / total_log if total_log != 0 else 0.0
        effects["PremiumImpact"] = (pure_premium - base_premium) * share

        return {
            "base_premium": base_premium,
            "pure_premium": pure_premium,
            "reference": reference,
            "effects": effects,
        }

    def save(self, path):
        joblib.dump(self, path)


class GLMPricingModel(PricingModel):
    """GLM frequency x severity. Stores coefficients and model terms only, so the saved file is small."""

    def __init__(self, freq_params, freq_terms, sev_params, sev_terms, sev_family, smearing_factor, **kwargs):
        super().__init__(**kwargs)
        self.freq_params = freq_params
        self.freq_terms = freq_terms
        self.sev_params = sev_params
        self.sev_terms = sev_terms
        self.sev_family = sev_family
        self.smearing_factor = smearing_factor

    def predict_frequency(self, df):
        return np.exp(glm.linear_predictor(self.freq_params, self.freq_terms, df))

    def predict_severity(self, df):
        # both the Gamma GLM and the log-normal model are linear on the log scale
        return np.exp(glm.linear_predictor(self.sev_params, self.sev_terms, df)) * self.smearing_factor

    def calculation(self, customer):
        """The GLM calculation for one customer, term by term.

        Returns a dict with:
        - terms: every model column that is non-zero for this customer - its value, coefficient
          and factor exp(coefficient × value). Frequency = product of the frequency factors;
          average claim cost = product of the severity factors × smearing factor.
        - smearing_factor, rebase_factor
        - frequency, severity, pure_premium: the results of the calculation
        """
        df = self.prepare(customer)
        rows = []
        results = {}
        for part, params, terms in [
            ("Frequency", self.freq_params, self.freq_terms),
            ("Severity", self.sev_params, self.sev_terms),
        ]:
            x = glm.design_matrix(terms, df)[params.index].iloc[0]
            for column, value in x.items():
                if value != 0:
                    rows.append({
                        "Model": part,
                        "Term": readable_term(column),
                        "Value": float(value),
                        "Coefficient": float(params[column]),
                        "Factor": float(np.exp(params[column] * value)),
                    })
            results[part] = float(np.exp(x @ params))

        severity = results["Severity"] * self.smearing_factor
        return {
            "terms": pd.DataFrame(rows),
            "smearing_factor": self.smearing_factor,
            "rebase_factor": self.rebase_factor,
            "frequency": results["Frequency"],
            "severity": severity,
            "pure_premium": results["Frequency"] * severity * self.rebase_factor,
        }

    def relativities(self):
        """Multiplicative factor for each model coefficient (frequency and severity), with
        readable terms, e.g. "DrivAgeBand = 18-20 (vs 41-50)". For numeric factors the
        relativity is per unit (e.g. per bonus-malus point)."""
        tables = []
        for part, params in [("frequency", self.freq_params), ("severity", self.sev_params)]:
            tables.append(pd.DataFrame({
                "Model": part,
                "Term": [readable_term(term) for term in params.index],
                "Coefficient": params.values,
                "Relativity": np.exp(params.values),
            }))
        return pd.concat(tables, ignore_index=True)


CATEGORY_TERM = re.compile(r"C\((\w+), Treatment\(reference='([^']*)'\)\)\[T\.([^\]]*)\]")


def readable_term(term):
    """patsy column name -> readable label, e.g.
    "C(Region, Treatment(reference='R24'))[T.R11]" -> "Region = R11 (vs R24)"."""
    return CATEGORY_TERM.sub(r"\1 = \3 (vs \2)", term).replace(":", " × ")


class XGBPricingModel(PricingModel):
    """XGBoost frequency (simple or hurdle) x XGBoost severity."""

    def __init__(self, frequency_model, severity_model, frequency_type, **kwargs):
        super().__init__(**kwargs)
        self.frequency_model = frequency_model
        self.severity_model = severity_model
        self.frequency_type = frequency_type

    def predict_frequency(self, df):
        return self.frequency_model.predict_frequency(df)

    def predict_severity(self, df):
        return self.severity_model.predict_severity(df)


def load_pricing_model(path):
    return joblib.load(path)
