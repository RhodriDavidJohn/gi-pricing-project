import numpy as np
import pandas as pd
import pytest

import train_final
from lib.data import feature_info
from lib.pricing_model import describe_inputs, load_pricing_model

GLM_SPEC = {
    "frequency": {
        "family": "poisson",
        "terms": ["DrivAgeBand", "BonusMalusCapped", "LogDensity", "Region", "YoungDrivAge:VehPower"],
    },
    "severity": {
        "family": "log_normal",
        "terms": ["BonusMalusCapped", "VehAgeBand", "VehBrand", "VehAgeBand:VehBrand"],
    },
}
XGB_SPEC = {
    "frequency_type": "hurdle",
    "tuned_params": {
        name: {"max_depth": 2, "learning_rate": 0.1}
        for name in ["hurdle_classifier", "hurdle_count", "simple_count", "severity"]
    },
}


@pytest.fixture
def common(cfg, raw_data):
    return {
        "feature_cfg": cfg["features"],
        "feature_info": feature_info(*raw_data, cfg["features"]),
        "input_fields": describe_inputs(raw_data[0]),
        "monthly_loading": cfg["pricing"]["monthly_loading"],
    }


@pytest.fixture
def glm_model(splits, cfg, common):
    df_train, df_val, _ = splits
    model, _ = train_final.train_glm(df_train, df_val, GLM_SPEC, cfg, common)
    return model.calibrate(df_val, minimum_premium_percentile=5)


@pytest.fixture
def xgb_model(splits, cfg, common):
    df_train, df_val, _ = splits
    model, _ = train_final.train_xgb(df_train, df_val, XGB_SPEC, cfg, common)
    return model.calibrate(df_val, minimum_premium_percentile=5)


@pytest.fixture
def customer():
    return {
        "DrivAge": 22, "BonusMalus": 120, "VehAge": 3, "VehPower": 9, "VehBrand": "B12",
        "VehGas": "Diesel", "Density": 2500, "Area": "D", "Region": "R82",
    }


def test_describe_inputs(raw_data):
    fields = describe_inputs(raw_data[0])
    assert fields["VehGas"]["options"] == ["Diesel", "Regular"]  # quotes removed
    assert fields["DrivAge"]["type"] == "integer"
    assert fields["DrivAge"]["min"] <= fields["DrivAge"]["default"] <= fields["DrivAge"]["max"]


def test_quote_columns_and_minimum_premium(glm_model, customer):
    quote = glm_model.quote(customer).iloc[0]

    assert quote["PurePremium"] == pytest.approx(
        quote["AnnualFrequency"] * quote["AverageClaimCost"] * glm_model.rebase_factor
    )
    assert quote["AnnualPremium"] == max(quote["PurePremium"], glm_model.minimum_premium)
    assert quote["MonthlyPremium"] == pytest.approx(quote["AnnualPremium"] / 12 * 1.06)

    # a very low-risk customer is raised to the minimum premium
    low_risk = {**customer, "DrivAge": 50, "BonusMalus": 50, "Density": 1}
    glm_model.minimum_premium = 1e6
    low_quote = glm_model.quote(low_risk).iloc[0]
    assert low_quote["AnnualPremium"] == 1e6 and low_quote["MinimumPremiumApplied"]


def test_quote_needs_all_details(glm_model, customer):
    del customer["DrivAge"]
    with pytest.raises(ValueError, match="DrivAge"):
        glm_model.quote(customer)


def test_rebase_gives_loss_ratio_of_one_on_calibration_data(glm_model, splits):
    _, df_val, _ = splits
    premium = glm_model.pure_premium(df_val) * df_val["Exposure"]
    assert premium.sum() == pytest.approx(df_val["TotalClaimAmount"].sum())


@pytest.mark.parametrize("model_name", ["glm_model", "xgb_model"])
def test_explain_adds_up_to_the_premium(model_name, customer, request):
    model = request.getfixturevalue(model_name)
    explanation = model.explain(customer)
    effects = explanation["effects"]

    assert list(effects.index) == list(customer)
    assert explanation["base_premium"] == pytest.approx(model.quote(model.default_customer()).iloc[0]["PurePremium"])
    assert explanation["pure_premium"] == pytest.approx(model.quote(customer).iloc[0]["PurePremium"])
    assert explanation["base_premium"] * effects["Factor"].prod() == pytest.approx(explanation["pure_premium"])
    assert effects["PremiumImpact"].sum() == pytest.approx(
        explanation["pure_premium"] - explanation["base_premium"]
    )
    np.testing.assert_allclose(effects["LogEffect"], effects["FrequencyEffect"] + effects["SeverityEffect"])

    # details that match the reference customer have no effect
    same = [col for col in customer if customer[col] == explanation["reference"][col]]
    assert (effects.loc[same, "LogEffect"] == 0).all()


def test_explain_is_exact_shapley(glm_model):
    # two details differ from the reference: DrivAge and VehPower, which interact in the
    # frequency model (YoungDrivAge:VehPower). Shapley value of DrivAge = average of its effect
    # with VehPower at the reference value and at the customer's value.
    reference = glm_model.default_customer()
    customer = {**reference, "DrivAge": 20, "VehPower": reference["VehPower"] + 3}

    def log_freq(**details):
        return np.log(glm_model.quote({**reference, **details}).iloc[0]["AnnualFrequency"])

    expected = 0.5 * (
        (log_freq(DrivAge=20) - log_freq())
        + (log_freq(DrivAge=20, VehPower=customer["VehPower"]) - log_freq(VehPower=customer["VehPower"]))
    )
    effects = glm_model.explain(customer)["effects"]
    assert effects.loc["DrivAge", "FrequencyEffect"] == pytest.approx(expected)


def test_explain_main_effect_is_the_relativity(glm_model):
    # BonusMalus only enters the frequency model as a main effect, so its effect is the
    # coefficient times the change
    reference = glm_model.default_customer()
    customer = {**reference, "BonusMalus": reference["BonusMalus"] + 20}
    effects = glm_model.explain(customer)["effects"]
    assert effects.loc["BonusMalus", "FrequencyEffect"] == pytest.approx(
        glm_model.freq_params["BonusMalusCapped"] * 20
    )


def test_relativities(glm_model):
    table = glm_model.relativities()
    assert set(table["Model"]) == {"frequency", "severity"}
    assert (table["Relativity"] > 0).all()
    assert any("Region" in term for term in table["Term"])


def test_price_curve(glm_model, customer):
    curve = glm_model.price_curve(customer, "DrivAge", [20, 30, 40])
    assert curve["DrivAge"].tolist() == [20, 30, 40]
    # young driver flag switches off after 25
    assert curve.loc[0, "PurePremium"] != curve.loc[1, "PurePremium"]

    density_values = glm_model.curve_values("Density")
    assert np.all(np.diff(density_values) > 0)
    assert len(glm_model.price_curve(customer, "VehGas")) == 2


def test_xgb_model_quotes_and_saves(splits, cfg, common, customer, glm_model, tmp_path):
    df_train, df_val, _ = splits
    for frequency_type in ["hurdle", "simple"]:
        spec = {**XGB_SPEC, "frequency_type": frequency_type}
        model, details = train_final.train_xgb(df_train, df_val, spec, cfg, common)
        model.calibrate(df_val, 5)

        model.save(tmp_path / "xgb.joblib")
        loaded = load_pricing_model(tmp_path / "xgb.joblib")
        pd.testing.assert_frame_equal(loaded.quote(customer), model.quote(customer))
        assert loaded.quote(customer).iloc[0]["AnnualPremium"] > 0

    glm_model.save(tmp_path / "glm.joblib")
    assert (tmp_path / "glm.joblib").stat().st_size < 50_000


def test_readable_terms():
    from lib.pricing_model import readable_term

    assert readable_term("C(Region, Treatment(reference='R24'))[T.R11]") == "Region = R11 (vs R24)"
    assert (
        readable_term("C(DrivAgeBand, Treatment(reference='41-50'))[T.18-20]:BonusMalusCapped")
        == "DrivAgeBand = 18-20 (vs 41-50) × BonusMalusCapped"
    )
    assert readable_term("LogDensity") == "LogDensity"


def test_glm_calculation_multiplies_out_to_the_quote(glm_model, customer):
    calc = glm_model.calculation(customer)
    terms = calc["terms"]
    quote = glm_model.quote(customer).iloc[0]

    freq = terms.loc[terms["Model"] == "Frequency", "Factor"].prod()
    sev = terms.loc[terms["Model"] == "Severity", "Factor"].prod() * calc["smearing_factor"]
    assert freq == pytest.approx(quote["AnnualFrequency"])
    assert sev == pytest.approx(quote["AverageClaimCost"])
    assert calc["pure_premium"] == pytest.approx(quote["PurePremium"])
    # base levels have no term; other levels show against the base
    assert "DrivAgeBand = 21-25 (vs " in " ".join(terms["Term"])
