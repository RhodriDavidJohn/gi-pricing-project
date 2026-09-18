import numpy as np
import pandas as pd
import pytest

from lib import glm

TERMS = ["BonusMalus", "YoungDrivAge", "LogDensity"]


def test_frequency_prediction_is_annual_rate(policy_data):
    # expected claims (annual rate x exposure) should add up to actual claims on the training data
    results = glm.fit_frequency_glm(policy_data, TERMS)
    annual = glm.predict_frequency(results, policy_data)

    expected_claims = annual * policy_data["Exposure"]
    assert expected_claims.sum() == pytest.approx(policy_data["ClaimNb"].sum(), rel=1e-6)


def test_severity_prediction_covers_every_policy(policy_data):
    # regression test: dmatrices with a claim amount response dropped non-claiming policies
    results = glm.fit_gamma_glm(policy_data, TERMS)
    pred = glm.predict_gamma_severity(results, policy_data)

    assert len(pred) == len(policy_data)
    assert not np.isnan(pred).any()


@pytest.mark.parametrize("fit_fn", [glm.fit_gamma_glm, glm.fit_log_normal_wls])
def test_severity_models_use_claim_amount_response(policy_data, fit_fn):
    results = fit_fn(policy_data, TERMS)
    assert results.model.endog_names in ("MeanClaimAmountCapped", "LogClaimAmountCapped")


def test_log_normal_prediction_is_on_claim_amount_scale(policy_data):
    results = glm.fit_log_normal_wls(policy_data, TERMS)
    pred = glm.predict_log_normal_severity(results, policy_data)
    mean_claim = policy_data["MeanClaimAmount"].mean()
    assert 0.5 < pred.mean() / mean_claim < 2


def test_nb_alpha_estimate():
    # counts with known overdispersion: Var = mu + alpha * mu^2 with alpha = 0.8
    rng = np.random.default_rng(1)
    n = 50_000
    df = pd.DataFrame({"x": rng.normal(size=n), "Exposure": rng.uniform(0.1, 1, n)})
    mu = np.exp(-1 + 0.5 * df["x"]) * df["Exposure"]
    df["ClaimNb"] = rng.poisson(mu * rng.gamma(1 / 0.8, 0.8, n))

    assert glm.estimate_nb_alpha(df, ["x"]) == pytest.approx(0.8, abs=0.1)


def test_select_interactions_adds_a_real_interaction_and_skips_noise():
    rng = np.random.default_rng(2)
    n = 50_000
    df = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "noise": rng.normal(size=n),
        "Exposure": 1.0,
    })
    df["ClaimNb"] = rng.poisson(np.exp(-1 + 0.3 * df["a"] + 0.3 * df["b"] + 0.4 * df["a"] * df["b"]))

    for criterion in ["aic", "bic"]:
        terms, history = glm.select_interactions(
            glm.fit_frequency_glm, df, ["a", "b", "noise"], ["a:noise", "a:b", "b:noise"], criterion=criterion
        )
        assert terms == ["a", "b", "noise", "a:b"]
        assert history["added"].tolist() == [glm.BASE_MODEL, "a:b"]
        assert criterion in history.columns


def test_select_interactions_can_keep_the_base_model(policy_data):
    # an improvement threshold no interaction can reach
    terms, history = glm.select_interactions(
        glm.fit_gamma_glm, policy_data, TERMS, ["BonusMalus:YoungDrivAge"], min_improvement=1e9
    )
    assert terms == TERMS
    assert len(history) == 1


def test_categorical_terms_use_largest_exposure_base(policy_data):
    terms = glm.expand_terms(["DrivAgeBand", "BonusMalusCapped", "DrivAgeBand:BonusMalusCapped"], policy_data)
    base = policy_data.groupby("DrivAgeBand", observed=True)["Exposure"].sum().idxmax()

    assert terms[0] == f"C(DrivAgeBand, Treatment(reference='{base}'))"
    assert terms[1] == "BonusMalusCapped"
    assert terms[2] == f"{terms[0]}:BonusMalusCapped"


def test_categorical_glm_predicts_single_rows(policy_data):
    # each quote is one row - the design must still have a column for every level
    covariates = ["DrivAgeBand", "VehAgeBand", "VehPowerBand", "BonusMalusCapped", "Region", "VehBrand"]
    results = glm.fit_frequency_glm(policy_data, covariates)

    one_at_a_time = [glm.predict_frequency(results, policy_data.iloc[[i]])[0] for i in range(20)]
    np.testing.assert_allclose(one_at_a_time, glm.predict_frequency(results, policy_data.iloc[:20]))

    # and the offset still gives the right total claims
    expected = glm.predict_frequency(results, policy_data) * policy_data["Exposure"]
    assert expected.sum() == pytest.approx(policy_data["ClaimNb"].sum(), rel=1e-6)
