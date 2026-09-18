import numpy as np
import pytest

from lib import gbm
from lib.evaluation import poisson_deviance


def params(cfg, model_key):
    return gbm.model_params(cfg["xgb"], model_key, {"max_depth": 3, "learning_rate": 0.1})


def test_simple_frequency_offset(splits, cfg):
    df_train, df_val, df_test = splits
    features = cfg["xgb"]["features"]
    model = gbm.fit_simple_frequency(df_train, df_val, features, params(cfg, "simple_count"))

    # annual frequency x exposure is the same as predicting with the log(Exposure) offset
    expected = model.predict_expected_claims(df_test)
    annual = model.predict_frequency(df_test)
    np.testing.assert_allclose(expected, annual * df_test["Exposure"], rtol=1e-4)

    # and it is on the right scale: expected claims roughly match actual claims
    assert expected.sum() == pytest.approx(df_test["ClaimNb"].sum(), rel=0.2)


def test_hurdle_frequency(splits, cfg):
    df_train, df_val, df_test = splits
    features = cfg["xgb"]["features"]

    hurdle = gbm.HurdleFrequency(
        gbm.fit_hurdle_classifier(df_train, df_val, features, params(cfg, "hurdle_classifier")),
        gbm.fit_hurdle_count(df_train, df_val, features, params(cfg, "hurdle_count")),
        features,
    )

    # a policy that has claimed has at least one claim
    assert (hurdle.predict_claims_if_claim(df_test) >= 1).all()

    # annual frequency doesn't depend on the exposure in the data
    freq = hurdle.predict_frequency(df_test)
    np.testing.assert_allclose(freq, hurdle.predict_frequency(df_test.assign(Exposure=0.1)))
    assert (freq > 0).all()

    # more exposure never lowers expected claims, so a part year is at most a full year
    assert (hurdle.predict_expected_claims(df_test) <= freq + 1e-6).all()


def test_severity_model_is_positive(splits, cfg):
    df_train, df_val, df_test = splits
    model = gbm.fit_severity(df_train, df_val, cfg["xgb"]["features"], params(cfg, "severity"))
    assert (model.predict_severity(df_test) > 0).all()


def test_grid_search_picks_lowest_score(splits, cfg):
    df_train, df_val, _ = splits
    features = cfg["xgb"]["features"]

    def fit_and_score(tuned):
        model = gbm.fit_simple_frequency(
            df_train, df_val, features, gbm.model_params(cfg["xgb"], "simple_count", tuned)
        )
        score = poisson_deviance(df_val["ClaimNb"], model.predict_expected_claims(df_val))
        return model, score, gbm.n_trees(model.model)

    best, _, table = gbm.grid_search(fit_and_score, {"max_depth": [1, 3], "learning_rate": [0.1]})

    assert len(table) == 2
    assert best["max_depth"] == table.iloc[0]["max_depth"]
    assert table["score"].is_monotonic_increasing
