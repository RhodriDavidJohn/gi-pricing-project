"""The Streamlit app, run headless on pipeline outputs from synthetic data."""

import pytest

pytest.importorskip("streamlit")

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import run_pipeline  # noqa: E402
from lib.config import PROJECT_DIR  # noqa: E402

APP = str(PROJECT_DIR / "app" / "streamlit_app.py")
PAGES = ["views/model_selection.py", "views/comparison.py", "views/quote.py"]


@pytest.fixture
def app(cfg, tmp_path, monkeypatch):
    run_pipeline.run(cfg, steps=["prepare", "select_glm", "select_xgb", "train_final", "report"])
    monkeypatch.setenv("GI_PRICING_CONFIG", str(tmp_path / "config.yaml"))
    st.cache_data.clear()
    st.cache_resource.clear()
    return AppTest.from_file(APP, default_timeout=120).run()


def price(at):
    return float(at.metric[0].value.replace(",", ""))


def test_every_page_runs(app):
    assert not app.exception
    assert [m.label for m in app.metric][:2] == ["GLM Gini", "XGBoost Gini"]

    for page in PAGES:
        app.switch_page(page).run()
        assert not app.exception, page
        assert not app.error, page


def test_quote_updates_with_details_and_model(app):
    app.switch_page("views/quote.py").run()
    typical = price(app)
    assert app.info  # typical customer: nothing to explain yet

    app.slider(key="detail_DrivAge").set_value(20).run()
    young = price(app)
    assert young != typical
    assert not app.exception
    assert any("GLM calculation" in e.label for e in app.expander)

    # the GLM calculation multiplies out to the quoted price
    steps = [m.value for m in app.markdown if "= pure premium" in m.value]
    assert steps and f"{young:,.2f}" in steps[0]

    app.get("button_group")[0].set_value("xgb").run()
    assert not app.exception
    assert price(app) != young
    assert not any("GLM calculation" in e.label for e in app.expander)

    for detail in ["VehBrand", "Region", "Density"]:
        app.selectbox(key="what_if_detail").set_value(detail).run()
        assert not app.exception

    app.button[0].click().run()  # reset to the typical customer
    assert app.slider(key="detail_DrivAge").value != 20
    assert app.info


def test_missing_outputs_show_instructions(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("GI_PRICING_CONFIG", str(tmp_path / "config.yaml"))
    st.cache_data.clear()
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert at.error and "run_pipeline.py" in at.error[0].value
