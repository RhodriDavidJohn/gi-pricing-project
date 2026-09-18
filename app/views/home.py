import streamlit as st

import app_data as data

data.check_outputs()

st.title("Motor insurance pricing: GLM vs XGBoost")
st.markdown(
    "Frequency-severity **pure premium** models (expected claims cost per policy year) for the "
    "French motor third-party liability data ([freMTPL2](https://www.openml.org/d/41214)). "
    "A GLM and an XGBoost model are each selected on a validation set, compared on a held-out "
    "test set, then retrained for quoting."
)

glm_test = data.selection_metrics("glm")["test_pure_premium"]
xgb_test = data.selection_metrics("xgb")["test_pure_premium"]
glm_spec, xgb_spec = data.selected_spec("glm"), data.selected_spec("xgb")
glm_card = data.model_card("glm")

st.subheader("Headline results (test set)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("GLM Gini", f"{glm_test['rebased']['gini']:.3f}",
            help="How well the model ranks low and high risks (higher is better)")
col2.metric("XGBoost Gini", f"{xgb_test['rebased']['gini']:.3f}",
            delta=f"{xgb_test['rebased']['gini'] - glm_test['rebased']['gini']:+.3f} vs GLM")
col3.metric("GLM loss ratio", f"{glm_test['rebased']['loss_ratio']:.3f}",
            help="Actual claims cost ÷ pure premium, after rebasing on the validation set (1 = on target)")
col4.metric("XGBoost loss ratio", f"{xgb_test['rebased']['loss_ratio']:.3f}")

st.subheader("Selected models")
col1, col2 = st.columns(2)
with col1, st.container(border=True):
    st.markdown("**GLM**")
    st.markdown(
        f"- Frequency: {data.FAMILY_NAMES[glm_spec['frequency']['family']]}\n"
        f"- Severity: {data.FAMILY_NAMES[glm_spec['severity']['family']]}\n"
        f"- Rating factors: {', '.join(glm_spec['frequency']['terms'])}"
    )
with col2, st.container(border=True):
    st.markdown("**XGBoost**")
    st.markdown(
        f"- Frequency: {data.FAMILY_NAMES[xgb_spec['frequency_type']]}\n"
        f"- Severity: XGBoost (`{data.config()['xgb']['severity']['objective']}`)\n"
        f"- Features: {', '.join(data.config()['xgb']['features'])}"
    )

st.subheader("What's in the app")
col1, col2, col3 = st.columns(3)
with col1, st.container(border=True):
    st.page_link("views/model_selection.py", label="**Model selection**", icon="🧪")
    st.caption("How each model's structure and settings were chosen on the validation set.")
with col2, st.container(border=True):
    st.page_link("views/comparison.py", label="**GLM vs XGBoost**", icon="⚖️")
    st.caption("Ranking, fit by rating factor, price changes, and how each model can be explained.")
with col3, st.container(border=True):
    st.page_link("views/quote.py", label="**Get a quote**", icon="💷")
    st.caption("Enter your details, get a pure premium and see exactly how it was built.")

st.caption(
    f"Final models trained on {glm_card['n_fit']:,} policies ({glm_card['n_holdout']:,} held back for "
    f"rebasing and calibration) on {glm_card['trained_at'][:10]}. Prices are pure premiums: expected "
    "claims cost only, without expenses, commission or profit."
)
