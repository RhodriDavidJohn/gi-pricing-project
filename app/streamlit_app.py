"""GI pricing app: `streamlit run app/streamlit_app.py` from the project folder."""

import streamlit as st

st.set_page_config(page_title="GI Pricing", page_icon="🚗", layout="wide")

pages = st.navigation([
    st.Page("views/home.py", title="Overview", icon="🏠", default=True),
    st.Page("views/model_selection.py", title="Model selection", icon="🧪"),
    st.Page("views/comparison.py", title="GLM vs XGBoost", icon="⚖️"),
    st.Page("views/quote.py", title="Get a quote", icon="💷"),
])
pages.run()
