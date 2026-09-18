import pandas as pd
import streamlit as st

import app_charts as charts
import app_data as data

data.check_outputs()
cfg = data.config()

st.title("GLM vs XGBoost")
st.markdown(
    "Both models were chosen on the validation set. Everything on the **Performance** tab uses the "
    "test set: the same unseen policies for both models. The **Interpretability** tab "
    "uses the final models."
)

tests = {name: data.selection_metrics(key)["test_pure_premium"] for key, name in data.MODELS.items()}
ginis = {name: t["rebased"]["gini"] for name, t in tests.items()}

performance_tab, interpretability_tab = st.tabs(["Performance", "Interpretability"])

# ----------------------------------------------------
# Performance
# ----------------------------------------------------
with performance_tab:
    rows = {
        "Gini (ranking - higher is better)": {n: f"{t['rebased']['gini']:.4f}" for n, t in tests.items()},
        "Loss ratio - model premium": {n: f"{t['predicted']['loss_ratio']:.3f}" for n, t in tests.items()},
        "Loss ratio - rebased on validation": {n: f"{t['rebased']['loss_ratio']:.3f}" for n, t in tests.items()},
        "Loss ratio - with minimum premium": {
            n: f"{t['annual_tariff']['loss_ratio']:.3f}" for n, t in tests.items()
        },
        "Loss ratio - everyone pays monthly": {
            n: f"{t['all_monthly']['loss_ratio']:.3f}" for n, t in tests.items()
        },
        "Rebase factor": {n: f"{t['rebase_factor']:.3f}" for n, t in tests.items()},
        "Minimum premium": {
            n: f"{t['minimum_premium']:,.0f}" if t["minimum_premium"] else "-" for n, t in tests.items()
        },
        "Policies raised to the minimum": {
            n: f"{t.get('share_raised_to_minimum', 0):.1%}" for n, t in tests.items()
        },
    }
    col1, col2 = st.columns([2, 3])
    with col1:
        st.subheader("Summary")
        st.dataframe(pd.DataFrame(rows).T.rename_axis("Test set"), width="stretch")
        st.caption(
            "Loss ratio = actual claims cost ÷ pure premium. The rebase factor scales each model to "
            "the validation set's claims cost, so a test loss ratio near 1 shows it holds on new data."
        )
    with col2:
        st.subheader("Ranking risks: Lorenz curves")
        st.altair_chart(charts.lorenz_chart(data.report_table("lorenz"), ginis), width="stretch")
        st.caption("Policies sorted from lowest to highest predicted premium. The further the curve "
                   "bends below the diagonal, the better the model separates low and high risks.")

    st.subheader("Lift: actual vs predicted cost by premium band")
    cols = st.columns(2)
    for col, name in zip(cols, ginis):
        table = data.report_table(f"lift_{name}", index_col=0)
        with col:
            st.markdown(f"**{name}**")
            st.altair_chart(charts.lift_chart(table, name), width="stretch")
    st.caption("Each band holds 10% of the exposure. A good model has the actual line rising steeply "
               "and the predicted line following it.")

    st.subheader("Double lift: where the models disagree")
    double = data.report_table("double_lift", index_col=0)
    outer = double.iloc[[0, -1]]
    gap_glm = (outer["AIndex"] - outer["ActualIndex"]).abs().mean()
    gap_xgb = (outer["BIndex"] - outer["ActualIndex"]).abs().mean()
    closer = "GLM" if gap_glm < gap_xgb else "XGBoost"
    st.altair_chart(charts.double_lift_chart(double), width="stretch")
    st.markdown(
        f"Policies are sorted by how much XGBoost's premium differs from the GLM's. In the outer bands, "
        f"where they disagree most, the **{closer}** premium is closer to actual experience "
        f"(average gap {100 * min(gap_glm, gap_xgb):.1f} vs {100 * max(gap_glm, gap_xgb):.1f} "
        "percentage points of average cost)."
    )

    st.subheader("Actual vs expected by rating factor")
    factors = data.factor_names()
    factor = st.selectbox("Rating factor", factors, index=0)
    table = data.report_table(f"ae_{factor}", index_col=0)
    edges = cfg["report"]["factor_bands"][factor]
    ordered = (
        edges is not None
        or pd.api.types.is_numeric_dtype(table.index)
        or factor in cfg["report"]["ordered_categories"]
    )
    table.index = table.index.astype(str)
    st.altair_chart(charts.ae_chart(table, factor, ordered), width="stretch")
    ae = table[["GLMAE", "XGBoostAE", "Exposure"]].rename(columns={"GLMAE": "GLM A/E", "XGBoostAE": "XGBoost A/E"})
    with st.expander("Actual ÷ expected by band"):
        st.dataframe(ae.style.format({"GLM A/E": "{:.2f}", "XGBoost A/E": "{:.2f}", "Exposure": "{:,.0f}"}),
                     width="stretch")
    st.caption("A/E = actual cost ÷ predicted cost (1.00 is a perfect fit). Bands with little exposure are noisy.")

    st.subheader("Price changes if XGBoost replaced the GLM")
    dislocation = data.report_table("dislocation", index_col=0)
    shares = dislocation["Share"]
    big_up = shares.loc[["+10% to +25%", "> +25%"]].sum()
    big_down = shares.loc[["< -25%", "-25% to -10%"]].sum()
    col1, col2 = st.columns([3, 1])
    with col1:
        st.altair_chart(charts.dislocation_chart(dislocation), width="stretch")
    with col2:
        st.metric("Premium up by more than 10%", f"{big_up:.1%}")
        st.metric("Premium down by more than 10%", f"{big_down:.1%}")
        st.caption("Large moves for individual customers matter even when overall accuracy improves.")

# ----------------------------------------------------
# Interpretability
# ----------------------------------------------------
with interpretability_tab:
    glm_model, xgb_model = data.final_model("glm"), data.final_model("xgb")
    typical = glm_model.default_customer()
    typical_key = data.as_key(typical)

    col1, col2 = st.columns(2)
    with col1, st.container(border=True):
        st.markdown("**GLM: readable directly**")
        st.markdown(
            "The premium is a product of factors: a base rate × one relativity per rating factor "
            "(× any interaction). Each relativity can be read, checked and challenged in a table."
        )
    with col2, st.container(border=True):
        st.markdown("**XGBoost: explained from the outside**")
        n_trees = data.model_card("xgb")["n_trees"]
        st.markdown(
            f"The premium comes from {sum(n_trees.values()):,} trees across its models, which can't be read "
            "directly. It is explained by what-if curves and Shapley values: how the price moves "
            "when details change."
        )

    st.subheader("GLM relativities")
    covariates = cfg["glm"]["covariates"]
    factor = st.selectbox("GLM rating factor", covariates, key="relativity_factor")
    rel = data.glm_factor_relativities(factor)
    if rel is not None:
        st.altair_chart(charts.relativity_chart(rel, factor), width="stretch")
        st.caption("The base level (relativity 1) is the one with the most exposure.")
    else:
        table = data.relativities()
        per_unit = table[table["Term"] == factor]
        if per_unit.empty:
            st.info(f"`{factor}` isn't in the final GLM.")
        else:
            st.dataframe(per_unit, hide_index=True, width="stretch")
            st.caption(f"`{factor}` is numeric: the relativity is the multiplier per unit increase.")
    with st.expander("All GLM coefficients"):
        st.dataframe(data.relativities().style.format({"Coefficient": "{:.4f}", "Relativity": "{:.4f}"}),
                     hide_index=True, width="stretch")

    st.subheader("How each model's price responds to one detail")
    st.caption("Starting from the typical customer, one detail is changed at a time.")
    col1, col2 = st.columns([1, 2])
    with col1:
        spread = data.price_spread(typical_key)
        labels = {d: data.detail_label(d) for d in glm_model.input_fields}
        st.altair_chart(charts.price_range_chart(spread, labels), width="stretch")
        st.caption("How far the price moves across each detail's full range.")
    with col2:
        detail = st.selectbox("Detail", list(glm_model.input_fields), key="curve_detail",
                              format_func=data.detail_label)
        field = glm_model.input_fields[detail]
        curves = {name: data.price_curve(key, typical_key, detail) for key, name in data.MODELS.items()}
        numeric = field["type"] != "category"
        log_x = numeric and field["min"] > 0 and field["max"] / field["min"] > 100
        st.altair_chart(charts.price_curve_chart(curves, detail, typical[detail], numeric, log_x,
                                                 data.detail_label(detail)), width="stretch")

    st.subheader("The same customer, explained by both models")
    examples = cfg["report"]["example_customers"]
    example = st.selectbox("Example customer", list(examples), format_func=lambda s: s.replace("_", " ").capitalize())
    customer = {**typical, **(examples[example] or {})}
    changed = {k: v for k, v in customer.items() if v != typical[k]}
    st.markdown("Differs from the typical customer in: "
                + ", ".join(f"**{data.detail_label(k)}** = {v}" for k, v in changed.items()))
    cols = st.columns(2)
    for col, (key, name) in zip(cols, data.MODELS.items()):
        explanation = data.explain(key, data.as_key(customer))
        with col:
            st.markdown(f"**{name}**: {explanation['base_premium']:,.0f} → {explanation['pure_premium']:,.0f}")
            st.altair_chart(charts.waterfall_chart(explanation, customer, labels), width="stretch")
    st.caption(
        "Each bar is a detail's Shapley value: its average effect on the price over every mix of this "
        "customer's and the typical customer's details. The bars add up exactly to the price difference."
    )
