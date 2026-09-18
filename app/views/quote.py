import pandas as pd
import streamlit as st

import app_charts as charts
import app_data as data

data.check_outputs()
cfg = data.config()

models = {key: data.final_model(key) for key in data.MODELS}
fields = models["glm"].input_fields
typical = models["glm"].default_customer()


label = data.detail_label


def reset_details():
    for detail, value in typical.items():
        st.session_state[f"detail_{detail}"] = value


# ----------------------------------------------------
# Inputs
# ----------------------------------------------------
for detail, value in typical.items():
    st.session_state.setdefault(f"detail_{detail}", value)

with st.sidebar:
    st.header("Your details")
    customer = {}
    for detail, field in fields.items():
        name, help_text = data.DETAIL_LABELS.get(detail, (detail, None))
        key = f"detail_{detail}"
        if field["type"] == "category":
            customer[detail] = st.selectbox(name, field["options"], key=key, help=help_text)
        elif detail == "Density":
            customer[detail] = int(st.number_input(name, min_value=field["min"], max_value=field["max"],
                                                   step=50, key=key, help=help_text))
        else:
            customer[detail] = st.slider(name, min_value=field["min"], max_value=field["max"],
                                         key=key, help=help_text)
    st.button("Reset to the typical customer", on_click=reset_details, width="stretch")

# ----------------------------------------------------
# Quote
# ----------------------------------------------------
st.title("Get a quote")
default_model = cfg.get("app", {}).get("default_model", "glm")
model_key = st.segmented_control(
    "Pricing model",
    options=list(data.MODELS),
    format_func=data.MODELS.get,
    default=default_model,
    help="The GLM is the default because every step of its calculation can be shown. "
         "XGBoost is explained with Shapley values.",
)
model_key = model_key or default_model
model_name = data.MODELS[model_key]
other_key = next(k for k in data.MODELS if k != model_key)

customer_key = data.as_key(customer)
quote = data.quote(model_key, customer_key)
typical_quote = data.quote(model_key, data.as_key(typical))
other_quote = data.quote(other_key, customer_key)

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Annual pure premium", f"{quote['AnnualPremium']:,.2f}",
    delta=f"{quote['AnnualPremium'] - typical_quote['AnnualPremium']:+,.2f} vs typical customer",
    delta_color="inverse",
)
col2.metric("Monthly", f"{quote['MonthlyPremium']:,.2f}",
            help=f"Annual premium ÷ 12, plus a {cfg['pricing']['monthly_loading']:.0%} monthly payment loading.")
col3.metric("Expected claims per year", f"{quote['AnnualFrequency']:.4f}")
col4.metric("Average cost per claim", f"{quote['AverageClaimCost']:,.0f}")

notes = [
    f"The {data.MODELS[other_key]} model would quote **{other_quote['AnnualPremium']:,.2f}** for the same details."
]
if quote["MinimumPremiumApplied"]:
    notes.append(
        f"The model's price ({quote['PurePremium']:,.2f}) is below the minimum premium, so the minimum "
        f"of {models[model_key].minimum_premium:,.2f} applies."
    )
st.markdown("  \n".join(notes))
st.caption("Pure premium = expected claims cost for one year = expected claims × average claim cost. "
           "It doesn't include expenses, commission or profit.")

# ----------------------------------------------------
# How the price was built
# ----------------------------------------------------
st.header("How your price was built")
explanation = data.explain(model_key, customer_key)
effects = explanation["effects"]
changed = effects[effects["LogEffect"] != 0].sort_values("PremiumImpact", key=abs, ascending=False)

if changed.empty:
    st.info("These are the typical customer's details. Change any detail in the sidebar to see how it "
            "moves the price.")
else:
    col1, col2 = st.columns([3, 2])
    with col1:
        st.altair_chart(charts.waterfall_chart(explanation, customer, {d: label(d) for d in fields}),
                        width="stretch")
    with col2:
        st.markdown(
            f"Starting from the **typical customer** ({explanation['base_premium']:,.2f}), your details "
            f"move the pure premium to **{explanation['pure_premium']:,.2f}**:"
        )
        lines = []
        for detail, row in changed.head(5).iterrows():
            direction = "raises" if row["PremiumImpact"] > 0 else "lowers"
            lines.append(
                f"- **{label(detail)}** {row['Value']} (typical {row['Reference']}) {direction} the price by "
                f"{abs(row['PremiumImpact']):,.2f} (× {row['Factor']:.3f})"
            )
        st.markdown("\n".join(lines))
        split = changed[["FrequencyEffect", "SeverityEffect"]].sum()
        st.caption(
            f"Of the total change, {split['FrequencyEffect']:+.3f} (log scale) comes from how often you are "
            f"expected to claim and {split['SeverityEffect']:+.3f} from how much each claim costs."
        )

    with st.expander("All details: effect on your price"):
        table = effects.rename(index=lambda d: label(d))[
            ["Value", "Reference", "Factor", "PremiumImpact", "FrequencyEffect", "SeverityEffect"]
        ].rename(columns={"Reference": "Typical", "Factor": "Multiplies price by",
                          "PremiumImpact": "Effect on price",
                          "FrequencyEffect": "Effect on claim frequency (log)",
                          "SeverityEffect": "Effect on claim cost (log)"})
        table[["Value", "Typical"]] = table[["Value", "Typical"]].astype(str)
        st.dataframe(table.style.format({
            "Multiplies price by": "{:.3f}", "Effect on price": "{:+,.2f}",
            "Effect on claim frequency (log)": "{:+.4f}", "Effect on claim cost (log)": "{:+.4f}",
        }), width="stretch")

st.caption(
    "Each effect is the detail's exact Shapley value: its average effect on the price over every mix of "
    "your details and the typical customer's. The effects multiply out exactly to your price."
)

if model_key == "glm":
    with st.expander("The GLM calculation, step by step", expanded=False):
        calc = models["glm"].calculation(customer)
        terms = calc["terms"]
        st.markdown(
            "The GLM multiplies a base rate by a factor for each rating factor that applies to you. "
            "Grouped factors show your group against the base group; numeric factors are "
            "exp(coefficient × your value)."
        )
        col1, col2 = st.columns(2)
        for col, part, result, unit in [
            (col1, "Frequency", calc["frequency"], "expected claims per year"),
            (col2, "Severity", calc["severity"] / calc["smearing_factor"], "average cost per claim"),
        ]:
            with col:
                st.markdown(f"**{part}** - product of factors = {result:,.4f} {unit}")
                st.dataframe(
                    terms[terms["Model"] == part].drop(columns="Model").style.format(
                        {"Value": "{:,.3f}", "Coefficient": "{:+.4f}", "Factor": "{:,.4f}"}
                    ),
                    hide_index=True, width="stretch",
                )
        st.markdown(
            f"""
| Step | Value |
|---|---|
| Expected claims per year | {calc['frequency']:,.4f} |
| × average cost per claim (factors {calc['severity'] / calc['smearing_factor']:,.2f} × smearing {calc['smearing_factor']:.4f}) | {calc['severity']:,.2f} |
| × rebase factor (scales the model to the portfolio's claims cost) | {calc['rebase_factor']:.4f} |
| **= pure premium** | **{calc['pure_premium']:,.2f}** |
| Minimum premium | {models['glm'].minimum_premium or 0:,.2f} |
| **Annual premium (the higher of the two)** | **{max(calc['pure_premium'], models['glm'].minimum_premium or 0):,.2f}** |
"""
        )
        if calc["smearing_factor"] == 1:
            st.caption("The severity model is a Gamma GLM, so no smearing correction is needed (factor 1).")

# ----------------------------------------------------
# What if
# ----------------------------------------------------
st.header("What if one detail changed?")
detail = st.selectbox("Detail", list(fields), format_func=label, key="what_if_detail")
field = fields[detail]
curves = {
    data.MODELS[key]: data.price_curve(key, customer_key, detail)
    for key in [model_key, other_key]
}
numeric = field["type"] != "category"
log_x = numeric and field["min"] > 0 and field["max"] / field["min"] > 100
st.altair_chart(charts.price_curve_chart(curves, detail, customer[detail], numeric, log_x, label(detail)),
                width="stretch")
st.caption(f"Your other details stay as they are. The dashed line (or the larger dot) marks your "
           f"{label(detail).lower()}. Both models are shown for comparison.")

with st.expander("Price table"):
    table = pd.DataFrame({detail: curves[model_name][detail]})
    for name, curve in curves.items():
        table[f"{name} annual premium"] = curve["AnnualPremium"].to_numpy()
    st.dataframe(table, hide_index=True, width="stretch")
