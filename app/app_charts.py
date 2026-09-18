"""Interactive (Altair) charts for the app. Colours match the report: GLM blue, XGBoost orange,
actual experience in grey. Every chart has hover tooltips."""

import altair as alt
import numpy as np
import pandas as pd

GLM_COLOR = "#2a78d6"
XGB_COLOR = "#eb6834"
ACTUAL_COLOR = "#898781"
INCREASE_COLOR = "#e34948"
DECREASE_COLOR = "#2a78d6"
TOTAL_COLOR = "#898781"

SERIES_COLORS = {"Actual": ACTUAL_COLOR, "GLM": GLM_COLOR, "XGBoost": XGB_COLOR}


# Charts on interactive pages are updated in place when inputs change, so category order is
# carried in the data (an "Order" column) rather than as a fixed list in the chart spec.
BY_ORDER = alt.EncodingSortField("Order", op="min")


def with_order(df, column, order):
    position = {value: i for i, value in enumerate(order)}
    return df.assign(Order=df[column].map(position))


def series_scale(names):
    names = list(names)
    return alt.Scale(domain=names, range=[SERIES_COLORS.get(n, ACTUAL_COLOR) for n in names])


def _lines(df, x, x_title, y_title, series, y_format=",.0f", x_sort=None, x_type="O", points=True, height=320):
    """df has columns x, 'Series', 'Value'."""
    if x_sort is not None:
        df = with_order(df, x, x_sort)
    x_enc = alt.X(f"{x}:{x_type}", title=x_title, sort=BY_ORDER if x_sort is not None else None,
                  axis=alt.Axis(labelAngle=0 if x_type != "O" else -30))
    base = alt.Chart(df).encode(
        x=x_enc,
        y=alt.Y("Value:Q", title=y_title, axis=alt.Axis(format=y_format), scale=alt.Scale(zero=False)),
        color=alt.Color("Series:N", scale=series_scale(series), legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip(f"{x}:{x_type}", title=x_title), "Series:N",
                 alt.Tooltip("Value:Q", title=y_title, format=y_format)],
    )
    chart = base.mark_line(strokeWidth=2)
    if points:
        chart = chart + base.mark_point(filled=True, size=60, opacity=1)
    return chart.properties(height=height)


# ----------------------------------------------------
# Ranking
# ----------------------------------------------------
def lorenz_chart(table, ginis):
    long = table.melt("ExposureShare", var_name="Series", value_name="Value")
    diagonal = alt.Chart(pd.DataFrame({"ExposureShare": [0, 1], "Value": [0, 1]})).mark_line(
        color=ACTUAL_COLOR, strokeWidth=1, strokeDash=[4, 4]
    ).encode(x="ExposureShare:Q", y="Value:Q")

    labels = {name: f"{name} (Gini {gini:.3f})" for name, gini in ginis.items()}
    long["Model"] = long["Series"].map(labels)
    lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=alt.X("ExposureShare:Q", title="Share of exposure (lowest predicted premium first)",
                axis=alt.Axis(format="%")),
        y=alt.Y("Value:Q", title="Share of claims cost", axis=alt.Axis(format="%")),
        color=alt.Color("Model:N", legend=alt.Legend(title=None, orient="top"),
                        scale=alt.Scale(domain=[labels[n] for n in ginis],
                                        range=[SERIES_COLORS[n] for n in ginis])),
        tooltip=["Model:N",
                 alt.Tooltip("ExposureShare:Q", title="Share of exposure", format=".0%"),
                 alt.Tooltip("Value:Q", title="Share of claims cost", format=".1%")],
    )
    return (diagonal + lines).properties(height=380)


def lift_chart(table, name):
    df = pd.DataFrame({
        "Band": table.index,
        "Actual": table["ActualBurningCost"].to_numpy(),
        name: table["PredictedBurningCost"].to_numpy(),
    }).melt("Band", var_name="Series", value_name="Value")
    return _lines(df, "Band", "Band of predicted premium (1 = lowest)", "Claims cost per policy year",
                  ["Actual", name], height=280)


def double_lift_chart(table):
    labels = [f"{lo:.2f}-{hi:.2f}" for lo, hi in zip(table["RatioMin"], table["RatioMax"])]
    df = pd.DataFrame({
        "Ratio": labels,
        "Actual": table["ActualIndex"].to_numpy(),
        "GLM": table["AIndex"].to_numpy(),
        "XGBoost": table["BIndex"].to_numpy(),
    }).melt("Ratio", var_name="Series", value_name="Value")
    return _lines(df, "Ratio", "XGBoost premium ÷ GLM premium (equal exposure bands)",
                  "Cost relative to average", ["Actual", "GLM", "XGBoost"], y_format=".0%", x_sort=labels)


# ----------------------------------------------------
# Rating factors
# ----------------------------------------------------
def ae_chart(table, factor, ordered=True):
    bands = list(table.index)
    df = pd.DataFrame({
        "Band": bands,
        "Actual": table["ActualBurningCost"].to_numpy(),
        "GLM": table["GLMBurningCost"].to_numpy(),
        "XGBoost": table["XGBoostBurningCost"].to_numpy(),
    }).melt("Band", var_name="Series", value_name="Value")
    df = with_order(df, "Band", bands)

    if ordered:
        top = _lines(df, "Band", factor, "Claims cost per policy year", ["Actual", "GLM", "XGBoost"],
                     x_sort=bands, height=280)
    else:
        top = alt.Chart(df).mark_point(filled=True, size=90, opacity=1).encode(
            x=alt.X("Band:N", title=factor, sort=BY_ORDER, axis=alt.Axis(labelAngle=-30)),
            xOffset=alt.XOffset("Series:N", sort=["Actual", "GLM", "XGBoost"]),
            y=alt.Y("Value:Q", title="Claims cost per policy year", scale=alt.Scale(zero=False)),
            color=alt.Color("Series:N", scale=series_scale(["Actual", "GLM", "XGBoost"]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=["Band:N", "Series:N", alt.Tooltip("Value:Q", title="Cost per policy year", format=",.0f")],
        ).properties(height=280)

    exposure = pd.DataFrame({"Band": bands, "Exposure": table["Exposure"].to_numpy(), "Order": range(len(bands))})
    bottom = alt.Chart(exposure).mark_bar(color=ACTUAL_COLOR, opacity=0.6, cornerRadiusEnd=3).encode(
        x=alt.X("Band:N", title=factor, sort=BY_ORDER, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("Exposure:Q", title="Total Exposure (policy years)", axis=alt.Axis(format="~s")),
        tooltip=["Band:N", alt.Tooltip("Exposure:Q", format=",.0f")],
    ).properties(height=110)

    return alt.vconcat(top, bottom).resolve_scale(x="shared")


def dislocation_chart(table):
    df = table.reset_index().rename(columns={"index": "Change"})
    df.columns = ["Change", "Policies", "Share"]
    order = list(df["Change"])
    bars = alt.Chart(df).mark_bar(color=XGB_COLOR, cornerRadiusEnd=4, size=40).encode(
        x=alt.X("Change:N", sort=order, title="Change in premium, GLM → XGBoost", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Share:Q", title="Share of policies", axis=alt.Axis(format="%")),
        tooltip=["Change:N", alt.Tooltip("Policies:Q", format=","), alt.Tooltip("Share:Q", format=".1%")],
    )
    text = bars.mark_text(dy=-8).encode(text=alt.Text("Share:Q", format=".0%"), color=alt.value("gray"))
    return (bars + text).properties(height=300)


# ----------------------------------------------------
# Model selection
# ----------------------------------------------------
def ranked_bars(df, label, value, title, highlight, fmt=".4f", ascending=False):
    """Horizontal bars, best first, with the chosen option highlighted."""
    df = df.sort_values(value, ascending=ascending).copy()
    df["Chosen"] = np.where(df[label] == highlight, "Chosen", "Other")
    df["Order"] = range(len(df))
    chart = alt.Chart(df).mark_bar(cornerRadiusEnd=4, size=22).encode(
        y=alt.Y(f"{label}:N", sort=BY_ORDER, title=None, axis=alt.Axis(labelLimit=320)),
        x=alt.X(f"{value}:Q", title=title),
        color=alt.Color("Chosen:N", scale=alt.Scale(domain=["Chosen", "Other"], range=[GLM_COLOR, "#c3c2b7"]),
                        legend=None),
        tooltip=[f"{label}:N", alt.Tooltip(f"{value}:Q", title=title, format=fmt)],
    )
    text = chart.mark_text(align="left", dx=4).encode(text=alt.Text(f"{value}:Q", format=fmt),
                                                       color=alt.value("gray"))
    return (chart + text).properties(height=alt.Step(34))


def grid_heatmap(df, score_title):
    df = df.copy()
    df["max_depth"] = df["max_depth"].astype(str)
    df["learning_rate"] = df["learning_rate"].astype(str)
    midpoint = (df["score"].min() + df["score"].max()) / 2
    df["Dark"] = df["score"] < midpoint  # lower score = darker cell
    base = alt.Chart(df).encode(
        x=alt.X("learning_rate:N", title="Learning rate", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("max_depth:N", title="Max depth"),
    )
    cells = base.mark_rect(cornerRadius=4).encode(
        color=alt.Color("score:Q", title=score_title, scale=alt.Scale(scheme="blues", reverse=True)),
        tooltip=["max_depth:N", "learning_rate:N", alt.Tooltip("score:Q", title=score_title, format=".5f"),
                 alt.Tooltip("n_trees:Q", title="Trees (early stopping)")],
    )
    text = base.mark_text(fontSize=12).encode(
        text=alt.Text("score:Q", format=".4f"),
        color=alt.condition("datum.Dark", alt.value("white"), alt.value("black")),
    )
    return (cells + text).properties(height=160)


# ----------------------------------------------------
# Quote
# ----------------------------------------------------
def _fmt(value):
    return f"{value:g}" if isinstance(value, (int, float, np.number)) else str(value)


def waterfall_chart(explanation, customer, labels=None):
    """Typical customer's premium, then each detail's effect, ending at this customer's premium."""
    effects = explanation["effects"]
    changed = effects[effects["LogEffect"] != 0].sort_values("PremiumImpact", key=abs, ascending=False)

    rows = [{"Step": "Typical customer", "Start": 0.0, "End": explanation["base_premium"],
             "Kind": "Premium", "Detail": "", "Factor": np.nan}]
    running = explanation["base_premium"]
    for detail, row in changed.iterrows():
        rows.append({
            "Step": f"{(labels or {}).get(detail, detail)}: {_fmt(row['Reference'])} → {_fmt(customer[detail])}",
            "Start": running, "End": running + row["PremiumImpact"],
            "Kind": "Raises price" if row["PremiumImpact"] > 0 else "Lowers price",
            "Detail": detail, "Factor": row["Factor"],
        })
        running += row["PremiumImpact"]
    rows.append({"Step": "Your pure premium", "Start": 0.0, "End": explanation["pure_premium"],
                 "Kind": "Premium", "Detail": "", "Factor": np.nan})

    df = pd.DataFrame(rows)
    df["Change"] = df["End"] - df["Start"]
    # the order lives in the data, so it stays right when Streamlit updates the chart in place
    df["Order"] = range(len(df))
    bars = alt.Chart(df).mark_bar(size=22, cornerRadius=3).encode(
        y=alt.Y("Step:N", sort=BY_ORDER, title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X("Start:Q", title="Annual pure premium"),
        x2="End:Q",
        color=alt.Color("Kind:N", legend=alt.Legend(title=None, orient="top"),
                        scale=alt.Scale(domain=["Premium", "Raises price", "Lowers price"],
                                        range=[TOTAL_COLOR, INCREASE_COLOR, DECREASE_COLOR])),
        tooltip=["Step:N", alt.Tooltip("Change:Q", title="Effect", format="+,.2f"),
                 alt.Tooltip("Factor:Q", title="Multiplies premium by", format=".3f"),
                 alt.Tooltip("End:Q", title="Premium after this step", format=",.2f")],
    )
    return bars.properties(height=alt.Step(38))


def price_curve_chart(curves, detail, current_value, numeric, log_x=False, title=None):
    """curves: model name -> DataFrame from price_curve."""
    frames = [c[[detail, "AnnualPremium"]].assign(Series=name) for name, c in curves.items()]
    df = pd.concat(frames, ignore_index=True)
    series = list(curves)

    if numeric:
        x = alt.X(f"{detail}:Q", title=title or detail,
                  scale=alt.Scale(type="log") if log_x else alt.Scale(zero=False))
        chart = alt.Chart(df).mark_line(strokeWidth=2, interpolate="step-after").encode(
            x=x,
            y=alt.Y("AnnualPremium:Q", title="Annual premium", scale=alt.Scale(zero=False)),
            color=alt.Color("Series:N", scale=series_scale(series), legend=alt.Legend(title=None, orient="top")),
            tooltip=[alt.Tooltip(f"{detail}:Q", format=",.0f"), "Series:N",
                     alt.Tooltip("AnnualPremium:Q", title="Annual premium", format=",.2f")],
        )
        rule = alt.Chart(pd.DataFrame({detail: [current_value]})).mark_rule(
            color=ACTUAL_COLOR, strokeDash=[4, 4]
        ).encode(x=f"{detail}:Q")
        return (chart + rule).properties(height=280)

    df["Selected"] = np.where(df[detail].astype(str) == str(current_value), "Your value", "Other")
    chart = alt.Chart(df).mark_point(filled=True, opacity=1).encode(
        x=alt.X(f"{detail}:N", title=title or detail, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("Series:N"),
        y=alt.Y("AnnualPremium:Q", title="Annual premium", scale=alt.Scale(zero=False)),
        color=alt.Color("Series:N", scale=series_scale(series), legend=alt.Legend(title=None, orient="top")),
        size=alt.Size("Selected:N", scale=alt.Scale(domain=["Your value", "Other"], range=[220, 70]),
                      legend=None),
        tooltip=[f"{detail}:N", "Series:N", alt.Tooltip("AnnualPremium:Q", title="Annual premium", format=",.2f")],
    )
    return chart.properties(height=280)


def price_range_chart(df, labels=None):
    """How much the premium can move when one detail changes (the rest held at typical values).
    df: Detail, Series, Spread (highest ÷ lowest premium - 1)."""
    order = list(df.groupby("Detail")["Spread"].max().sort_values(ascending=False).index)
    df = with_order(df, "Detail", order)
    if labels:
        df["Detail"] = df["Detail"].map(lambda d: labels.get(d, d))
    chart = alt.Chart(df).mark_bar(cornerRadiusEnd=3, size=10).encode(
        y=alt.Y("Detail:N", sort=BY_ORDER, title=None),
        yOffset=alt.YOffset("Series:N"),
        x=alt.X("Spread:Q", title="Price range (highest vs lowest)",
                axis=alt.Axis(format="%")),
        color=alt.Color("Series:N", scale=series_scale(["GLM", "XGBoost"]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=["Detail:N", "Series:N", alt.Tooltip("Spread:Q", title="Highest vs lowest", format="+.0%")],
    )
    return chart.properties(height=alt.Step(26))


def relativity_chart(df, factor):
    """df: Model, Level, Relativity (base level = 1)."""
    levels = list(dict.fromkeys(df["Level"]))
    df = with_order(df, "Level", levels)
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=3, size=12).encode(
        y=alt.Y("Level:N", sort=BY_ORDER, title=factor),
        yOffset=alt.YOffset("Model:N"),
        x=alt.X("Relativity:Q", title="Relativity (base level = 1)"),
        color=alt.Color("Model:N", scale=alt.Scale(domain=["Frequency", "Severity"], range=[GLM_COLOR, "#86b6ef"]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=["Level:N", "Model:N", alt.Tooltip("Relativity:Q", format=".3f")],
    )
    one = alt.Chart(pd.DataFrame({"x": [1]})).mark_rule(color=ACTUAL_COLOR, strokeDash=[4, 4]).encode(x="x:Q")
    return (bars + one).properties(height=alt.Step(30))
