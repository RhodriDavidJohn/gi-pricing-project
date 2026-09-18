"""Report charts (static PNGs for the README / results write-up).

Colours: GLM blue and XGBoost orange throughout (checked for colour-blind separation),
actual experience in dark grey. Each function returns a matplotlib Figure; save_figure
writes it. Uses matplotlib's Figure API directly, so no display is needed.
"""

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, PercentFormatter

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

MODEL_COLORS = {"GLM": "#2a78d6", "XGBoost": "#eb6834"}
ACTUAL_COLOR = INK_SECONDARY
INCREASE_COLOR = "#e34948"
DECREASE_COLOR = "#2a78d6"

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.edgecolor": AXIS,
    "axes.linewidth": 1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 1,
    "grid.linestyle": "-",
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "lines.linewidth": 2,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "lines.markersize": 5,
    "legend.frameon": False,
    "legend.labelcolor": INK_SECONDARY,
}

money = FuncFormatter(lambda value, _: f"{value:,.0f}")


def new_figure(width=8, height=4.5, **subplot_kw):
    with mpl.rc_context(STYLE):
        fig = Figure(figsize=(width, height), layout="constrained")
        axes = fig.subplots(**subplot_kw)
    return fig, axes


def save_figure(fig, path):
    with mpl.rc_context(STYLE):
        fig.savefig(path, dpi=150)


def _title(fig, title, subtitle=None):
    """Title and subtitle in a fixed-height header (in inches), so tall figures don't get a big gap."""
    height = fig.get_figheight()
    fig.text(0.01, 1 - 0.12 / height, title, ha="left", va="top", fontsize=13,
             fontweight="semibold", color=INK)
    header = 0.45
    if subtitle:
        fig.text(0.01, 1 - 0.42 / height, subtitle, ha="left", va="top", fontsize=9.5,
                 color=INK_SECONDARY)
        header = 0.7
    fig.get_layout_engine().set(rect=(0, 0, 1, 1 - header / height))


def _actual_line(ax, x, y, label="Actual", joined=True):
    ax.plot(x, y, color=ACTUAL_COLOR, marker="o", markersize=6 if joined else 8, markeredgecolor=SURFACE,
            markeredgewidth=1.5, linestyle="-" if joined else "none", label=label, zorder=3)


def _model_line(ax, x, y, name, label=None, joined=True):
    ax.plot(x, y, color=MODEL_COLORS[name], marker="o", markersize=6 if joined else 8,
            markeredgecolor=SURFACE, markeredgewidth=1.5, linestyle="-" if joined else "none",
            label=label or name, zorder=4)


# ----------------------------------------------------
# Ranking
# ----------------------------------------------------


def lorenz_chart(curves, ginis):
    """curves: model name -> (cumulative exposure, cumulative cost)."""
    with mpl.rc_context(STYLE):
        fig, ax = new_figure(6.5, 5.5)
        ax.grid(True, axis="both")
        ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, label="No ranking (Gini 0)")
        for name, (x, y) in curves.items():
            # thin the points - a line of 100k points looks the same as 500
            idx = np.unique(np.linspace(0, len(x) - 1, 500).astype(int))
            ax.plot(x[idx], y[idx], color=MODEL_COLORS[name], label=f"{name} (Gini {ginis[name]:.3f})")

        ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal",
               xlabel="Cumulative share of exposure (lowest predicted premium first)",
               ylabel="Cumulative share of claims cost")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(loc="upper left")
        _title(fig, "Lorenz curves - test set",
               "The further below the diagonal, the better the model separates low and high risks")
    return fig


def lift_chart(tables):
    """tables: model name -> lift_table."""
    with mpl.rc_context(STYLE):
        fig, axes = new_figure(10, 4.5, ncols=len(tables), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, (name, table) in zip(axes, tables.items()):
            _actual_line(ax, table.index, table["ActualBurningCost"])
            _model_line(ax, table.index, table["PredictedBurningCost"], name, f"{name} predicted")
            ax.set_title(name)
            ax.set_xticks(table.index)
            ax.set_xlabel("Band of predicted premium (equal exposure, 1 = lowest)")
            ax.yaxis.set_major_formatter(money)
            ax.legend(loc="upper left")
        axes[0].set_ylabel("Claims cost per policy year")
        _title(fig, "Lift charts - test set",
               "Actual vs predicted cost by band of predicted premium; a steeper actual line means better ranking")
    return fig


def double_lift_chart(table, name_a="GLM", name_b="XGBoost"):
    with mpl.rc_context(STYLE):
        fig, ax = new_figure(9, 5)
        x = table.index
        _actual_line(ax, x, table["ActualIndex"])
        _model_line(ax, x, table["AIndex"], name_a)
        _model_line(ax, x, table["BIndex"], name_b)
        ax.axhline(1, color=AXIS, linewidth=1, zorder=1)

        labels = [f"{lo:.2f}-{hi:.2f}" for lo, hi in zip(table["RatioMin"], table["RatioMax"])]
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_xlabel(f"{name_b} premium ÷ {name_a} premium (equal exposure bands)")
        ax.set_ylabel("Cost relative to average")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(loc="upper left")
        _title(fig, f"Double lift: {name_b} vs {name_a} - test set",
               "Where the models disagree most (left and right), the better model's line follows the actual line")
    return fig


# ----------------------------------------------------
# Rating factors
# ----------------------------------------------------


def factor_chart(table, factor, models, ordered=True):
    """Actual vs predicted cost by band (top) with exposure by band underneath - two panels
    rather than a second y-axis. Unordered categories (ordered=False) are shown as dots."""
    with mpl.rc_context(STYLE):
        fig, (ax, ax_exp) = new_figure(9, 6, nrows=2, sharex=True, height_ratios=[3, 1])
        x = np.arange(len(table))

        # unordered categories: dots side by side so equal values don't hide each other
        offsets = [0] * (len(models) + 1) if ordered else np.linspace(-0.2, 0.2, len(models) + 1)
        _actual_line(ax, x + offsets[0], table["ActualBurningCost"], joined=ordered)
        for offset, name in zip(offsets[1:], models):
            _model_line(ax, x + offset, table[f"{name}BurningCost"], name, joined=ordered)
        ax.set_ylabel("Claims cost per policy year")
        ax.yaxis.set_major_formatter(money)

        # show at least +/-20% around the average, so small differences don't look dramatic
        average = table["Actual"].sum() / table["Exposure"].sum()
        low, high = ax.get_ylim()
        pad = 0.05 * (high - low)
        ax.set_ylim(min(low - pad, 0.8 * average), max(high + pad, 1.2 * average))
        ax.legend(loc="best")

        ax_exp.bar(x, table["Exposure"], width=0.6, color=AXIS)
        ax_exp.set_ylabel("Exposure (years)")
        ax_exp.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k" if v else "0"))
        ax_exp.set_xticks(x, table.index, rotation=35 if len(x) > 8 else 0,
                          ha="right" if len(x) > 8 else "center")
        ax_exp.set_xlabel(factor)

        _title(fig, f"Actual vs expected by {factor} - test set",
               "Claims cost per policy year; thin bands (small exposure) are noisy")
    return fig


# ----------------------------------------------------
# Price changes and effects
# ----------------------------------------------------


def dislocation_chart(table, name_from="GLM", name_to="XGBoost"):
    with mpl.rc_context(STYLE):
        fig, ax = new_figure(9, 4.5)
        x = np.arange(len(table))
        ax.bar(x, table["Share"], width=0.6, color=MODEL_COLORS[name_to])
        for xi, share in zip(x, table["Share"]):
            label = f"{share:.0%}" if share >= 0.01 or share == 0 else f"{share:.1%}"
            ax.text(xi, share, label, ha="center", va="bottom", fontsize=9, color=INK_SECONDARY)

        ax.set_xticks(x, table.index)
        ax.set_xlabel("Change in premium")
        ax.set_ylabel("Share of policies")
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        _title(fig, f"Price changes moving from {name_from} to {name_to} - test set",
               "Share of policies by change in annual pure premium (both models rebased)")
    return fig


def price_curves_chart(curves, details):
    """curves: detail -> model name -> DataFrame from PricingModel.price_curve."""
    n_cols = 2
    n_rows = int(np.ceil(len(details) / n_cols))
    with mpl.rc_context(STYLE):
        fig, axes = new_figure(10, 3.2 * n_rows, nrows=n_rows, ncols=n_cols, squeeze=False)
        for ax, detail in zip(axes.flat, details):
            for name, curve in curves[detail].items():
                x = curve[detail]
                if pd.api.types.is_numeric_dtype(x):
                    # steps: grouped GLM factors and tree splits both give flat stretches
                    ax.plot(x, curve["AnnualPremium"], color=MODEL_COLORS[name], label=name,
                            drawstyle="steps-mid")
                else:
                    # categories: dots, not a line joining unrelated values
                    positions = np.arange(len(curve))
                    ax.plot(positions, curve["AnnualPremium"], linestyle="none", marker="o", markersize=9,
                            markeredgecolor=SURFACE, markeredgewidth=1.5, color=MODEL_COLORS[name],
                            label=name)
                    ax.set_xticks(positions, x)
                    ax.margins(x=0.2)
            ax.set_title(detail)
            ax.yaxis.set_major_formatter(money)
            if detail == "Density":
                ax.set_xscale("log")
        for ax in axes.flat[len(details):]:
            ax.set_visible(False)
        axes.flat[0].legend(loc="best")
        _title(fig, "How the annual premium responds to each detail",
               "Final models; one detail changed at a time, others at the typical customer's values")
    return fig


def effects_chart(explanation, customer, model_name):
    effects = explanation["effects"].sort_values("PremiumImpact")
    with mpl.rc_context(STYLE):
        fig, ax = new_figure(8, 4.5)
        y = np.arange(len(effects))
        colors = [INCREASE_COLOR if v > 0 else DECREASE_COLOR for v in effects["PremiumImpact"]]
        ax.barh(y, effects["PremiumImpact"], height=0.6, color=colors)
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)

        labels = [
            f"{detail} = {customer[detail]}" + ("" if customer[detail] == row.Reference else f" (vs {row.Reference})")
            for detail, row in effects.iterrows()
        ]
        ax.set_yticks(y, labels)
        for yi, value in zip(y, effects["PremiumImpact"]):
            if abs(value) >= 0.5:
                ax.text(value, yi, f" {value:+,.0f} ", va="center", fontsize=9, color=INK_SECONDARY,
                        ha="left" if value > 0 else "right")
        ax.xaxis.set_major_formatter(money)
        ax.set_xlabel("Effect on annual pure premium (red raises it, blue lowers it)")
        ax.margins(x=0.15)
        _title(fig, f"Why this customer pays what they pay ({model_name})",
               f"Typical customer {explanation['base_premium']:,.0f} → this customer "
               f"{explanation['pure_premium']:,.0f}; each bar is a detail's Shapley effect")
    return fig


# ----------------------------------------------------
# Claims
# ----------------------------------------------------


def claim_size_chart(claim_amounts, summary):
    amounts = np.asarray(claim_amounts, dtype=float)
    amounts = amounts[amounts > 0]
    with mpl.rc_context(STYLE):
        fig, ax = new_figure(9, 4.5)
        bins = np.geomspace(amounts.min(), amounts.max(), 60)
        ax.hist(amounts, bins=bins, color=MODEL_COLORS["GLM"], rwidth=0.85)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(money)

        threshold = summary["large_threshold"]
        ax.axvline(threshold, color=INK_SECONDARY, linewidth=1)
        ax.text(threshold, ax.get_ylim()[1] * 0.95,
                f"  top {100 - summary['large_percentile']:.0f}% of claims (> {threshold:,.0f})\n"
                f"  = {summary['large_share_of_cost']:.0%} of total cost",
                va="top", fontsize=9, color=INK_SECONDARY)
        ax.set_xlabel("Claim amount (log scale)")
        ax.set_ylabel("Number of claims")
        _title(fig, "Claim sizes - all data",
               f"{summary['n_claims']:,} claims, median {summary['median']:,.0f}, "
               f"mean {summary['mean']:,.0f}, largest {summary['max']:,.0f}")
    return fig
