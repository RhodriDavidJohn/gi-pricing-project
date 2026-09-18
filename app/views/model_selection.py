import pandas as pd
import streamlit as st

import app_charts as charts
import app_data as data

data.check_outputs()
cfg = data.config()

st.title("Model selection")
st.markdown(
    f"""
The data is split once into **training** ({1 - cfg['split']['val_size'] - cfg['split']['test_size']:.0%}),
**validation** ({cfg['split']['val_size']:.0%}) and **test** ({cfg['split']['test_size']:.0%}) sets,
stratified on whether a policy claimed. Both models go through the same three steps:

1. **Structure** - GLM interactions / XGBoost settings, chosen on the training and validation sets.
2. **Components** - frequency and severity candidates compared by validation deviance (lower is better).
3. **Pair** - the frequency × severity pair with the highest validation **Gini** (how well the pure
   premium ranks risks) is chosen. The test set is used only after this, on the GLM vs XGBoost page.
"""
)


def pair_label(row):
    return f"{data.FAMILY_NAMES.get(row['frequency'], row['frequency'])} × " \
           f"{data.FAMILY_NAMES.get(row['severity'], row['severity'])}"


def pair_section(key, chosen_freq, chosen_sev):
    pairs = data.selection_table(key, "pair_comparison_val")
    pairs["Pair"] = pairs.apply(pair_label, axis=1)
    chosen = pair_label({"frequency": chosen_freq, "severity": chosen_sev})
    st.altair_chart(
        charts.ranked_bars(pairs, "Pair", "val_gini", "Validation Gini (pure premium)", chosen),
        width="stretch",
    )
    st.caption(f"Chosen: **{chosen}**. Ties (to 4 decimal places) go to the simpler model.")


glm_tab, xgb_tab = st.tabs(["GLM", "XGBoost"])

# ----------------------------------------------------
# GLM
# ----------------------------------------------------
with glm_tab:
    glm_cfg = cfg["glm"]
    metrics = data.selection_metrics("glm")
    spec = data.selected_spec("glm")
    criterion = glm_cfg["criterion"]

    st.header("1. Interactions")
    st.markdown(
        f"Every model starts from the same rating factors: {', '.join(f'`{c}`' for c in glm_cfg['covariates'])}. "
        f"Candidate interactions are then added one at a time (forward selection), keeping the one that "
        f"lowers **{criterion.upper()}** the most, until none improves it by at least "
        f"{glm_cfg['min_improvement']}. {criterion.upper()} charges for every extra parameter, so an "
        "interaction has to earn its place."
    )
    st.dataframe(
        pd.DataFrame(
            [{"Candidate interaction": term, "Why test it": reason}
             for term, reason in glm_cfg["interactions_to_test"].items()]
        ),
        hide_index=True, width="stretch",
    )

    histories = []
    kept_lines = []
    for name, file in [
        ("Frequency (Poisson)", "interaction_selection_frequency"),
        ("Severity (Gamma)", "interaction_selection_gamma"),
        ("Severity (log-normal)", "interaction_selection_log_normal"),
    ]:
        history = data.selection_table("glm", file)
        added = [a for a in history["added"] if a != "(none)"]
        kept_lines.append(f"- **{name}**: " + (", ".join(f"`{a}`" for a in added) if added else "no interactions"))
        histories.append(pd.DataFrame({
            "Model": name,
            "Step": history["step"],
            "Added": history["added"].replace("(none)", "(start: rating factors only)"),
            criterion.upper(): history[criterion],
            "Improvement": history["improvement"],
        }))
    st.markdown("**Interactions kept**\n" + "\n".join(kept_lines))
    with st.expander("Selection steps"):
        st.dataframe(
            pd.concat(histories, ignore_index=True).style.format(
                {criterion.upper(): "{:,.1f}", "Improvement": "{:,.1f}"}, na_rep="-"
            ),
            hide_index=True, width="stretch",
        )

    st.header("2. Frequency and severity models")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Frequency** (validation set)")
        freq = pd.DataFrame(
            [{"Model": data.FAMILY_NAMES[k], "Poisson deviance": v["poisson_deviance"], "Gini": v["gini"]}
             for k, v in metrics["frequency"].items() if isinstance(v, dict)]
        )
        st.dataframe(freq.style.format({"Poisson deviance": "{:.5f}", "Gini": "{:.4f}"}),
                     hide_index=True, width="stretch")
        alpha = metrics["frequency"].get("nb_alpha")
        if alpha is not None:
            st.caption(
                f"Negative binomial dispersion estimated at α = {alpha:.3g}. "
                "Values near 0 mean little overdispersion, so it behaves like the Poisson model."
            )
    with col2:
        st.markdown("**Severity** (validation claims, capped amounts)")
        sev = data.selection_table("glm", "severity_comparison_val")
        sev = sev.set_index(sev.columns[0]).rename_axis("Metric")
        sev.columns = [data.FAMILY_NAMES.get(c, c) for c in sev.columns]
        st.dataframe(sev.style.format("{:,.3f}"), width="stretch")
        st.caption("Gamma deviance is the main comparison; MAE and RMSE are in currency.")

    st.header("3. Frequency × severity pair")
    pair_section("glm", spec["frequency"]["family"], spec["severity"]["family"])

    with st.expander("Selected specification and model summaries"):
        st.json(spec)
        for label, file in [("Frequency", "frequency_summary.txt"), ("Severity", "severity_summary.txt")]:
            path = data.selection_file("glm", file)
            if path:
                st.markdown(f"**{label} model (training set)**")
                st.code(path.read_text(), language=None)

# ----------------------------------------------------
# XGBoost
# ----------------------------------------------------
with xgb_tab:
    xgb_cfg = cfg["xgb"]
    metrics = data.selection_metrics("xgb")
    spec = data.selected_spec("xgb")

    st.header("1. Settings")
    grid_desc = ", ".join(f"`{k}` in {v}" for k, v in xgb_cfg["param_grid"].items())
    st.markdown(
        f"Each model is fitted for every combination of {grid_desc}. Trees are added until the "
        f"validation score stops improving for {xgb_cfg['early_stopping_rounds']} rounds (up to "
        f"{xgb_cfg['n_estimators_max']}). The combination with the best validation score is kept - "
        "darker is better."
    )
    models = [
        ("simple_count", "Simple frequency (Poisson, exposure offset)", "Poisson deviance"),
        ("hurdle_classifier", "Hurdle: chance of a claim", "Log loss"),
        ("hurdle_count", "Hurdle: claims given a claim", "Poisson deviance"),
        ("severity", "Severity (Gamma)", "Gamma deviance"),
    ]
    cols = st.columns(2)
    for i, (key, title, score) in enumerate(models):
        grid = data.selection_table("xgb", f"grid_{key}")
        if grid is None:
            continue
        with cols[i % 2], st.container(border=True):
            best = spec["tuned_params"][key]
            st.markdown(f"**{title}**  \nBest: " + ", ".join(f"{k} = {v}" for k, v in best.items())
                        + f" ({spec['n_trees'][key]} trees)")
            st.altair_chart(charts.grid_heatmap(grid, score), width="stretch")

    st.header("2. Simple vs hurdle frequency")
    freq = pd.DataFrame(
        [{"Model": data.FAMILY_NAMES[k], "Poisson deviance": v["poisson_deviance"], "Gini": v["gini"]}
         for k, v in metrics["frequency"].items()]
    )
    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(freq.style.format({"Poisson deviance": "{:.5f}", "Gini": "{:.4f}"}),
                     hide_index=True, width="stretch")
        st.markdown(
            "- **Simple**: one Poisson model on claim counts, with log(exposure) as an offset.\n"
            "- **Hurdle**: chance of at least one claim × expected claims given a claim. Exposure is a "
            "feature (not an offset) and can only increase the prediction.\n\n"
            "Deviance measures accuracy and Gini measures ranking; they can disagree."
        )
    with col2:
        plot = data.selection_file("xgb", "hurdle_calibration_test.png")
        if plot:
            st.image(str(plot), caption="Hurdle classifier: predicted vs observed chance of a claim (test set)")

    st.header("3. Frequency × severity pair")
    pair_section("xgb", spec["frequency_type"], "xgb")

    with st.expander("Selected specification"):
        st.json(spec)
