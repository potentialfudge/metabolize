"""
Statistics rendering, shared between the always-visible in-round panel
(render_stats, embedded via an expander in round_flow.py) and the full
dedicated stats page shown for completed campaigns (stats_page, below).
"""

import pandas as pd
import streamlit as st

from db.client import get_client
from db.queries import get_data_points, get_rounds


def render_stats(client, campaign: dict) -> None:
    """Renders best-so-far progress and per-round blend info. Safe to call
    with zero data (shows a friendly empty state instead of erroring)."""
    campaign_id = campaign["id"]
    target_name = campaign.get("config", {}).get("target_name", "target")

    data_points = get_data_points(client, campaign_id)
    ingested = [d for d in data_points if d["target_value"] is not None]

    if not ingested:
        st.caption("No results yet — statistics will appear after your first round is ingested.")
        return

    df = pd.DataFrame(ingested).sort_values(["round_number", "created_at"])
    df["best_so_far"] = df["target_value"].cummax()

    col1, col2, col3 = st.columns(3)
    col1.metric(f"Best {target_name} so far", f"{df['target_value'].max():.4g}")
    col2.metric("Experiments run", len(df))
    col3.metric("Rounds completed", int(df["round_number"].max()) + 1)

    per_round_best = df.groupby("round_number")["best_so_far"].max()
    st.line_chart(per_round_best, x_label="Round", y_label=f"Best {target_name} so far")

    rounds = get_rounds(client, campaign_id)
    with st.expander("Blend decisions per round"):
        for r in sorted(rounds, key=lambda r: r["round_number"]):
            if r.get("blender_output"):
                st.text(f"Round {r['round_number']}: {r['blender_output'].get('reason', '(no reason recorded)')}")


def _blend_weights_df(rounds: list) -> pd.DataFrame:
    """Pull UCB/EI/PI weights per round out of each round's stored
    blender_output, for the weight-trajectory chart. Round 0 (init) has no
    blend decision, so it's naturally skipped."""
    rows = []
    for r in sorted(rounds, key=lambda r: r["round_number"]):
        bo = r.get("blender_output")
        if bo and "weights" in bo:
            rows.append({
                "round": r["round_number"],
                "UCB": bo["weights"].get("UCB"),
                "EI": bo["weights"].get("EI"),
                "PI": bo["weights"].get("PI"),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("round")


def stats_page(campaign: dict) -> None:
    """Full dedicated statistics page, shown for status='completed' campaigns."""
    client = get_client()
    campaign_id = campaign["id"]
    config = campaign["config"]
    target_name = config.get("target_name", "target")

    st.title(f"{campaign['name']} — Results")

    if st.button("← Back to campaigns"):
        st.session_state.pop("active_campaign_id", None)
        st.rerun()

    st.divider()

    data_points = get_data_points(client, campaign_id)
    ingested = [d for d in data_points if d["target_value"] is not None]

    if not ingested:
        st.info("This campaign has no ingested results yet.")
        return

    df = pd.DataFrame(ingested).sort_values(["round_number", "created_at"])
    df["best_so_far"] = df["target_value"].cummax()
    df = df.reset_index(drop=True)
    df["trial_index"] = df.index + 1

    # ---- headline metrics ----
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Best {target_name}", f"{df['target_value'].max():.4g}")
    col2.metric("Total experiments", len(df))
    col3.metric("Rounds completed", int(df["round_number"].max()) + 1)
    col4.metric(f"Mean {target_name}", f"{df['target_value'].mean():.4g}")

    # ---- learning curve: every trial, plus best-so-far trend ----
    st.subheader("Learning curve")
    st.caption(f"Every individual experiment's {target_name}, with the running best-so-far.")
    learning_curve_df = df.set_index("trial_index")[["target_value", "best_so_far"]]
    learning_curve_df.columns = [target_name, "Best so far"]
    st.line_chart(learning_curve_df, x_label="Experiment #", y_label=target_name)

    # ---- best-so-far by round (coarser view) ----
    st.subheader("Progress over rounds")
    per_round_best = df.groupby("round_number")["best_so_far"].max()
    st.line_chart(per_round_best, x_label="Round", y_label=f"Best {target_name} so far")

    # ---- objective distribution ----
    st.subheader(f"{target_name.capitalize()} distribution")
    binned = df["target_value"].value_counts(bins=10).sort_index()
    # value_counts(bins=...) returns pandas Interval objects as the index;
    # left as-is, Streamlit/Vega renders their raw repr (e.g. left/right
    # dict) as the axis label instead of a readable range. Format explicitly.
    binned.index = [f"{interval.left:.3g}–{interval.right:.3g}" for interval in binned.index]
    st.bar_chart(binned, x_label=target_name, y_label="Count")

    # ---- blend weight trajectory ----
    rounds = get_rounds(client, campaign_id)
    weights_df = _blend_weights_df(rounds)
    if not weights_df.empty:
        st.subheader("Acquisition blend weights per round")
        st.caption("How the meta blender balanced exploration (UCB) vs. exploitation (EI/PI) each round.")
        st.line_chart(weights_df, x_label="Round", y_label="Weight")

    # ---- per-parameter analysis ----
    st.subheader("Parameter analysis")
    param_names = [p["name"] for p in config.get("parameters", [])]
    param_types = {p["name"]: p["type"] for p in config.get("parameters", [])}

    # Expand the param_values dict column (already present in df, in the
    # same sorted row order) into real columns -- avoids rebuilding a
    # separate dataframe from `ingested` directly, which would be in a
    # different (unsorted) row order and misalign if concatenated by position.
    param_expanded = pd.DataFrame(df["param_values"].tolist(), index=df.index)
    df_with_params = pd.concat([df, param_expanded], axis=1)

    for p_name in param_names:
        if p_name not in df_with_params.columns:
            continue
        with st.expander(f"{p_name} ({param_types.get(p_name, '?')})"):
            if param_types.get(p_name) == "categorical":
                summary = df_with_params.groupby(p_name)["target_value"].agg(["mean", "count"])
                summary.columns = [f"Mean {target_name}", "Count"]
                st.bar_chart(summary[f"Mean {target_name}"])
                st.dataframe(summary, width="stretch")
            else:
                st.scatter_chart(df_with_params, x=p_name, y="target_value", x_label=p_name, y_label=target_name)

    # ---- top trials ----
    st.subheader("Top results")
    top_n = min(10, len(df_with_params))
    top_df = df_with_params.nlargest(top_n, "target_value")[["trial_index", "round_number", "target_value"] + param_names]
    st.dataframe(top_df, width="stretch", hide_index=True)

    # ---- blend reasoning log ----
    with st.expander("Blend decisions per round"):
        for r in sorted(rounds, key=lambda r: r["round_number"]):
            if r.get("blender_output"):
                st.text(f"Round {r['round_number']}: {r['blender_output'].get('reason', '(no reason recorded)')}")

    # ---- full results table + download ----
    st.subheader("All results")
    display_rows = []
    for d in ingested:
        row = dict(d["param_values"])
        row[target_name] = d["target_value"]
        row["round"] = d["round_number"]
        display_rows.append(row)
    display_df = pd.DataFrame(display_rows)[["round"] + param_names + [target_name]].sort_values("round")
    st.dataframe(display_df, width="stretch", hide_index=True)

    st.download_button(
        "⬇ Download full results (CSV)",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{campaign['name']}_final_results.csv",
        mime="text/csv",
    )