"""
The core campaign loop: generate/upload init, ingest results, generate the
next round's recommendations via the meta blender, repeat until the
declared number of rounds is reached, then offer to extend or finish.

Round numbering: round 0 = init batch. Rounds 1..config['n_rounds'] are the
"specified" rounds the user asked for at setup.
"""

import pandas as pd
import streamlit as st

from db.client import get_client
from db.queries import (
    get_rounds, get_data_points, create_round, mark_round_ingested,
    insert_data_points, update_target_value, update_campaign_config,
    update_campaign_status,
)
from core.baybe_integration import generate_sobol_init, validate_uploaded_init, recommend_next_batch
from ui.stats import render_stats


def _param_columns(config: dict) -> list[str]:
    return [p["name"] for p in config["parameters"]]


def _exit_button() -> None:
    st.divider()
    if st.button("← Save and exit to campaigns", width="stretch"):
        st.session_state.pop("active_campaign_id", None)
        st.rerun()


def _init_step(client, campaign: dict) -> None:
    campaign_id = campaign["id"]
    config = campaign["config"]
    st.subheader("Initial batch")

    if config["init_type"] == "sobol":
        seed = st.number_input("Sobol seed", min_value=0, value=67, step=1)
        if st.button("Generate initial batch", type="primary"):
            df = generate_sobol_init(config, seed=int(seed))
            rows = [{"param_values": row.to_dict(), "target_value": None} for _, row in df.iterrows()]
            insert_data_points(client, campaign_id, round_number=0, rows=rows)
            create_round(client, campaign_id, round_number=0)
            st.rerun()

    else:  # upload
        uploaded = st.file_uploader(
            f"Upload your initial batch as CSV (columns: {', '.join(_param_columns(config))}, "
            f"{config['target_name']})",
            type="csv",
        )
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            problems = validate_uploaded_init(df.drop(columns=[config["target_name"]], errors="ignore"), config)
            if problems:
                for p in problems:
                    st.error(p)
            elif (
                config["target_name"].strip().lower() == "yield"
                and config["target_name"] in df.columns
                and df[config["target_name"]].notna().any()
                and not df[config["target_name"]].dropna().between(0, 1).all()
            ):
                st.error(f"{config['target_name']} must be between 0 and 1 for every row that has a value.")
            else:
                has_targets = config["target_name"] in df.columns and df[config["target_name"]].notna().all()
                param_cols = _param_columns(config)
                rows = [
                    {
                        "param_values": row[param_cols].to_dict(),
                        "target_value": float(row[config["target_name"]]) if has_targets else None,
                    }
                    for _, row in df.iterrows()
                ]
                if st.button("Confirm upload", type="primary"):
                    insert_data_points(client, campaign_id, round_number=0, rows=rows)
                    create_round(client, campaign_id, round_number=0)
                    if has_targets:
                        mark_round_ingested(client, campaign_id, round_number=0)
                    st.rerun()

    _exit_button()


def _ingest_step(client, campaign: dict, round_number: int) -> None:
    campaign_id = campaign["id"]
    config = campaign["config"]
    target_name = config["target_name"]

    st.subheader(f"Round {round_number}: enter results")
    data_points = get_data_points(client, campaign_id, round_number=round_number)

    rows = []
    for d in data_points:
        row = dict(d["param_values"])
        row[target_name] = d["target_value"]
        row["_id"] = d["id"]
        rows.append(row)
    df = pd.DataFrame(rows)

    is_yield = target_name.strip().lower() == "yield"

    if is_yield:
        st.caption(
            f"After running these {len(df)} experiments, enter the results in the **{target_name}** "
            f"column below. Yield must be a number between 0 and 1 (e.g. 0.75 for 75%)."
        )
    else:
        st.caption(f"After running these {len(df)} experiments, enter the results in the **{target_name}** column below.")

    target_column_config = (
        st.column_config.NumberColumn(target_name, min_value=0.0, max_value=1.0, step=0.01,
                                       help="Must be a number between 0 and 1.")
        if is_yield else
        st.column_config.NumberColumn(target_name)
    )
    edited = st.data_editor(
        df.drop(columns=["_id"]),
        disabled=_param_columns(config),
        column_config={target_name: target_column_config},
        key=f"editor_{campaign_id}_{round_number}",
        width="stretch",
    )

    st.download_button(
        "⬇ Download this round's run sheet (CSV)",
        data=edited.to_csv(index=False).encode("utf-8"),
        file_name=f"{campaign['name']}_round{round_number}_run_sheet.csv",
        mime="text/csv",
    )

    if st.button("Ingest round", type="primary"):
        if edited[target_name].isna().any():
            st.error(f"Please fill in {target_name} for every row before ingesting.")
        elif is_yield and not edited[target_name].between(0, 1).all():
            st.error(f"{target_name} must be between 0 and 1 for every row.")
        else:
            for i, d in enumerate(data_points):
                update_target_value(client, d["id"], float(edited.iloc[i][target_name]))
            mark_round_ingested(client, campaign_id, round_number=round_number)
            st.rerun()

    _exit_button()


def _json_safe(obj):
    """Recursively convert numpy/pandas types to plain JSON-serializable
    Python types. decide_blend() mixes numpy floats/bools into its output
    dict in a few places; this catches those instead of erroring on save."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items() if not isinstance(v, pd.DataFrame)}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    return obj


def _generate_next_round(client, campaign: dict, next_round_number: int) -> None:
    campaign_id = campaign["id"]
    config = campaign["config"]
    target_name = config["target_name"]

    st.subheader(f"Ready for round {next_round_number}")
    if st.button(f"Generate round {next_round_number} recommendations", type="primary"):
        # Guard against a race: a fast double-click (or Streamlit re-firing
        # during the several seconds the GP fit takes) could otherwise let
        # two calls both pass this check before either finishes writing,
        # inserting the round's data points twice. Re-check right before
        # writing, and create the (uniquely-constrained) round row FIRST --
        # if a duplicate call is already in flight, this insert fails fast
        # and we skip generating/inserting data points at all.
        existing = [r for r in get_rounds(client, campaign_id) if r["round_number"] == next_round_number]
        if existing:
            st.rerun()
            return

        all_points = [d for d in get_data_points(client, campaign_id) if d["target_value"] is not None]
        history_rows = []
        for d in all_points:
            row = dict(d["param_values"])
            row[target_name] = d["target_value"]
            row["batch"] = d["round_number"]
            history_rows.append(row)
        history_df = pd.DataFrame(history_rows)

        result = recommend_next_batch(
            config, history_df,
            batch_size=config["batch_size"],
            total_batches=config["n_rounds"],
        )
        recs = result["recommendations"]
        rows = [
            {"param_values": {p: row[p] for p in _param_columns(config)}, "target_value": None}
            for _, row in recs.iterrows()
        ]

        # decide_blend()'s output includes 'batch_summary' (a raw DataFrame)
        # and some numpy-typed numbers, neither of which are JSON-serializable
        # as-is. _json_safe() strips/converts those before saving.
        blend_for_storage = _json_safe(result["blend"])
        try:
            create_round(client, campaign_id, round_number=next_round_number, blender_output=blend_for_storage)
        except Exception:
            # Unique constraint on (campaign_id, round_number) means a
            # concurrent duplicate call already created this round -- bail
            # out without inserting data points.
            st.rerun()
            return

        insert_data_points(client, campaign_id, round_number=next_round_number, rows=rows)

        if result["actual_batch_size"] < result["requested_batch_size"]:
            st.info(
                f"Only {result['actual_batch_size']} new, unique combinations remain in your "
                f"search space (requested {result['requested_batch_size']}). Showing all that are left."
            )
        st.rerun()

    _exit_button()


def _finished_prompt(client, campaign: dict) -> None:
    campaign_id = campaign["id"]
    config = campaign["config"]
    st.success(f"You've completed all {config['n_rounds']} planned rounds!")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Add 1 more round"):
            update_campaign_config(client, campaign_id, {**config, "n_rounds": config["n_rounds"] + 1})
            st.rerun()
    with col2:
        extra = st.number_input("Add N rounds", min_value=1, value=5, step=1, label_visibility="collapsed")
        if st.button("Add rounds"):
            update_campaign_config(client, campaign_id, {**config, "n_rounds": config["n_rounds"] + int(extra)})
            st.rerun()
    with col3:
        if st.button("Proceed to statistics", type="primary"):
            update_campaign_status(client, campaign_id, "completed")
            st.session_state["active_campaign_view"] = "stats"
            st.rerun()

    _exit_button()


def _settings_viewer(campaign: dict) -> None:
    config = campaign["config"]
    st.caption(f"Mode: {campaign['mode'].capitalize()}")
    st.caption(f"Target: {config.get('target_name', '—')}")
    st.caption(f"Batch size: {config.get('batch_size', '—')}")
    st.caption(f"Rounds: {config.get('n_rounds', '—')}")
    st.caption(f"Init size: {config.get('init_size', '—')} ({config.get('init_type', '—').capitalize()})")
    st.write("**Parameters**")
    for p in config.get("parameters", []):
        if p["type"] == "categorical":
            detail = ", ".join(p.get("values", []))
        else:
            detail = f"{p.get('min')}–{p.get('max')}"
            if p.get("interval"):
                detail += f" (step {p['interval']})"
        st.caption(f"**{p['name']}** ({p['type']}): {detail}")
    st.caption("Locked for the rest of this campaign.")


def round_flow(campaign: dict) -> None:
    client = get_client()
    campaign_id = campaign["id"]
    config = campaign["config"]

    st.title(campaign["name"])

    # ---- read-only campaign settings, shown in the sidebar ----
    with st.sidebar:
        st.divider()
        st.subheader("Campaign settings")
        _settings_viewer(campaign)

    # ---- always-visible statistics panel ----
    with st.expander("Campaign statistics", expanded=False):
        render_stats(client, campaign)

    all_points = [d for d in get_data_points(client, campaign_id) if d["target_value"] is not None]
    if all_points:
        rows = []
        for d in all_points:
            row = dict(d["param_values"])
            row[config["target_name"]] = d["target_value"]
            row["round"] = d["round_number"]
            rows.append(row)
        st.download_button(
            "⬇ Download all ingested results so far (CSV)",
            data=pd.DataFrame(rows).to_csv(index=False).encode("utf-8"),
            file_name=f"{campaign['name']}_all_results.csv",
            mime="text/csv",
        )

    st.divider()

    rounds = get_rounds(client, campaign_id)

    if not rounds:
        _init_step(client, campaign)
        return

    latest = max(rounds, key=lambda r: r["round_number"])

    if not latest["ingested"]:
        _ingest_step(client, campaign, latest["round_number"])
        return

    if latest["round_number"] >= config["n_rounds"]:
        _finished_prompt(client, campaign)
    else:
        _generate_next_round(client, campaign, latest["round_number"] + 1)