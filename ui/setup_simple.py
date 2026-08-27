"""
Campaign setup form, shared between simple and advanced mode.

Shown for a campaign with status='setup'. Collects batch size, number of
rounds, init settings, target name, and a dynamically-built parameter list,
then saves it all into campaigns.config, flips status to 'active', and
routes into the round flow (which handles actually generating the init set).

Simple mode (setup_simple): encoding is auto-picked (OHE for categorical,
none needed for continuous) -- matches core/baybe_integration.py's
_auto_encoding(). Advanced mode (ui/setup_advanced.py) reuses the same
_render_setup_form() with advanced=True, which additionally exposes a
per-categorical-parameter encoding choice and local-run instructions.
"""

import streamlit as st

from db.client import get_client
from db.queries import update_campaign_config, update_campaign_status


def _default_param() -> dict:
    return {"name": "", "type": "categorical", "values_text": "", "min": 0.0, "max": 1.0,
            "use_interval": False, "interval": 1.0, "encoding": "OHE"}


def _params_key(campaign_id: str) -> str:
    return f"_setup_params_{campaign_id}"


def _build_config(campaign_id: str, batch_size, n_rounds, init_size, init_type, target_name) -> dict | None:
    """Validate and convert the in-progress form state into the
    config['parameters'] shape core/baybe_integration.py expects. Returns
    None (and shows errors) if anything is invalid."""
    raw_params = st.session_state[_params_key(campaign_id)]
    if not raw_params:
        st.error("Add at least one parameter.")
        return None
    if not target_name.strip():
        st.error("Enter a target name.")
        return None

    seen_names = set()
    parameters = []
    for i, p in enumerate(raw_params):
        name = p["name"].strip()
        if not name:
            st.error(f"Parameter {i + 1}: name is required.")
            return None
        if name in seen_names:
            st.error(f"Parameter name '{name}' is used more than once.")
            return None
        seen_names.add(name)

        if p["type"] == "categorical":
            values = [v.strip() for v in p["values_text"].split(",") if v.strip()]
            if len(values) < 2:
                st.error(f"Parameter '{name}': enter at least 2 comma-separated values.")
                return None
            parameters.append({
                "name": name, "type": "categorical", "values": values,
                "encoding": p.get("encoding", "OHE"),
            })
        else:
            if p["min"] >= p["max"]:
                st.error(f"Parameter '{name}': min must be less than max.")
                return None
            entry = {"name": name, "type": "continuous", "min": float(p["min"]), "max": float(p["max"])}
            if p["use_interval"]:
                if p["interval"] <= 0:
                    st.error(f"Parameter '{name}': interval must be > 0.")
                    return None
                entry["interval"] = float(p["interval"])
            parameters.append(entry)

    return {
        "parameters": parameters,
        "target_name": target_name.strip(),
        "batch_size": int(batch_size),
        "n_rounds": int(n_rounds),
        "init_type": init_type,
        "init_size": int(init_size),
    }


def _build_partial_config(campaign_id: str, batch_size, n_rounds, init_size, init_type, target_name) -> dict:
    """Like _build_config, but silent (no st.error calls) and lenient --
    used to save in-progress setup when the user navigates away before
    finishing, so returning later doesn't lose their work. Skips parameters
    that aren't filled in enough to be meaningful yet, rather than blocking
    the save entirely over one incomplete row."""
    raw_params = st.session_state[_params_key(campaign_id)]
    parameters = []
    seen_names = set()
    for p in raw_params:
        name = p["name"].strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        if p["type"] == "categorical":
            values = [v.strip() for v in p["values_text"].split(",") if v.strip()]
            if len(values) < 2:
                continue
            parameters.append({
                "name": name, "type": "categorical", "values": values,
                "encoding": p.get("encoding", "OHE"),
            })
        else:
            if p["min"] >= p["max"]:
                continue
            entry = {"name": name, "type": "continuous", "min": float(p["min"]), "max": float(p["max"])}
            if p["use_interval"] and p["interval"] > 0:
                entry["interval"] = float(p["interval"])
            parameters.append(entry)

    return {
        "parameters": parameters,
        "target_name": target_name.strip(),
        "batch_size": int(batch_size),
        "n_rounds": int(n_rounds),
        "init_type": init_type,
        "init_size": int(init_size),
    }


def _standalone_script_section(campaign: dict, batch_size, n_rounds, init_size, init_type, target_name) -> None:
    campaign_id = campaign["id"]
    with st.expander("Download a standalone script to run this campaign locally"):
        st.caption(
            "A Python script you can run on your own IDE. "
            "It generates each round's recommendations, saves them to a CSV, and pauses for you to "
            "fill in your results. "
            "You will need core/baybe_integration.py and its dependencies alongside it."
        )
        config = _build_config(campaign_id, batch_size, n_rounds, init_size, init_type, target_name)
        if config is None:
            st.info("Fix the issue(s) above to generate the script.")
        else:
            from core.generate_runner_script import generate_standalone_script
            script_text = generate_standalone_script(config, campaign["name"])
            st.download_button(
                "⬇ Download runner script (.py)",
                data=script_text,
                file_name=f"{campaign['name'].lower().replace(' ', '_')}_runner.py",
                mime="text/x-python",
            )


def _render_setup_form(campaign: dict, advanced: bool) -> None:
    campaign_id = campaign["id"]
    st.title(campaign["name"])
    st.caption("Advanced mode: full control over encodings." if advanced else
               "Simple mode: set up your campaign below.")

    saved_config = campaign.get("config") or {}

    if _params_key(campaign_id) not in st.session_state:
        saved_params = saved_config.get("parameters")
        if saved_params:
            # Reconstruct the form's working representation from a
            # previously-saved (possibly partial) config, so returning to
            # this page after a real session loss doesn't lose progress.
            restored = []
            for p in saved_params:
                if p["type"] == "categorical":
                    restored.append({
                        "name": p["name"], "type": "categorical",
                        "values_text": ", ".join(p.get("values", [])),
                        "min": 0.0, "max": 1.0, "use_interval": False, "interval": 1.0,
                        "encoding": p.get("encoding", "OHE"),
                    })
                else:
                    restored.append({
                        "name": p["name"], "type": "continuous",
                        "values_text": "",
                        "min": p.get("min", 0.0), "max": p.get("max", 1.0),
                        "use_interval": "interval" in p, "interval": p.get("interval", 1.0),
                        "encoding": "OHE",
                    })
            st.session_state[_params_key(campaign_id)] = restored
        else:
            st.session_state[_params_key(campaign_id)] = [_default_param()]

    # ---- batch / round settings ----
    st.subheader("Campaign settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        batch_size = st.number_input("Batch size (per round)", min_value=1,
                                      value=saved_config.get("batch_size", 6), step=1)
    with col2:
        n_rounds = st.number_input("Number of rounds (excluding init)", min_value=1,
                                    value=saved_config.get("n_rounds", 10), step=1)
    with col3:
        init_size = st.number_input("Init batch size", min_value=1,
                                     value=saved_config.get("init_size", 8), step=1)

    init_options = ["Sobol (generate for me)", "Upload my own (CSV)"]
    init_default_idx = 1 if saved_config.get("init_type") == "upload" else 0
    init_type = st.radio("Initial batch", init_options, index=init_default_idx, horizontal=True)
    init_type = "sobol" if init_type.startswith("Sobol") else "upload"

    target_name = st.text_input("Target name", value=saved_config.get("target_name", ""),
                                 placeholder="e.g. yield")

    st.divider()

    # ---- parameters ----
    st.subheader("Parameters")
    params = st.session_state[_params_key(campaign_id)]

    for i, p in enumerate(params):
        with st.container(border=True):
            top1, top2, top3 = st.columns([3, 2, 1])
            with top1:
                p["name"] = st.text_input("Parameter name", value=p["name"], key=f"name_{campaign_id}_{i}",
                                           placeholder="e.g. ligand")
            with top2:
                p["type"] = st.selectbox("Type", ["categorical", "continuous"],
                                          index=0 if p["type"] == "categorical" else 1,
                                          key=f"type_{campaign_id}_{i}")
            with top3:
                st.write("")
                st.write("")
                if len(params) > 1 and st.button("Remove", key=f"remove_{campaign_id}_{i}"):
                    params.pop(i)
                    st.rerun()

            if p["type"] == "categorical":
                p["values_text"] = st.text_input(
                    "Possible values (comma-separated)",
                    value=p["values_text"],
                    key=f"values_{campaign_id}_{i}",
                    placeholder="e.g. XPhos, SPhos, dppf",
                )
                if advanced:
                    p["encoding"] = st.selectbox(
                        "Encoding",
                        ["OHE", "SMILES"],
                        index=0 if p.get("encoding", "OHE") == "OHE" else 1,
                        key=f"encoding_{campaign_id}_{i}",
                        help="OHE: one-hot (best for named categories like ligand/solvent names). "
                             "SMILES: substance/structure-based encoding (best if the values are "
                             "actual SMILES strings). Note: encoding currently affects BayBE's "
                             "search-space representation; the GP surrogate used for recommendations "
                             "always fits on a plain one-hot encoding regardless of this setting.",
                    )
                else:
                    st.caption("Encoded automatically as one-hot (OHE).")
            else:
                c1, c2, c3 = st.columns(3)
                with c1:
                    p["min"] = st.number_input("Min", value=p["min"], key=f"min_{campaign_id}_{i}")
                with c2:
                    p["max"] = st.number_input("Max", value=p["max"], key=f"max_{campaign_id}_{i}")
                with c3:
                    p["use_interval"] = st.checkbox(
                        "Discrete steps", value=p["use_interval"], key=f"useint_{campaign_id}_{i}",
                        help="Check this if only specific numbers within the range are valid "
                             "(e.g. 1, 2, 3). Leave unchecked for any decimal value in the range.",
                    )
                if p["use_interval"]:
                    p["interval"] = st.number_input(
                        "Interval", value=p["interval"], min_value=0.0001, key=f"interval_{campaign_id}_{i}",
                        help="e.g. min=1, max=3, interval=1 gives allowed values 1, 2, 3.",
                    )

    if st.button("+ Add parameter"):
        params.append(_default_param())
        st.rerun()

    st.divider()

    if advanced:
        _standalone_script_section(campaign, batch_size, n_rounds, init_size, init_type, target_name)
        st.divider()

    if st.button("Save and start campaign", type="primary"):
        config = _build_config(campaign_id, batch_size, n_rounds, init_size, init_type, target_name)
        if config is not None:
            client = get_client()
            update_campaign_config(client, campaign_id, config)
            update_campaign_status(client, campaign_id, "active")
            st.session_state.pop(_params_key(campaign_id), None)
            st.session_state["active_campaign_view"] = "round_flow"
            st.rerun()

    if st.button("← Save and back to campaigns"):
        partial_config = _build_partial_config(campaign_id, batch_size, n_rounds, init_size, init_type, target_name)
        client = get_client()
        update_campaign_config(client, campaign_id, partial_config)
        st.session_state.pop(_params_key(campaign_id), None)
        st.session_state.pop("active_campaign_id", None)
        st.rerun()


def setup_simple(campaign: dict) -> None:
    _render_setup_form(campaign, advanced=False)