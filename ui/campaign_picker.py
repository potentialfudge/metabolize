"""
Campaign picker -- the screen shown right after login.

Lets the user resume an in-progress campaign, view a completed one, or
start a new one. Sets st.session_state["active_campaign_id"] and
st.session_state["active_campaign_view"] ("setup" | "round_flow" | "stats")
to tell app.py what to render next.
"""

import streamlit as st

from db.client import get_client
from db.queries import get_my_campaigns, create_campaign, delete_campaign


def _status_badge(status: str) -> str:
    return {
        "setup": "Setup",
        "active": "Active",
        "completed": "Completed",
    }.get(status, status)


def _open_campaign(campaign: dict) -> None:
    """Route into the right screen depending on campaign status."""
    st.session_state["active_campaign_id"] = campaign["id"]
    if campaign["status"] == "setup":
        st.session_state["active_campaign_view"] = "setup"
    elif campaign["status"] == "completed":
        st.session_state["active_campaign_view"] = "stats"
    else:
        st.session_state["active_campaign_view"] = "round_flow"
    st.rerun()


def campaign_picker() -> None:
    st.title("Your campaigns")

    client = get_client()
    campaigns = get_my_campaigns(client)

    in_progress = [c for c in campaigns if c["status"] in ("setup", "active")]
    completed = [c for c in campaigns if c["status"] == "completed"]

    # ---- Start a new campaign ----
    st.subheader("Start a new campaign")
    with st.form("new_campaign_form", enter_to_submit=False):
        name = st.text_input(
            "Campaign name",
            placeholder="e.g. Sulfonamide coupling optimization",
        )
        mode = st.radio("Mode", ["simple", "advanced"], horizontal=True, format_func=str.capitalize)
        submitted = st.form_submit_button("Create campaign")
    if submitted:
        if not name.strip():
            st.error("Please give your campaign a name.")
        else:
            campaign = create_campaign(client, st.user.sub, name.strip(), mode, {})
            _open_campaign(campaign)

    st.divider()

    # ---- Resume in-progress ----
    st.subheader("Resume a campaign")
    if not in_progress:
        st.caption("No campaigns in progress.")
    else:
        for c in in_progress:
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                st.write(f"**{c['name']}**")
                st.caption(f"{_status_badge(c['status'])} · {c['mode'].capitalize()} mode")
            with col2:
                st.caption(f"Updated {c['updated_at'][:10]}")
            with col3:
                if st.button("Resume", key=f"resume_{c['id']}"):
                    _open_campaign(c)
            with col4:
                if st.button("Delete", key=f"delete_{c['id']}"):
                    st.session_state["_pending_delete"] = c["id"]
                    st.rerun()

            # Confirmation step: only shows up right under the campaign the
            # user just clicked Delete on, and requires an explicit second
            # click before anything is actually removed.
            if st.session_state.get("_pending_delete") == c["id"]:
                st.warning(f"Delete **{c['name']}**? This cannot be undone — all rounds and data will be lost.")
                confirm_col, cancel_col = st.columns([1, 1])
                with confirm_col:
                    if st.button("Yes, delete permanently", key=f"confirm_delete_{c['id']}", type="primary"):
                        delete_campaign(client, c["id"])
                        st.session_state.pop("_pending_delete", None)
                        st.rerun()
                with cancel_col:
                    if st.button("Cancel", key=f"cancel_delete_{c['id']}"):
                        st.session_state.pop("_pending_delete", None)
                        st.rerun()

    st.divider()

    # ---- View completed ----
    st.subheader("Completed campaigns")
    if not completed:
        st.caption("No completed campaigns yet.")
    else:
        for c in completed:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"**{c['name']}**")
                st.caption(f"{_status_badge(c['status'])} · {c['mode'].capitalize()} mode")
            with col2:
                st.caption(f"Updated {c['updated_at'][:10]}")
            with col3:
                if st.button("View", key=f"view_{c['id']}"):
                    _open_campaign(c)