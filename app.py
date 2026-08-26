import streamlit as st

st.set_page_config(page_title="metabolize", page_icon="🧪")

if not st.user.is_logged_in:
    st.title("metabolize")
    st.caption("Sign in to start or resume a campaign.")
    st.button("Log in", on_click=st.login, args=("auth0",))
    st.stop()

st.sidebar.write(f"Logged in as **{st.user.email}**")
if st.sidebar.button("Log out"):
    st.logout()

# From here on: st.user.sub / st.user.email are available, and
# db.client.get_client() automatically attaches st.user.id_token to every
# Supabase request. This is where routing to campaign_picker / setup /
# round_flow will go next.
from db.client import get_client
from db.queries import get_campaign
from ui.campaign_picker import campaign_picker
from ui.setup_simple import setup_simple

if "active_campaign_id" not in st.session_state:
    campaign_picker()
else:
    client = get_client()
    campaign = get_campaign(client, st.session_state["active_campaign_id"])
    view = st.session_state.get("active_campaign_view")
    if view == "setup":
        setup_simple(campaign)
    else:
        st.write(f"View '{view}' not built yet.")
        if st.button("Back to campaigns"):
            del st.session_state["active_campaign_id"]
            st.rerun()