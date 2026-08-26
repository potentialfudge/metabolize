import streamlit as st
from ui.campaign_picker import campaign_picker

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
from db.queries import create_campaign, get_my_campaigns

client = get_client()

if "active_campaign_id" not in st.session_state:
    campaign_picker()
else:
    st.write(f"Campaign {st.session_state['active_campaign_id']} — screen not built yet.")
    if st.button("Back to campaigns"):
        del st.session_state["active_campaign_id"]
        st.rerun()