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
st.write("Logged in. Campaign picker goes here next.")