import streamlit as st
import base64

from db.client import get_client
from db.queries import get_campaign
from ui.campaign_picker import campaign_picker
from ui.setup_simple import setup_simple
from ui.setup_advanced import setup_advanced
from ui.round_flow import round_flow
from ui.stats import stats_page

st.set_page_config(page_title="metabolize", page_icon="🧪")

st.markdown(
    """
    <style>
    h1, h2, h3 { color: #0C2245; }
    section[data-testid="stSidebar"] {
        background-color: #0C2245;
    }
    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
        color: #FFFFFF !important;
        opacity: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if not st.user.is_logged_in:
    st.markdown(
    """
    <style>
    div[data-testid="stImage"] {
        display: flex;
        justify-content: center;
        margin-bottom: -30px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        icol1, icol2, icol3 = st.columns([1, 1, 1])
        with icol2:
            st.image("assets/logo.png", width=80)
        st.markdown(
            "<h1 style='text-align: center;'>metabolize</h1>"
            "<p style='text-align: center; color: gray;'>Sign in to start or resume a campaign.</p>",
            unsafe_allow_html=True,
        )
    
        st.button("Log in", on_click=st.login, args=("auth0",), width="stretch", type="primary")
    st.stop()

logo_col, spacer_col, user_col = st.columns([1, 1.5, 2])

with logo_col:
    st.image("assets/logo.png", width=67)

with user_col:
    st.markdown(
        f"<p style='text-align: right; margin-bottom: 0; white-space: nowrap;'>"
        f"Logged in as <b>{st.user.email}</b></p>",
        unsafe_allow_html=True,
    )
    spacer, button_col = st.columns([2, 1])
    with button_col:
        if st.button("Log out", key="logout_btn", width="stretch", type="secondary"):
            st.logout()

# ---- routing ----
try:
    if "active_campaign_id" not in st.session_state:
        campaign_picker()
    else:
        client = get_client()
        campaign = get_campaign(client, st.session_state["active_campaign_id"])
        view = st.session_state.get("active_campaign_view")

        if campaign is None:
            # campaign was deleted or doesn't belong to this user -- bounce back
            st.session_state.pop("active_campaign_id", None)
            st.session_state.pop("active_campaign_view", None)
            st.rerun()
        elif view == "setup":
            if campaign["mode"] == "advanced":
                setup_advanced(campaign)
            else:
                setup_simple(campaign)
        elif view == "round_flow":
            round_flow(campaign)
        elif view == "stats":
            stats_page(campaign)
        else:
            st.write(f"Unknown view '{view}'.")
            if st.button("← Back to campaigns"):
                st.session_state.pop("active_campaign_id", None)
                st.rerun()

except Exception as e:
    # Auth0 sessions expire after a while (commonly ~1 hour). Without this,
    # an expired token surfaces as a raw "JWT expired" traceback from
    # Supabase -- confusing for a non-coder user. Catch it here and offer a
    # one-click way back in instead. Anything else genuinely unexpected
    # still shows the real error, so real bugs aren't hidden.
    if "JWT expired" in str(e) or "PGRST303" in str(e):
        st.warning("Your session has expired. Please log in again to continue.")
        st.button("Log in again", on_click=st.login, args=("auth0",))
    else:
        raise