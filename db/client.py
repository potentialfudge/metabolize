"""
Supabase client initialization.

This is the ONLY file that should read raw Supabase credentials from
st.secrets. Every other file that needs database access should import
get_client() from here, not touch st.secrets directly.
"""

import streamlit as st
from supabase import create_client, Client


@st.cache_resource
def get_client() -> Client:
    """
    Returns a cached Supabase client. @st.cache_resource means this only
    actually runs once per app process (not once per user) -- the same
    client object is reused across reruns and across users, which is safe
    because privacy is enforced by RLS policies keyed on each request's
    auth token, not by having separate client objects per user.
    """
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_ANON_KEY"]
    return create_client(url, key)