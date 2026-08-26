"""
Supabase client initialization.

Identity now comes from Auth0 (via st.login()/st.user), not Supabase's own
auth. Supabase is storage-only. For RLS policies (which check
auth.jwt() ->> 'sub') to see the logged-in user, every request must carry
Auth0's id_token -- this file's job is to attach it.

This is the ONLY file that should read raw Supabase credentials from
st.secrets. Every other file that needs database access should import
get_client() from here.
"""

import streamlit as st
from supabase import create_client, Client


def get_client() -> Client:
    """
    Returns a Supabase client with the current user's Auth0 token attached.

    NOT cached with @st.cache_resource: the token is specific to whichever
    user is logged in during THIS script run, and cache_resource would share
    one cached client (and its token) across all users' sessions, which
    would break privacy. Creating a lightweight client per call is cheap
    enough that this isn't a real performance concern.
    """
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_ANON_KEY"]
    client = create_client(url, key)

    if st.user.is_logged_in:
        # Attaches the Auth0 JWT so Postgres can verify it and RLS policies
        # checking auth.jwt() ->> 'sub' see the right user.
        client.postgrest.auth(st.user.id_token)

    return client