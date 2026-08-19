"""
THROWAWAY test script -- not part of the real app.

Proves two things:
  1. queries.py works against the live schema.
  2. RLS actually enforces privacy: a second user cannot see the first
     user's campaigns.

Before running:
  - Create two users in Supabase Dashboard -> Authentication -> Users:
      test-a@example.com / (password)
      test-b@example.com / (password)
  - Fill in the emails/passwords below to match.

Run with:  python test_queries.py
(plain python is fine here -- this doesn't need to run inside Streamlit)
"""

import streamlit as st
from supabase import create_client

from db.queries import create_campaign, get_my_campaigns

# --- fill these in to match the users you created in Supabase ---
USER_A_EMAIL = "test-a@example.com"
USER_A_PASSWORD = "test"
USER_B_EMAIL = "test-b@example.com"
USER_B_PASSWORD = "test"


def _new_client():
    """A fresh, unauthenticated client -- separate from db.client.get_client()
    since that one is cached and we need independent sessions here."""
    # Reads secrets.toml directly since this script runs outside a normal
    # Streamlit app context (no cache_resource needed for a one-off script).
    import toml
    secrets = toml.load(".streamlit/secrets.toml")
    return create_client(secrets["SUPABASE_URL"], secrets["SUPABASE_ANON_KEY"])


def main():
    # --- sign in as user A, create a campaign ---
    client_a = _new_client()
    client_a.auth.sign_in_with_password({"email": USER_A_EMAIL, "password": USER_A_PASSWORD})

    campaign = create_campaign(
        client_a,
        name="Test campaign A",
        mode="simple",
        config={"batch_size": 6, "n_rounds": 5},
    )
    print(f"[A] created campaign: {campaign['id']}")

    campaigns_a_sees = get_my_campaigns(client_a)
    print(f"[A] sees {len(campaigns_a_sees)} campaign(s) -- expected: 1")
    assert len(campaigns_a_sees) >= 1, "User A should see their own campaign"

    # --- sign in as user B, confirm they see NONE of user A's campaigns ---
    client_b = _new_client()
    client_b.auth.sign_in_with_password({"email": USER_B_EMAIL, "password": USER_B_PASSWORD})

    campaigns_b_sees = get_my_campaigns(client_b)
    print(f"[B] sees {len(campaigns_b_sees)} campaign(s) -- expected: 0")
    assert len(campaigns_b_sees) == 0, "PRIVACY BUG: User B can see User A's campaigns!"

    print("\nPASSED: RLS is correctly isolating campaigns per user.")


if __name__ == "__main__":
    main()