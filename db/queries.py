"""
Database query functions.

Every function here takes an already-configured Supabase `client` (see
db/client.py -> get_client(), which attaches the current user's Auth0
token). RLS policies check auth.jwt() ->> 'sub' against each row's
user_id, so as long as the client carries the right token, every query
below is automatically scoped to that user's own rows -- there is no way
for these functions to accidentally fetch someone else's data.

create_campaign is the one function that needs user_id passed in explicitly
(as st.user.sub), since -- unlike Supabase's own auth -- there's no
server-side "get the current user" lookup available via the Auth0 token
alone; the caller already has it from st.user.
"""

from typing import Any, Dict, List, Optional
from supabase import Client


# --------------------------------------------------------------------------- #
# campaigns
# --------------------------------------------------------------------------- #
def create_campaign(client: Client, user_id: str, name: str, mode: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new campaign. user_id should be st.user.sub."""
    result = client.table("campaigns").insert({
        "user_id": user_id,
        "name": name,
        "mode": mode,
        "status": "setup",
        "config": config,
    }).execute()
    return result.data[0]


def get_my_campaigns(client: Client, status: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return all campaigns belonging to the signed-in user (RLS-scoped via
    the Auth0 token attached to `client`). Optionally filter by status."""
    query = client.table("campaigns").select("*").order("updated_at", desc=True)
    if status is not None:
        query = query.eq("status", status)
    result = query.execute()
    return result.data


def get_campaign(client: Client, campaign_id: str) -> Optional[Dict[str, Any]]:
    result = client.table("campaigns").select("*").eq("id", campaign_id).execute()
    return result.data[0] if result.data else None


def update_campaign_status(client: Client, campaign_id: str, status: str) -> Dict[str, Any]:
    result = client.table("campaigns").update({"status": status}).eq("id", campaign_id).execute()
    return result.data[0]


def update_campaign_config(client: Client, campaign_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    result = client.table("campaigns").update({"config": config}).eq("id", campaign_id).execute()
    return result.data[0]


# --------------------------------------------------------------------------- #
# campaign_rounds
# --------------------------------------------------------------------------- #
def create_round(client: Client, campaign_id: str, round_number: int) -> Dict[str, Any]:
    result = client.table("campaign_rounds").insert({
        "campaign_id": campaign_id,
        "round_number": round_number,
        "ingested": False,
    }).execute()
    return result.data[0]


def get_rounds(client: Client, campaign_id: str) -> List[Dict[str, Any]]:
    result = (
        client.table("campaign_rounds")
        .select("*")
        .eq("campaign_id", campaign_id)
        .order("round_number")
        .execute()
    )
    return result.data


def mark_round_ingested(client: Client, campaign_id: str, round_number: int, blender_output: Dict[str, Any]) -> Dict[str, Any]:
    result = (
        client.table("campaign_rounds")
        .update({
            "ingested": True,
            "ingested_at": "now()",
            "blender_output": blender_output,
        })
        .eq("campaign_id", campaign_id)
        .eq("round_number", round_number)
        .execute()
    )
    return result.data[0]


# --------------------------------------------------------------------------- #
# campaign_data_points
# --------------------------------------------------------------------------- #
def insert_data_points(
    client: Client, campaign_id: str, round_number: int, rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    payload = [
        {
            "campaign_id": campaign_id,
            "round_number": round_number,
            "param_values": r["param_values"],
            "target_value": r.get("target_value"),
        }
        for r in rows
    ]
    result = client.table("campaign_data_points").insert(payload).execute()
    return result.data


def get_data_points(client: Client, campaign_id: str, round_number: Optional[int] = None) -> List[Dict[str, Any]]:
    query = client.table("campaign_data_points").select("*").eq("campaign_id", campaign_id)
    if round_number is not None:
        query = query.eq("round_number", round_number)
    result = query.order("created_at").execute()
    return result.data


def update_target_value(client: Client, data_point_id: str, target_value: float) -> Dict[str, Any]:
    result = (
        client.table("campaign_data_points")
        .update({"target_value": target_value})
        .eq("id", data_point_id)
        .execute()
    )
    return result.data[0]