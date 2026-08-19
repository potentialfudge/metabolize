"""
Database query functions.

Every function here takes an already-authenticated Supabase `client` as its
first argument (see db/client.py -> get_client()). Because RLS policies key
off auth.uid(), which Postgres reads from the CLIENT'S OWN SESSION -- not from
any value we pass in -- these functions never take a user_id parameter. As
long as the client is signed in as the right user, Postgres already knows who
"you" are, and every query below is automatically scoped to that user's own
rows. There is no way for these functions to accidentally fetch someone
else's data, because the database itself refuses to return it.
"""

from typing import Any, Dict, List, Optional
from supabase import Client


# --------------------------------------------------------------------------- #
# campaigns
# --------------------------------------------------------------------------- #
def create_campaign(client: Client, name: str, mode: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new campaign owned by the currently signed-in user."""
    user_id = client.auth.get_user().user.id
    result = client.table("campaigns").insert({
        "user_id": user_id,
        "name": name,
        "mode": mode,
        "status": "setup",
        "config": config,
    }).execute()
    return result.data[0]


def get_my_campaigns(client: Client, status: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return all campaigns belonging to the signed-in user (RLS-scoped).
    Optionally filter by status ('setup' | 'active' | 'completed')."""
    query = client.table("campaigns").select("*").order("updated_at", desc=True)
    if status is not None:
        query = query.eq("status", status)
    result = query.execute()
    return result.data


def get_campaign(client: Client, campaign_id: str) -> Optional[Dict[str, Any]]:
    """Return a single campaign by id, or None if it doesn't exist / isn't yours."""
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
    """Create a new (not-yet-ingested) round row."""
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
    """
    Insert a batch of experiment rows for a round. `rows` is a list of
    {"param_values": {...}, "target_value": None-or-number} dicts.
    """
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