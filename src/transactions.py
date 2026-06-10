"""Fetch recent MLB roster transactions: injuries (IL moves), call-ups,
send-downs, activations, and trades."""
import datetime as dt

import pandas as pd

from src.data.fetch import _get_json

TRANS_URL = "https://statsapi.mlb.com/api/v1/transactions"

CATEGORY_RULES = [
    ("injured list", "IL (injury)"),
    ("recalled", "Call-up"),
    ("selected the contract", "Call-up"),
    ("activated", "Activated"),
    ("optioned", "Sent down"),
    ("traded", "Trade"),
    ("claimed", "Waiver claim"),
]


def _categorize(description: str) -> str | None:
    desc = (description or "").lower()
    for needle, category in CATEGORY_RULES:
        if needle in desc:
            return category
    return None


def fetch_recent_transactions(days: int = 10) -> pd.DataFrame:
    end = dt.date.today()
    start = end - dt.timedelta(days=days)
    data = _get_json(
        TRANS_URL, {"startDate": start.isoformat(), "endDate": end.isoformat()}
    )

    rows = []
    for t in data.get("transactions", []):
        category = _categorize(t.get("description", ""))
        if category is None:
            continue
        team = t.get("toTeam", {}).get("name") or t.get("fromTeam", {}).get("name", "")
        rows.append(
            {
                "date": t.get("date", ""),
                "team": team,
                "player": t.get("person", {}).get("fullName", ""),
                "category": category,
                "description": t.get("description", ""),
            }
        )
    df = pd.DataFrame(rows, columns=["date", "team", "player", "category", "description"])
    return df.sort_values("date", ascending=False).reset_index(drop=True)
