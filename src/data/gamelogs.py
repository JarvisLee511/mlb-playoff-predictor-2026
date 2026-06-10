"""Per-team, per-game hitting and pitching logs from the MLB Stats API.

These power the advanced season-to-date features (OPS, ERA, FIP, WHIP,
Pythagorean win%). Historical seasons never change, so the combined table is
committed at data/gamelogs.csv; the daily CI job refetches only the current
season (60 light requests) and splices it in.
"""
import time

import pandas as pd

from src.config import CURRENT_SEASON, DATA_RAW, FIRST_SEASON, ROOT
from src.data.fetch import _get_json

GAMELOGS_CSV = ROOT / "data" / "gamelogs.csv"
TEAM_STATS_URL = "https://statsapi.mlb.com/api/v1/teams/{team_id}/stats"

HIT_COLS = ["atBats", "hits", "doubles", "triples", "homeRuns", "baseOnBalls",
            "hitByPitch", "sacFlies", "strikeOuts", "plateAppearances", "runs"]
PIT_COLS = ["earnedRuns", "runs", "hits", "baseOnBalls", "strikeOuts",
            "homeRuns", "hitByPitch"]


def _innings_to_float(ip: str) -> float:
    """'5.2' means 5 innings + 2 outs, not 5.2 innings."""
    whole, _, outs = str(ip).partition(".")
    return int(whole) + int(outs or 0) / 3


def _fetch_group(team_id: int, season: int, group: str) -> list[dict]:
    data = _get_json(
        TEAM_STATS_URL.format(team_id=team_id),
        {"stats": "gameLog", "group": group, "season": season},
    )
    stats = data.get("stats", [])
    return stats[0].get("splits", []) if stats else []


def fetch_season_gamelogs(season: int, team_ids: list[int]) -> pd.DataFrame:
    rows = {}
    for team_id in team_ids:
        for split in _fetch_group(team_id, season, "hitting"):
            gid = split.get("game", {}).get("gamePk")
            if gid is None:
                continue
            row = rows.setdefault((team_id, gid), {
                "season": season, "team_id": team_id, "game_id": gid,
                "date": split.get("date", ""),
            })
            for c in HIT_COLS:
                row[f"h_{c}"] = split["stat"].get(c, 0)
        for split in _fetch_group(team_id, season, "pitching"):
            gid = split.get("game", {}).get("gamePk")
            if gid is None or (team_id, gid) not in rows:
                continue
            row = rows[(team_id, gid)]
            row["p_ip"] = _innings_to_float(split["stat"].get("inningsPitched", "0.0"))
            for c in PIT_COLS:
                row[f"p_{c}"] = split["stat"].get(c, 0)
        time.sleep(0.2)
    df = pd.DataFrame(rows.values())
    if not df.empty:
        df = df.dropna(subset=["p_ip"]).sort_values(["team_id", "date", "game_id"])
    return df.reset_index(drop=True)


def _team_ids() -> list[int]:
    return pd.read_csv(DATA_RAW / "teams.csv")["team_id"].tolist()


def build_all() -> pd.DataFrame:
    """One-time full history build (FIRST_SEASON..CURRENT_SEASON)."""
    frames = []
    for season in range(FIRST_SEASON, CURRENT_SEASON + 1):
        print(f"Fetching {season} team game logs...")
        frames.append(fetch_season_gamelogs(season, _team_ids()))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(GAMELOGS_CSV, index=False)
    print(f"Saved {len(df)} team-game log rows -> {GAMELOGS_CSV.name}")
    return df


def update_current() -> pd.DataFrame:
    """Daily refresh: refetch only the current season, splice into the table."""
    if not GAMELOGS_CSV.exists():
        return build_all()
    hist = pd.read_csv(GAMELOGS_CSV)
    hist = hist[hist["season"] < CURRENT_SEASON]
    print(f"Refreshing {CURRENT_SEASON} team game logs...")
    cur = fetch_season_gamelogs(CURRENT_SEASON, _team_ids())
    df = pd.concat([hist, cur], ignore_index=True)
    df.to_csv(GAMELOGS_CSV, index=False)
    print(f"Game logs: {len(hist)} historical + {len(cur)} current season rows")
    return df


def load() -> pd.DataFrame:
    return pd.read_csv(GAMELOGS_CSV)
