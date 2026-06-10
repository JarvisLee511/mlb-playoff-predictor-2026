"""Fetch MLB game results and schedules from the official MLB Stats API.

Uses the raw schedule endpoint without hydrations (the statsapi.schedule()
helper hydrates linescores/broadcasts/media, which makes full-season requests
heavy enough that the API 503s). Retries with backoff on transient errors.

Produces:
    data/raw/games.csv          - all completed regular-season games, FIRST_SEASON..CURRENT_SEASON
    data/raw/remaining_2026.csv - unplayed 2026 regular-season games (for simulation)
    data/raw/teams.csv          - team id / name / league / division for CURRENT_SEASON
"""
import time

import pandas as pd
import requests

from src.config import DATA_RAW, FIRST_SEASON, CURRENT_SEASON

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
TEAMS_URL = "https://statsapi.mlb.com/api/v1/teams"


def _get_json(url: str, params: dict, tries: int = 5) -> dict:
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout):
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)


def fetch_season_games(season: int) -> list[dict]:
    """Pull every regular-season game record for one season."""
    data = _get_json(
        SCHEDULE_URL,
        {
            "sportId": 1,
            "startDate": f"{season}-03-01",
            "endDate": f"{season}-11-30",
            "gameType": "R",
        },
    )
    games = []
    for day in data.get("dates", []):
        games.extend(day["games"])
    return games


def fetch_teams(season: int) -> pd.DataFrame:
    data = _get_json(TEAMS_URL, {"sportId": 1, "season": season})
    rows = [
        {
            "team_id": t["id"],
            "team_name": t["name"],
            "abbrev": t.get("abbreviation", ""),
            "league": t["league"]["name"],
            "division": t["division"]["name"],
        }
        for t in data["teams"]
    ]
    return pd.DataFrame(rows).sort_values("team_name").reset_index(drop=True)


def _to_row(g: dict, season: int) -> dict:
    home, away = g["teams"]["home"], g["teams"]["away"]
    return {
        "season": season,
        "game_id": g["gamePk"],
        "date": g.get("officialDate", g["gameDate"][:10]),
        "status": g["status"]["detailedState"],
        "home_id": home["team"]["id"],
        "home_name": home["team"]["name"],
        "away_id": away["team"]["id"],
        "away_name": away["team"]["name"],
        "home_score": home.get("score"),
        "away_score": away.get("score"),
    }


def main() -> None:
    completed, remaining = [], []
    for season in range(FIRST_SEASON, CURRENT_SEASON + 1):
        print(f"Fetching {season} schedule...")
        for g in fetch_season_games(season):
            row = _to_row(g, season)
            if g["status"]["abstractGameState"] == "Final" and row["status"] != "Cancelled":
                completed.append(row)
            elif season == CURRENT_SEASON:
                remaining.append(row)
        time.sleep(1)  # be polite to the API

    games = pd.DataFrame(completed)
    games = games[(games["home_score"].notna()) & (games["away_score"].notna())]
    games["home_score"] = games["home_score"].astype(int)
    games["away_score"] = games["away_score"].astype(int)
    games = games[games["home_score"] != games["away_score"]]  # drop suspended ties
    games = games.drop_duplicates(subset="game_id")
    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)
    games = games.sort_values(["date", "game_id"]).reset_index(drop=True)
    games.to_csv(DATA_RAW / "games.csv", index=False)
    print(f"Saved {len(games)} completed games -> games.csv")

    rem = pd.DataFrame(remaining).drop_duplicates(subset="game_id")
    rem = rem.sort_values(["date", "game_id"]).reset_index(drop=True)
    rem.to_csv(DATA_RAW / "remaining_2026.csv", index=False)
    print(f"Saved {len(rem)} remaining 2026 games -> remaining_2026.csv")

    teams = fetch_teams(CURRENT_SEASON)
    teams.to_csv(DATA_RAW / "teams.csv", index=False)
    print(f"Saved {len(teams)} teams -> teams.csv")


if __name__ == "__main__":
    main()
