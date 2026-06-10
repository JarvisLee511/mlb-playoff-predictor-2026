"""Probable starting pitchers per game + per-pitcher game logs.

Powers the starting-pitcher and bullpen features. Like gamelogs.csv, history
is committed (data/probables.csv, data/pitcher_logs.csv) and the daily CI job
refetches only the current season.
"""
import time

import pandas as pd

from src.config import CURRENT_SEASON, FIRST_SEASON, ROOT
from src.data.fetch import SCHEDULE_URL, _get_json
from src.data.gamelogs import _innings_to_float

PROBABLES_CSV = ROOT / "data" / "probables.csv"
PITCHER_LOGS_CSV = ROOT / "data" / "pitcher_logs.csv"
PEOPLE_STATS_URL = "https://statsapi.mlb.com/api/v1/people/{pid}/stats"


def fetch_season_probables(season: int) -> pd.DataFrame:
    rows = []
    halves = (
        (f"{season}-03-01", f"{season}-06-30"),
        (f"{season}-07-01", f"{season}-11-30"),
    )
    for start, end in halves:
        data = _get_json(
            SCHEDULE_URL,
            {"sportId": 1, "gameType": "R", "startDate": start, "endDate": end,
             "hydrate": "probablePitcher"},
        )
        for day in data.get("dates", []):
            for g in day["games"]:
                rows.append(
                    {
                        "season": season,
                        "game_id": g["gamePk"],
                        "home_sp_id": (g["teams"]["home"].get("probablePitcher") or {}).get("id"),
                        "away_sp_id": (g["teams"]["away"].get("probablePitcher") or {}).get("id"),
                    }
                )
    return pd.DataFrame(rows).drop_duplicates("game_id")


def fetch_pitcher_log(pid: int, season: int) -> list[dict]:
    data = _get_json(
        PEOPLE_STATS_URL.format(pid=pid),
        {"stats": "gameLog", "group": "pitching", "season": season},
    )
    stats = data.get("stats", [])
    splits = stats[0].get("splits", []) if stats else []
    rows = []
    for sp in splits:
        st = sp["stat"]
        rows.append(
            {
                "season": season,
                "pitcher_id": pid,
                "team_id": sp.get("team", {}).get("id"),
                "date": sp.get("date", ""),
                "game_id": sp.get("game", {}).get("gamePk"),
                "gs": st.get("gamesStarted", 0),
                "ip": _innings_to_float(st.get("inningsPitched", "0.0")),
                "er": st.get("earnedRuns", 0),
                "bb": st.get("baseOnBalls", 0),
                "so": st.get("strikeOuts", 0),
                "hr": st.get("homeRuns", 0),
                "hbp": st.get("hitByPitch", 0),
                "bf": st.get("battersFaced", 0),
            }
        )
    return rows


def fetch_season(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    probables = fetch_season_probables(season)
    pids = sorted(
        set(probables["home_sp_id"].dropna().astype(int))
        | set(probables["away_sp_id"].dropna().astype(int))
    )
    print(f"{season}: {len(probables)} games, {len(pids)} starters")
    log_rows = []
    for pid in pids:
        log_rows.extend(fetch_pitcher_log(pid, season))
        time.sleep(0.1)
    return probables, pd.DataFrame(log_rows)


def build_all() -> None:
    probs, logs = [], []
    for season in range(FIRST_SEASON, CURRENT_SEASON + 1):
        p, l = fetch_season(season)
        probs.append(p)
        logs.append(l)
    pd.concat(probs, ignore_index=True).to_csv(PROBABLES_CSV, index=False)
    log_df = pd.concat(logs, ignore_index=True)
    log_df.to_csv(PITCHER_LOGS_CSV, index=False)
    print(f"Saved {len(log_df)} pitcher-log rows, {sum(len(p) for p in probs)} probables")


def update_current() -> None:
    if not (PROBABLES_CSV.exists() and PITCHER_LOGS_CSV.exists()):
        build_all()
        return
    print(f"Refreshing {CURRENT_SEASON} probables + pitcher logs...")
    p_cur, l_cur = fetch_season(CURRENT_SEASON)
    probs = pd.read_csv(PROBABLES_CSV)
    logs = pd.read_csv(PITCHER_LOGS_CSV)
    pd.concat([probs[probs["season"] < CURRENT_SEASON], p_cur], ignore_index=True).to_csv(
        PROBABLES_CSV, index=False
    )
    pd.concat([logs[logs["season"] < CURRENT_SEASON], l_cur], ignore_index=True).to_csv(
        PITCHER_LOGS_CSV, index=False
    )


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(PROBABLES_CSV), pd.read_csv(PITCHER_LOGS_CSV)
