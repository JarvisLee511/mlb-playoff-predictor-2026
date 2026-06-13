"""Starting batting lineups per game + per-batter game logs.

Powers the lineup-strength feature. Official lineups post only ~2-3 hours
before first pitch, so the morning pipeline uses whatever is available and the
feature degrades gracefully (missing -> 0 diff); the value of knowing the
actual lineup is meant to be captured by a separate pre-game prediction pass.

Like probables.csv / pitcher_logs.csv, history is committed (data/lineups.csv,
data/batter_logs.csv) and the daily CI job refetches only the current season.
"""
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from src.config import CURRENT_SEASON, FIRST_SEASON, ROOT
from src.data.fetch import SCHEDULE_URL, _get_json

FETCH_WORKERS = 6  # modest concurrency; _get_json already retries/backs off on 503

LINEUPS_CSV = ROOT / "data" / "lineups.csv"
BATTER_LOGS_CSV = ROOT / "data" / "batter_logs.csv"
PEOPLE_STATS_URL = "https://statsapi.mlb.com/api/v1/people/{pid}/stats"

HIT_COLS = ["atBats", "hits", "doubles", "triples", "homeRuns", "baseOnBalls",
            "hitByPitch", "sacFlies", "plateAppearances", "intentionalWalks"]


def fetch_season_lineups(season: int) -> pd.DataFrame:
    """One row per (game, side, batting slot) with the starting batter id."""
    rows = []
    halves = (
        (f"{season}-03-01", f"{season}-06-30"),
        (f"{season}-07-01", f"{season}-11-30"),
    )
    for start, end in halves:
        data = _get_json(
            SCHEDULE_URL,
            {"sportId": 1, "gameType": "R", "startDate": start, "endDate": end,
             "hydrate": "lineups"},
        )
        for day in data.get("dates", []):
            for g in day["games"]:
                lu = g.get("lineups", {})
                for side, key in (("home", "homePlayers"), ("away", "awayPlayers")):
                    for order, p in enumerate(lu.get(key, []) or []):
                        pid = p.get("id")
                        if pid is not None:
                            rows.append({
                                "season": season,
                                "game_id": g["gamePk"],
                                "side": side,
                                "batting_order": order + 1,
                                "player_id": pid,
                            })
    return pd.DataFrame(rows)


def fetch_batter_log(pid: int, season: int) -> list[dict]:
    data = _get_json(
        PEOPLE_STATS_URL.format(pid=pid),
        {"stats": "gameLog", "group": "hitting", "season": season},
    )
    stats = data.get("stats", [])
    splits = stats[0].get("splits", []) if stats else []
    rows = []
    for sp in splits:
        st = sp["stat"]
        rows.append({
            "season": season,
            "player_id": pid,
            "team_id": sp.get("team", {}).get("id"),
            "date": sp.get("date", ""),
            "game_id": sp.get("game", {}).get("gamePk"),
            **{c: st.get(c, 0) for c in HIT_COLS},
        })
    return rows


def fetch_season(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    lineups = fetch_season_lineups(season)
    pids = sorted(lineups["player_id"].dropna().astype(int).unique()) if len(lineups) else []
    print(f"{season}: {lineups['game_id'].nunique() if len(lineups) else 0} games with lineups, "
          f"{len(pids)} distinct starters", flush=True)
    log_rows = []
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        for i, rows in enumerate(pool.map(lambda p: fetch_batter_log(p, season), pids)):
            log_rows.extend(rows)
            if (i + 1) % 100 == 0:
                print(f"  {season}: {i + 1}/{len(pids)} batters", flush=True)
    return lineups, pd.DataFrame(log_rows)


def _splice(csv_path, season: int, new: pd.DataFrame) -> None:
    """Write `new` for `season` into csv_path, replacing any existing rows for
    that season. Incremental so a long backfill is resumable."""
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        old = old[old["season"] != season]
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    combined.to_csv(csv_path, index=False)


def build_all(seasons=None) -> None:
    """Full history build, written season-by-season so it can resume."""
    seasons = seasons or range(FIRST_SEASON, CURRENT_SEASON + 1)
    for season in seasons:
        lineups, logs = fetch_season(season)
        _splice(LINEUPS_CSV, season, lineups)
        _splice(BATTER_LOGS_CSV, season, logs)
        print(f"  {season} written ({len(lineups)} lineup rows, {len(logs)} batter-log rows)")


def update_current() -> None:
    if not (LINEUPS_CSV.exists() and BATTER_LOGS_CSV.exists()):
        build_all()
        return
    print(f"Refreshing {CURRENT_SEASON} lineups + batter logs...")
    lineups, logs = fetch_season(CURRENT_SEASON)
    _splice(LINEUPS_CSV, CURRENT_SEASON, lineups)
    _splice(BATTER_LOGS_CSV, CURRENT_SEASON, logs)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(LINEUPS_CSV), pd.read_csv(BATTER_LOGS_CSV)


if __name__ == "__main__":
    build_all()
