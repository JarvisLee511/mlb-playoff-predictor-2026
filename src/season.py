"""Where the season stands, and what the postseason did to the pre-playoff forecast.

The daily pipeline predicts regular-season games. Once the last one is played and scored
there is nothing left for it to do, but until November 2026 it kept running anyway: three
commits a day of timestamp and transaction-list churn, with the pre-playoff World Series
odds still presented as current after the World Series had been played.

Now `update()` runs first every morning. While regular-season games remain (or a logged
prediction is still waiting for its result) it returns "regular" and the pipeline runs as
before. After that it returns "postseason" or "complete": the pipeline stops, the last
odds stay frozen as the pre-playoff forecast, and docs/data/season.json records what
actually happened next to it. The file is written only when its content changes, so a
finished season produces no commits at all.
"""
from __future__ import annotations

import json

import pandas as pd

from src.config import CURRENT_SEASON, ROOT
from src.data.fetch import SCHEDULE_URL, _get_json
from src.predictions import PRED_LOG, load_log

SITE_DATA = ROOT / "docs" / "data"
SEASON_JSON = SITE_DATA / "season.json"
ODDS_JSON = SITE_DATA / "odds.json"

ROUNDS = {"F": "Wild Card Series", "D": "Division Series",
          "L": "League Championship Series", "W": "World Series"}
ROUND_ORDER = list(ROUNDS)
LOST_IN = {"F": "Lost Wild Card Series", "D": "Lost Division Series",
           "L": "Lost LCS", "W": "Lost World Series"}


def _games(season: int, game_type: str, hydrate: str | None = None) -> list[dict]:
    params = {"sportId": 1, "season": season, "gameType": game_type}
    if hydrate:
        params["hydrate"] = hydrate
    data = _get_json(SCHEDULE_URL, params)
    return [g for day in data.get("dates", []) for g in day["games"]]


def games_left(regular: list[dict]) -> int:
    """Regular-season games not yet over. Postponed and cancelled entries count as over
    (the API marks them Final); a postponed game's make-up is its own entry."""
    return sum(g["status"]["abstractGameState"] in ("Preview", "Live") for g in regular)


def void_cancelled(log: pd.DataFrame, regular: list[dict]) -> tuple[pd.DataFrame, int]:
    """A prediction for a game that was cancelled can never be scored: mark it so instead
    of leaving it pending forever (2026-09-27 BAL @ NYY, game 823490, was the first)."""
    states: dict[int, set[str]] = {}
    for g in regular:
        states.setdefault(g["gamePk"], set()).add(g["status"]["detailedState"])
    never_played = {pk for pk, s in states.items()
                    if "Cancelled" in s and not s & {"Final", "Completed Early", "Game Over"}}
    mask = (log["status"] == "pending") & log["game_id"].isin(sorted(never_played))
    out = log.copy()
    out.loc[mask, "status"] = "cancelled"
    return out, int(mask.sum())


def series_results(post: list[dict]) -> list[dict]:
    """One row per finished postseason series, in bracket order."""
    series: dict[tuple, dict] = {}
    for g in post:
        ss = g.get("seriesStatus") or {}
        if not ss.get("isOver") or g["gameType"] not in ROUNDS:
            continue
        home, away = g["teams"]["home"]["team"], g["teams"]["away"]["team"]
        key = (g["gameType"], g.get("seriesDescription"), frozenset((home["id"], away["id"])))
        series[key] = {
            "round_code": g["gameType"],
            "round": ROUNDS[g["gameType"]],
            "winner": ss["winningTeam"]["name"],
            "loser": ss["losingTeam"]["name"],
            "result": ss.get("result") or "",
            "ended": g["officialDate"],
        }
    return sorted(series.values(),
                  key=lambda s: (ROUND_ORDER.index(s["round_code"]), s["ended"], s["winner"]))


def team_results(series: list[dict]) -> dict[str, str]:
    """How far each playoff team got: its last series decides it."""
    out: dict[str, str] = {}
    for s in series:  # bracket order, so later rounds overwrite earlier ones
        out[s["loser"]] = LOST_IN[s["round_code"]]
        out[s["winner"]] = "Won World Series" if s["round_code"] == "W" else "Advanced"
    return out


def summarise(season: int, regular: list[dict], post: list[dict], odds: list[dict]) -> dict:
    """The season.json payload. Pure: everything it needs is passed in."""
    series = series_results(post)
    results = team_results(series)
    champion = next((s["winner"] for s in series if s["round_code"] == "W"), None)
    abbrev = {r["team_name"]: r.get("abbrev") or r["team_name"] for r in odds}
    playoff = sorted((r for r in odds if r["make_playoffs"] >= 0.5),
                     key=lambda r: -r["win_world_series"])

    def forecast(team: str | None) -> dict | None:
        if team is None:
            return None
        row = next((r for r in playoff if r["team_name"] == team), None)
        if row is None:
            return {"team": team, "abbrev": abbrev.get(team, team)}
        league = [r for r in playoff if r["league"] == row["league"]]
        league.sort(key=lambda r: -r["win_pennant"])
        return {
            "team": team, "abbrev": abbrev.get(team, team), "league": row["league"],
            "p_world_series": row["win_world_series"], "p_pennant": row["win_pennant"],
            "rank_world_series": playoff.index(row) + 1,
            "rank_pennant_in_league": league.index(row) + 1,
        }

    pennants = [s["winner"] for s in series if s["round_code"] == "L"]
    regular_dates = [g["officialDate"] for g in regular]
    post_dates = [g["officialDate"] for g in post]
    return {
        "season": season,
        "status": "complete" if champion else "postseason",
        "regular_season_ended": max(regular_dates) if regular_dates else None,
        "postseason_began": min(post_dates) if post_dates else None,
        "n_playoff_teams": len(playoff),
        "favourite": forecast(playoff[0]["team_name"]) if playoff else None,
        "champion": forecast(champion),
        "pennant_winners": [forecast(t) for t in pennants],
        "results": {abbrev.get(t, t): r for t, r in sorted(results.items())},
        "series": [{k: v for k, v in s.items() if k != "round_code"} for s in series],
    }


def write_if_changed(payload: dict) -> bool:
    text = json.dumps(payload, indent=1) + "\n"
    if SEASON_JSON.exists() and SEASON_JSON.read_text(encoding="utf-8") == text:
        return False
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    SEASON_JSON.write_text(text, encoding="utf-8")
    return True


def update(season: int = CURRENT_SEASON) -> str:
    """'regular' while there is work for the daily pipeline, else 'postseason'/'complete'."""
    regular = _games(season, "R")
    log = load_log()
    log, n_void = void_cancelled(log, regular)
    if n_void:
        log.to_csv(PRED_LOG, index=False)
        print(f"Marked {n_void} prediction(s) for cancelled games as cancelled")
    if games_left(regular) or (log["status"] == "pending").any():
        write_if_changed({"season": season, "status": "regular"})
        return "regular"

    post = _games(season, "F,D,L,W", hydrate="seriesStatus")
    odds = json.loads(ODDS_JSON.read_text(encoding="utf-8")) if ODDS_JSON.exists() else []
    payload = summarise(season, regular, post, odds)
    changed = write_if_changed(payload)
    print(f"{season} regular season is over ({payload['status']}); "
          f"season.json {'updated' if changed else 'unchanged'}")
    return payload["status"]
