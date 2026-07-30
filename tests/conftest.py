"""Shared fixtures.

Every fixture here is synthetic and hand-computable. The point of these tests is
to pin down the invariants the pipeline relies on — chronology, no leakage,
probabilities that sum to one — not to re-check numbers against a stored file.
"""
import numpy as np
import pandas as pd
import pytest

HITTING = ["h_atBats", "h_hits", "h_doubles", "h_triples", "h_homeRuns",
           "h_baseOnBalls", "h_hitByPitch", "h_sacFlies", "h_strikeOuts",
           "h_plateAppearances", "h_runs"]
PITCHING = ["p_ip", "p_earnedRuns", "p_runs", "p_hits", "p_baseOnBalls",
            "p_strikeOuts", "p_homeRuns", "p_hitByPitch"]


def make_games(results, season=2025, start="2025-04-01"):
    """`results` is a list of (home_id, away_id, home_score, away_score)."""
    dates = pd.date_range(start, periods=len(results), freq="D")
    rows = []
    for i, ((h, a, hs, as_), d) in enumerate(zip(results, dates)):
        rows.append({
            "season": season, "game_id": 100_000 + i, "date": d,
            "home_id": h, "away_id": a,
            "home_score": hs, "away_score": as_,
            "home_win": int(hs > as_),
        })
    return pd.DataFrame(rows)


def make_gamelog_rows(team_id, per_game, season=2025, start="2025-04-01", game_id0=200_000):
    """`per_game` is a list of dicts overriding any of the counting columns."""
    dates = pd.date_range(start, periods=len(per_game), freq="D")
    rows = []
    for i, (over, d) in enumerate(zip(per_game, dates)):
        row = {"season": season, "team_id": team_id, "game_id": game_id0 + i, "date": d}
        # a bland league-average-ish baseline so rate stats are finite
        row.update({
            "h_atBats": 34, "h_hits": 9, "h_doubles": 2, "h_triples": 0, "h_homeRuns": 1,
            "h_baseOnBalls": 3, "h_hitByPitch": 0, "h_sacFlies": 0, "h_strikeOuts": 8,
            "h_plateAppearances": 38, "h_runs": 4,
            "p_ip": 9.0, "p_earnedRuns": 4, "p_runs": 4, "p_hits": 8,
            "p_baseOnBalls": 3, "p_strikeOuts": 9, "p_homeRuns": 1, "p_hitByPitch": 0,
        })
        row.update(over)
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def two_team_season():
    """20 games between two teams; team 1 wins every one by a single run."""
    return make_games([(1, 2, 3, 2) if i % 2 == 0 else (2, 1, 2, 3)
                       for i in range(20)])


@pytest.fixture
def flat_gamelogs():
    """One team, 25 identical games — every pre-game rate stat is constant."""
    return make_gamelog_rows(1, [{} for _ in range(25)])


@pytest.fixture
def teams_frame():
    """Two leagues x three divisions x five teams, ids 1..30."""
    rows = []
    tid = 1
    for league in ("American League", "National League"):
        for div in ("East", "Central", "West"):
            for _ in range(5):
                rows.append({"team_id": tid, "league": league,
                             "division": f"{league} {div}"})
                tid += 1
    return pd.DataFrame(rows)


@pytest.fixture
def rng():
    return np.random.default_rng(511)
