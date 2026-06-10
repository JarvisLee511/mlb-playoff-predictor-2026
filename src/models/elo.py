"""Elo rating system for MLB teams (FiveThirtyEight-style).

Ratings start at 1500, use a low K typical for baseball's high game count,
a margin-of-victory multiplier, and regress 1/3 toward the mean each offseason.
"""
import math

import pandas as pd

from src.config import (
    ELO_HOME_ADV,
    ELO_K,
    ELO_SEASON_CARRYOVER,
    ELO_START,
)


def elo_win_prob(home_elo: float, away_elo: float, home_adv: float = ELO_HOME_ADV) -> float:
    """Probability the home team wins."""
    diff = home_elo + home_adv - away_elo
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def run_elo(games: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, float]]:
    """Replay games chronologically, attaching pre-game Elo columns.

    Returns the augmented games frame and the final rating per team_id.
    """
    games = games.sort_values(["date", "game_id"]).reset_index(drop=True)
    ratings: dict[int, float] = {}
    last_season: dict[int, int] = {}

    home_elos, away_elos, probs = [], [], []
    for g in games.itertuples():
        for team in (g.home_id, g.away_id):
            if team not in ratings:
                ratings[team] = ELO_START
            elif last_season.get(team) != g.season:
                # offseason regression toward the mean
                ratings[team] = (
                    ELO_START * (1 - ELO_SEASON_CARRYOVER)
                    + ratings[team] * ELO_SEASON_CARRYOVER
                )
            last_season[team] = g.season

        h, a = ratings[g.home_id], ratings[g.away_id]
        p_home = elo_win_prob(h, a)
        home_elos.append(h)
        away_elos.append(a)
        probs.append(p_home)

        margin = abs(g.home_score - g.away_score)
        mov_mult = math.log(margin + 1)
        shift = ELO_K * mov_mult * (g.home_win - p_home)
        ratings[g.home_id] = h + shift
        ratings[g.away_id] = a - shift

    games["home_elo_pre"] = home_elos
    games["away_elo_pre"] = away_elos
    games["elo_prob_home"] = probs
    return games, ratings
