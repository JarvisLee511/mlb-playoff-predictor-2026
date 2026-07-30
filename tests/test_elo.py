"""Elo invariants.

The Elo baseline is the yardstick every model in this project is measured
against, so a silent error here would quietly move every reported result.
"""
import numpy as np
import pandas as pd
import pytest

from src.config import ELO_HOME_ADV, ELO_SEASON_CARRYOVER, ELO_START
from src.models.elo import elo_win_prob, run_elo

from .conftest import make_games


class TestWinProb:
    def test_equal_ratings_no_home_edge_is_a_coin_flip(self):
        assert elo_win_prob(1500, 1500, home_adv=0) == pytest.approx(0.5)

    def test_probabilities_of_a_reversed_matchup_sum_to_one(self):
        p = elo_win_prob(1600, 1450, home_adv=0)
        assert p + elo_win_prob(1450, 1600, home_adv=0) == pytest.approx(1.0)

    def test_four_hundred_points_is_ten_to_one(self):
        # the definition of the Elo scale: 400 points = 10x the odds
        assert elo_win_prob(1900, 1500, home_adv=0) == pytest.approx(10 / 11)

    def test_home_advantage_helps_the_home_team(self):
        assert elo_win_prob(1500, 1500) > 0.5
        assert elo_win_prob(1500, 1500) == pytest.approx(
            elo_win_prob(1500 + ELO_HOME_ADV, 1500, home_adv=0))

    def test_monotone_in_rating_difference(self):
        probs = [elo_win_prob(1500 + d, 1500) for d in range(-200, 201, 50)]
        assert probs == sorted(probs)


class TestRunElo:
    def test_rating_transfer_is_zero_sum_within_a_season(self, two_team_season):
        _, ratings = run_elo(two_team_season)
        assert sum(ratings.values()) == pytest.approx(2 * ELO_START)

    def test_winning_every_game_raises_the_rating(self, two_team_season):
        _, ratings = run_elo(two_team_season)
        assert ratings[1] > ELO_START > ratings[2]

    def test_pregame_elo_of_a_first_game_is_the_starting_rating(self, two_team_season):
        """The leakage check: the attached rating must be the one *before* the
        game, so both teams enter their opener at ELO_START."""
        out, _ = run_elo(two_team_season)
        first = out.iloc[0]
        assert first["home_elo_pre"] == ELO_START
        assert first["away_elo_pre"] == ELO_START

    def test_pregame_elo_never_reflects_the_current_result(self, two_team_season):
        """Team 1 wins everything, so its pre-game rating must be strictly
        increasing — and each value must be below its post-game rating."""
        out, ratings = run_elo(two_team_season)
        t1 = [r.home_elo_pre if r.home_id == 1 else r.away_elo_pre
              for r in out.itertuples()]
        assert t1 == sorted(t1)
        assert t1[-1] < ratings[1]

    def test_row_order_of_the_input_does_not_matter(self, two_team_season):
        shuffled = two_team_season.sample(frac=1, random_state=0)
        a, ratings_a = run_elo(two_team_season)
        b, ratings_b = run_elo(shuffled)
        assert ratings_a == pytest.approx(ratings_b)
        assert a["elo_prob_home"].to_numpy() == pytest.approx(b["elo_prob_home"].to_numpy())

    def test_bigger_margins_move_ratings_further(self):
        blowout = make_games([(1, 2, 12, 0)])
        squeaker = make_games([(1, 2, 1, 0)])
        _, big = run_elo(blowout)
        _, small = run_elo(squeaker)
        assert big[1] - ELO_START > small[1] - ELO_START > 0

    def test_margin_of_victory_can_be_switched_off(self):
        games = make_games([(1, 2, 12, 0)])
        _, with_mov = run_elo(games)
        _, without = run_elo(games, use_mov=False)
        assert with_mov[1] != pytest.approx(without[1])

    def test_offseason_regresses_toward_the_mean(self):
        """After a season boundary a team keeps ELO_SEASON_CARRYOVER of its
        distance from 1500 — this is what stops ratings drifting forever."""
        y1 = make_games([(1, 2, 5, 1)] * 10, season=2024, start="2024-04-01")
        _, end_of_2024 = run_elo(y1)

        y2 = make_games([(1, 2, 5, 1)], season=2025, start="2025-04-01")
        both, _ = run_elo(pd.concat([y1, y2], ignore_index=True))
        opener_2025 = both[both["season"] == 2025].iloc[0]

        expected = ELO_START * (1 - ELO_SEASON_CARRYOVER) + end_of_2024[1] * ELO_SEASON_CARRYOVER
        assert opener_2025["home_elo_pre"] == pytest.approx(expected)
        # and it moved back toward 1500, not away from it
        assert abs(opener_2025["home_elo_pre"] - ELO_START) < abs(end_of_2024[1] - ELO_START)

    def test_attached_probability_matches_the_attached_ratings(self, two_team_season):
        out, _ = run_elo(two_team_season)
        expected = [elo_win_prob(r.home_elo_pre, r.away_elo_pre)
                    for r in out.itertuples()]
        assert out["elo_prob_home"].to_numpy() == pytest.approx(np.array(expected))
