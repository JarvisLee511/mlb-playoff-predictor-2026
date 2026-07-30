"""Season and postseason simulation.

The published playoff odds are the output of this bracket, so the structural
rules matter as much as the probabilities: 12 teams in, three division winners
per league, a best-of-N that a team cannot lose while winning a majority.
"""
import numpy as np
import pytest

from src.simulate import SERIES_HOME, seed_league, sim_postseason, sim_series


def flat_elos(teams, value=1500.0):
    return {t: value for t in teams["team_id"]}


class TestSeries:
    @pytest.mark.parametrize("length", [3, 5, 7])
    def test_home_schedule_covers_the_maximum_number_of_games(self, length):
        assert len(SERIES_HOME[length]) == length

    @pytest.mark.parametrize("length", [3, 5, 7])
    def test_the_higher_seed_hosts_the_majority(self, length):
        home = SERIES_HOME[length]
        assert sum(home) > len(home) / 2

    @pytest.mark.parametrize("length", [3, 5, 7])
    def test_winner_is_one_of_the_two_teams(self, length, rng):
        elos = {1: 1500.0, 2: 1500.0}
        winners = {sim_series(1, 2, elos, length, rng) for _ in range(200)}
        assert winners <= {1, 2}

    def test_an_overwhelming_favourite_almost_always_advances(self, rng):
        elos = {1: 2100.0, 2: 1300.0}
        wins = sum(sim_series(1, 2, elos, 7, rng) == 1 for _ in range(500))
        assert wins > 480

    def test_evenly_matched_teams_split_roughly_evenly(self, rng):
        elos = {1: 1500.0, 2: 1500.0}
        wins = sum(sim_series(1, 2, elos, 7, rng) == 1 for _ in range(2000))
        # home-field advantage tilts it to the higher seed, but not far
        assert 0.5 < wins / 2000 < 0.62

    def test_longer_series_favour_the_better_team(self, rng):
        elos = {1: 1650.0, 2: 1500.0}
        short = sum(sim_series(1, 2, elos, 3, rng) == 1 for _ in range(4000)) / 4000
        long_ = sum(sim_series(1, 2, elos, 7, rng) == 1 for _ in range(4000)) / 4000
        assert long_ > short


class TestSeeding:
    def test_returns_six_distinct_seeds(self, teams_frame, rng):
        records = {t: 0.4 + 0.01 * t for t in teams_frame["team_id"]}
        seeds = seed_league(records, teams_frame, "American League", rng)
        assert len(seeds) == 6
        assert len(set(seeds)) == 6

    def test_only_teams_from_that_league_are_seeded(self, teams_frame, rng):
        records = {t: 0.4 + 0.01 * t for t in teams_frame["team_id"]}
        al = set(teams_frame[teams_frame["league"] == "American League"]["team_id"])
        assert set(seed_league(records, teams_frame, "American League", rng)) <= al

    def test_top_three_seeds_are_the_division_winners(self, teams_frame, rng):
        """Seeds 1-3 must be one team from each division, ordered by record."""
        records = {t: 0.400 + 0.004 * t for t in teams_frame["team_id"]}
        seeds = seed_league(records, teams_frame, "American League", rng)
        al = teams_frame[teams_frame["league"] == "American League"]

        divisions = [al.loc[al["team_id"] == t, "division"].iloc[0] for t in seeds[:3]]
        assert len(set(divisions)) == 3
        for t in seeds[:3]:
            div = al.loc[al["team_id"] == t, "division"].iloc[0]
            rivals = al[al["division"] == div]["team_id"]
            assert records[t] == max(records[r] for r in rivals)
        assert [records[t] for t in seeds[:3]] == sorted(
            (records[t] for t in seeds[:3]), reverse=True)

    def test_wildcards_are_the_next_three_non_winners(self, teams_frame, rng):
        records = {t: 0.400 + 0.004 * t for t in teams_frame["team_id"]}
        seeds = seed_league(records, teams_frame, "American League", rng)
        al = teams_frame[teams_frame["league"] == "American League"]
        rest = [t for t in al["team_id"] if t not in seeds[:3]]
        expected = sorted(rest, key=lambda t: records[t], reverse=True)[:3]
        assert seeds[3:] == expected

    def test_a_division_winner_can_have_a_worse_record_than_a_wildcard(self, teams_frame, rng):
        """MLB seeds division winners 1-3 regardless of record — a 100-win
        wildcard still seeds behind an 85-win division champion."""
        al = teams_frame[teams_frame["league"] == "American League"]
        east = al[al["division"].str.endswith("East")]["team_id"].tolist()
        west = al[al["division"].str.endswith("West")]["team_id"].tolist()

        records = {t: 0.400 for t in al["team_id"]}
        records[west[0]] = 0.520              # weak division winner
        for t in east[:2]:
            records[t] = 0.640               # two juggernauts, one wins the East
        seeds = seed_league(records, al, "American League", rng)

        assert west[0] in seeds[:3]
        runner_up = [t for t in east[:2] if t not in seeds[:3]][0]
        assert seeds.index(runner_up) > seeds.index(west[0])


class TestPostseason:
    def _bracket(self, teams_frame, rng):
        records = {t: 0.400 + 0.004 * t for t in teams_frame["team_id"]}
        elos = flat_elos(teams_frame)
        seeds = {lg: seed_league(records, teams_frame, lg, rng)
                 for lg in ("American League", "National League")}
        return seeds, sim_postseason(seeds, elos, records, rng)

    def test_twelve_teams_reach_the_postseason(self, teams_frame, rng):
        seeds, _ = self._bracket(teams_frame, rng)
        assert sum(len(s) for s in seeds.values()) == 12

    def test_champion_played_in_the_world_series(self, teams_frame, rng):
        for _ in range(50):
            _, out = self._bracket(teams_frame, rng)
            assert out["champion"] in out["ws_teams"]

    def test_the_world_series_is_one_team_from_each_league(self, teams_frame, rng):
        for _ in range(50):
            seeds, out = self._bracket(teams_frame, rng)
            al, nl = set(seeds["American League"]), set(seeds["National League"])
            assert len(out["ws_teams"]) == 2
            assert len({t in al for t in out["ws_teams"]}) == 2

    def test_every_advancing_team_qualified(self, teams_frame, rng):
        seeds, out = self._bracket(teams_frame, rng)
        qualified = set(seeds["American League"]) | set(seeds["National League"])
        for key in ("ds", "cs", "ws_teams"):
            assert set(out[key]) <= qualified

    def test_byes_go_to_the_top_two_seeds(self, teams_frame, rng):
        """Seeds 1 and 2 skip the wild-card round, so they appear in `ds`
        without having played a series."""
        seeds, out = self._bracket(teams_frame, rng)
        for lg in seeds:
            assert seeds[lg][0] in out["ds"]
            assert seeds[lg][1] in out["ds"]

    def test_eight_teams_reach_the_division_series(self, teams_frame, rng):
        _, out = self._bracket(teams_frame, rng)
        assert len(out["ds"]) == 8
        assert len(set(out["ds"])) == 8

    def test_four_teams_reach_the_championship_series(self, teams_frame, rng):
        _, out = self._bracket(teams_frame, rng)
        assert len(out["cs"]) == 4
        assert len(set(out["cs"])) == 4

    def test_the_higher_seed_gets_home_field_in_every_matchup(self, teams_frame, rng):
        """Added after a mutation test: swapping the arguments of one wild-card
        series passed the whole suite. Each matchup is counted separately on
        purpose — summing them lets one reversed series be cancelled out by the
        other, which is exactly how the first version of this test missed it.

        With identical Elo ratings, home field is the only thing separating the
        teams, so each higher seed must advance more than half the time.
        """
        records = {t: 0.400 + 0.004 * t for t in teams_frame["team_id"]}
        elos = flat_elos(teams_frame)          # identical strength on purpose
        seeds = {lg: seed_league(records, teams_frame, lg, rng)
                 for lg in ("American League", "National League")}

        n = 3000
        # per (league, matchup) -> times the higher seed advanced
        wc_wins = {(lg, i): 0 for lg in seeds for i in (0, 1)}
        ds_wins = {(lg, i): 0 for lg in seeds for i in (0, 1)}

        for _ in range(n):
            out = sim_postseason(seeds, elos, records, rng)
            for lg, s in seeds.items():
                # wild card: 3v6 and 4v5
                for i, high in enumerate((s[2], s[3])):
                    if high in out["ds"]:
                        wc_wins[(lg, i)] += 1
                # division series: the two bye teams host
                for i, high in enumerate((s[0], s[1])):
                    if high in out["cs"]:
                        ds_wins[(lg, i)] += 1

        for key, wins in wc_wins.items():
            rate = wins / n
            assert rate > 0.5, f"wild-card {key} higher seed advanced only {rate:.1%}"
            assert rate < 0.62, f"wild-card {key} looks like a talent gap: {rate:.1%}"
        for key, wins in ds_wins.items():
            assert wins / n > 0.5, f"division series {key}: {wins / n:.1%}"
