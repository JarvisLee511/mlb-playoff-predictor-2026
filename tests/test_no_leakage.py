"""No-leakage guarantees.

Everything in this project rests on one claim: every feature is knowable at
first pitch. A leak would inflate the backtest while the live tracker stayed
flat, which is the hardest kind of error to notice — the numbers just look
better. These tests make a leak fail loudly instead.

The technique throughout: plant an extreme value in game N, then assert the
feature attached to game N does not move.
"""
import numpy as np
import pandas as pd
import pytest

from src.features import (
    ADV_MIN_GAMES,
    _asof_bullpen,
    _asof_starter,
    build_advanced_pregame,
    build_bullpen_snapshots,
    build_pitcher_snapshots,
    bullpen_fatigue,
)

from .conftest import make_gamelog_rows


def _pre(gamelogs, game_id, col):
    out = build_advanced_pregame(gamelogs)
    return out.loc[out["game_id"] == game_id, col].iloc[0]


class TestSeasonToDateStats:
    def test_current_game_is_excluded(self):
        """Game 20 is a 20-run explosion. The pre-game OPS for game 20 must be
        identical to the flat-season value; only game 21 may see it."""
        n = 25
        flat = make_gamelog_rows(1, [{} for _ in range(n)])
        spiked = make_gamelog_rows(
            1, [{"h_hits": 25, "h_homeRuns": 8, "h_runs": 20} if i == 19 else {}
                for i in range(n)])

        target = flat["game_id"].iloc[19]
        assert _pre(spiked, target, "ops") == pytest.approx(_pre(flat, target, "ops"))

        after = flat["game_id"].iloc[20]
        assert _pre(spiked, after, "ops") > _pre(flat, after, "ops")

    def test_a_stat_only_reflects_strictly_earlier_games(self):
        """Hand-computable. Every game is a 1-1 draw except a 30-0 blowout at
        index 20 — the first index far enough in for the stat to be published."""
        n = 25
        blowout = 20
        logs = make_gamelog_rows(
            1, [{"h_runs": 30, "p_runs": 0} if i == blowout else {"h_runs": 1, "p_runs": 1}
                for i in range(n)])
        # before the blowout the history is 20 games of 1-1 -> pyth is exactly 0.5
        assert _pre(logs, logs["game_id"].iloc[blowout], "pyth") == pytest.approx(0.5)
        # the next game sees it: 21 games of history, runs 50-20
        odds = (50 / 20) ** 1.83
        assert _pre(logs, logs["game_id"].iloc[blowout + 1], "pyth") == pytest.approx(
            odds / (1 + odds))

    def test_thin_samples_are_nan_not_noise(self):
        logs = make_gamelog_rows(1, [{} for _ in range(ADV_MIN_GAMES + 3)])
        out = build_advanced_pregame(logs)
        early = out.iloc[:ADV_MIN_GAMES]
        later = out.iloc[ADV_MIN_GAMES:]
        assert early["ops"].isna().all()
        assert later["ops"].notna().all()

    def test_history_does_not_cross_the_season_boundary(self):
        """A team's 2026 opener must not inherit 2025's totals."""
        y1 = make_gamelog_rows(1, [{"h_runs": 12} for _ in range(25)],
                               season=2025, start="2025-04-01", game_id0=200_000)
        y2 = make_gamelog_rows(1, [{} for _ in range(25)],
                               season=2026, start="2026-04-01", game_id0=300_000)
        out = build_advanced_pregame(pd.concat([y1, y2], ignore_index=True))

        opener_2026 = out.loc[out["game_id"] == 300_000]
        # zero prior games in the new season -> below the minimum -> NaN
        assert opener_2026["ops"].isna().all()

        # and once 2026 has enough games, its stats reflect 2026 only
        late_2026 = out.loc[out["game_id"] == 300_000 + 20, "pyth"].iloc[0]
        flat = make_gamelog_rows(1, [{} for _ in range(25)], season=2026,
                                 start="2026-04-01", game_id0=300_000)
        assert late_2026 == pytest.approx(_pre(flat, 300_000 + 20, "pyth"))

    def test_infinities_are_scrubbed(self):
        """A team with zero innings pitched would divide by zero in ERA/WHIP."""
        logs = make_gamelog_rows(1, [{"p_ip": 0.0} for _ in range(25)])
        out = build_advanced_pregame(logs)
        assert np.isfinite(out[["era", "whip"]].to_numpy(dtype=float)).sum() == 0
        assert not out.isin([np.inf, -np.inf]).to_numpy().any()


def _pitcher_logs(rows, season=2025, start="2025-04-01"):
    dates = pd.date_range(start, periods=len(rows), freq="7D")
    out = []
    for i, (over, d) in enumerate(zip(rows, dates)):
        row = {"season": season, "pitcher_id": 7, "team_id": 1, "date": d,
               "game_id": 400_000 + i, "gs": 1, "ip": 6.0, "er": 2, "bb": 2,
               "so": 6, "hr": 1, "hbp": 0, "bf": 24}
        row.update(over)
        out.append(row)
    return pd.DataFrame(out)


class TestStarterAsOfJoin:
    def test_a_snapshot_from_the_same_day_is_not_used(self):
        """merge_asof runs with allow_exact_matches=False. If that ever flips,
        a start would be scored using its own line."""
        logs = _pitcher_logs([{}, {}, {"so": 30, "bb": 0, "hr": 0}])
        snaps = build_pitcher_snapshots(logs)

        game_date = logs["date"].iloc[2]
        games = pd.DataFrame([{"game_id": 900, "date": game_date, "season": 2025,
                               "home_pitcher": 7}])
        joined = _asof_starter(games, snaps, "home_pitcher", "h")

        # the value must equal the snapshot after appearance 2, not appearance 3
        expected = snaps.loc[snaps["date"] == logs["date"].iloc[1], "sp_kbb"].iloc[0]
        assert joined["h_sp_kbb"].iloc[0] == pytest.approx(expected)

        dominant = snaps.loc[snaps["date"] == game_date, "sp_kbb"].iloc[0]
        assert joined["h_sp_kbb"].iloc[0] < dominant

    def test_a_debut_start_has_no_snapshot(self):
        logs = _pitcher_logs([{}])
        snaps = build_pitcher_snapshots(logs)
        games = pd.DataFrame([{"game_id": 900, "date": logs["date"].iloc[0],
                               "season": 2025, "home_pitcher": 7}])
        joined = _asof_starter(games, snaps, "home_pitcher", "h")
        assert joined["h_sp_fip"].isna().all()

    def test_snapshots_do_not_leak_across_seasons(self):
        y1 = _pitcher_logs([{"so": 40, "bb": 0, "hr": 0}] * 3,
                           season=2025, start="2025-04-01")
        y2 = _pitcher_logs([{}] * 3, season=2026, start="2026-04-01")
        logs = pd.concat([y1, y2], ignore_index=True)
        snaps = build_pitcher_snapshots(logs)

        games = pd.DataFrame([{"game_id": 900, "date": y2["date"].iloc[0],
                               "season": 2026, "home_pitcher": 7}])
        joined = _asof_starter(games, snaps, "home_pitcher", "h")
        # left_by includes season, so 2025's dominant form is unavailable in 2026
        assert joined["h_sp_kbb"].isna().all()

    def test_snapshot_is_cumulative_over_prior_appearances(self):
        logs = _pitcher_logs([{}] * 5)
        snaps = build_pitcher_snapshots(logs).sort_values("date")
        # a pitcher repeating an identical line: shrinkage pulls the estimate
        # from the league prior toward the true rate, so |fip - prior| grows
        from src.features import LEAGUE_FIP
        gaps = (snaps["sp_fip"] - LEAGUE_FIP).abs().to_numpy()
        assert (np.diff(gaps) >= -1e-12).all()


class TestBullpenAsOfJoin:
    """Same guarantee as the starter join, on the other as-of merge. Added after
    a mutation test showed this one was unguarded."""

    def _team_history(self, n=4, spike_index=None):
        gamelogs = make_gamelog_rows(1, [{} for _ in range(n)], game_id0=500_000)
        rows = []
        for i, d in enumerate(gamelogs["date"]):
            # the starter goes 6 innings; the bullpen covers the other 3
            spike = (spike_index is not None and i == spike_index)
            rows.append({"season": 2025, "pitcher_id": 7, "team_id": 1, "date": d,
                         "game_id": 500_000 + i, "gs": 1, "ip": 6.0, "er": 2,
                         "bb": 1, "so": 6, "hr": 0 if not spike else 0,
                         "hbp": 0, "bf": 24})
            if spike:
                # a bullpen meltdown: the team line stays, but the starter gave
                # up nothing, so every walk/homer lands on the relievers
                gamelogs.loc[gamelogs.index[i], ["p_homeRuns", "p_baseOnBalls"]] = [8, 12]
        return gamelogs, pd.DataFrame(rows)

    def test_a_snapshot_from_the_same_day_is_not_used(self):
        gamelogs, plogs = self._team_history(n=4, spike_index=3)
        snaps, _ = build_bullpen_snapshots(gamelogs, plogs)

        game_date = gamelogs["date"].iloc[3]
        games = pd.DataFrame([{"game_id": 900, "date": game_date, "season": 2025,
                               "home_id": 1}])
        joined = _asof_bullpen(games, snaps, "home_id", "h")

        before = snaps.loc[snaps["date"] == gamelogs["date"].iloc[2], "bp_fip"].iloc[0]
        same_day = snaps.loc[snaps["date"] == game_date, "bp_fip"].iloc[0]
        assert joined["h_bp_fip"].iloc[0] == pytest.approx(before)
        assert joined["h_bp_fip"].iloc[0] < same_day

    def test_a_teams_first_game_has_no_bullpen_snapshot(self):
        gamelogs, plogs = self._team_history(n=2)
        snaps, _ = build_bullpen_snapshots(gamelogs, plogs)
        games = pd.DataFrame([{"game_id": 900, "date": gamelogs["date"].iloc[0],
                               "season": 2025, "home_id": 1}])
        assert _asof_bullpen(games, snaps, "home_id", "h")["h_bp_fip"].isna().all()

    def test_fatigue_counts_only_the_days_before_the_game(self):
        gamelogs, plogs = self._team_history(n=4)
        _, daily_ip = build_bullpen_snapshots(gamelogs, plogs)
        game_date = gamelogs["date"].iloc[3]
        # three prior days at 3 relief innings each
        assert bullpen_fatigue(1, game_date, daily_ip, days=3) == pytest.approx(9.0)
        # and the game's own innings are never included
        assert bullpen_fatigue(1, gamelogs["date"].iloc[0], daily_ip, days=3) == 0.0
