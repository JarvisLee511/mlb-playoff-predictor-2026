"""Season state: when the daily pipeline stops, and how the postseason is recorded.

Synthetic schedule payloads in the MLB Stats API's own shape; nothing is fetched."""
import pandas as pd
import pytest

from src import season


def game(pk, date, detailed, abstract="Final"):
    return {"gamePk": pk, "officialDate": date,
            "status": {"detailedState": detailed, "abstractGameState": abstract}}


def series_game(code, desc, date, home, away, winner, loser, result, over=True):
    team = lambda n: {"id": abs(hash(n)) % 10_000, "name": n}  # noqa: E731
    return {"gameType": code, "seriesDescription": desc, "officialDate": date,
            "teams": {"home": {"team": team(home)}, "away": {"team": team(away)}},
            "seriesStatus": {"isOver": over, "winningTeam": team(winner),
                             "losingTeam": team(loser), "result": result}}


def odds_row(name, abbrev, league, ws, pennant, playoffs=1.0):
    return {"team_name": name, "abbrev": abbrev, "league": league,
            "make_playoffs": playoffs, "win_world_series": ws, "win_pennant": pennant}


def test_games_left_counts_only_unplayed():
    regular = [game(1, "2026-09-27", "Final"), game(2, "2026-09-27", "Postponed"),
               game(3, "2026-09-28", "Scheduled", "Preview"),
               game(4, "2026-09-28", "In Progress", "Live")]
    assert season.games_left(regular) == 2


def test_cancelled_game_is_voided_but_a_made_up_postponement_is_not():
    regular = [game(823490, "2026-09-27", "Cancelled"),
               game(555, "2026-05-01", "Postponed"), game(555, "2026-05-02", "Final")]
    log = pd.DataFrame({"game_id": [823490, 555, 777],
                        "status": ["pending", "pending", "final"]})
    out, n = season.void_cancelled(log, regular)
    assert n == 1
    assert out["status"].tolist() == ["cancelled", "pending", "final"]
    assert log["status"].tolist() == ["pending", "pending", "final"]  # input untouched


BRACKET = [
    series_game("W", "World Series", "2026-10-28", "Rays", "Brewers", "Brewers", "Rays", "MIL wins 4-1"),
    series_game("F", "AL Wild Card", "2026-09-30", "Yankees", "Red Sox", "Yankees", "Red Sox", "NYY wins 2-0"),
    series_game("D", "ALDS", "2026-10-07", "Rays", "Yankees", "Rays", "Yankees", "TB wins 3-0"),
    series_game("L", "NLCS", "2026-10-15", "Brewers", "Dodgers", "Brewers", "Dodgers", "MIL wins 4-0"),
    series_game("L", "ALCS", "2026-10-19", "Rays", "Guardians", "Rays", "Guardians", "TB wins 4-2"),
]
ODDS = [
    odds_row("Brewers", "MIL", "NL", 0.2576, 0.3815),
    odds_row("Dodgers", "LAD", "NL", 0.1907, 0.3029),
    odds_row("Rays", "TB", "AL", 0.1329, 0.3094),
    odds_row("Yankees", "NYY", "AL", 0.0875, 0.1950),
    odds_row("Guardians", "CLE", "AL", 0.0800, 0.2111),
    odds_row("Red Sox", "BOS", "AL", 0.0300, 0.0700),
    odds_row("Orioles", "BAL", "AL", 0.0, 0.0, playoffs=0.0),
]


def test_series_come_out_in_bracket_order():
    rounds = [s["round"] for s in season.series_results(BRACKET)]
    assert rounds == ["Wild Card Series", "Division Series", "League Championship Series",
                      "League Championship Series", "World Series"]


def test_each_team_is_labelled_by_its_last_series():
    results = season.team_results(season.series_results(BRACKET))
    assert results == {"Brewers": "Won World Series", "Rays": "Lost World Series",
                       "Dodgers": "Lost LCS", "Guardians": "Lost LCS",
                       "Yankees": "Lost Division Series", "Red Sox": "Lost Wild Card Series"}


def test_summary_scores_the_champion_against_the_frozen_forecast():
    regular = [game(1, "2026-09-27", "Final")]
    out = season.summarise(2026, regular, BRACKET, ODDS)
    assert out["status"] == "complete"
    assert out["n_playoff_teams"] == 6  # the team that missed is not in the field
    assert out["champion"]["abbrev"] == "MIL"
    assert out["champion"]["rank_world_series"] == 1
    assert out["favourite"]["abbrev"] == "MIL"
    assert [p["abbrev"] for p in out["pennant_winners"]] == ["MIL", "TB"]
    assert out["pennant_winners"][1]["rank_pennant_in_league"] == 1
    assert out["results"]["BOS"] == "Lost Wild Card Series"
    assert out["regular_season_ended"] == "2026-09-27"
    assert out["postseason_began"] == "2026-09-30"


def test_summary_before_the_world_series_ends_is_postseason():
    unfinished = BRACKET[1:]  # no World Series result yet
    out = season.summarise(2026, [game(1, "2026-09-27", "Final")], unfinished, ODDS)
    assert out["status"] == "postseason"
    assert out["champion"] is None


def test_season_file_is_rewritten_only_when_it_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(season, "SITE_DATA", tmp_path)
    monkeypatch.setattr(season, "SEASON_JSON", tmp_path / "season.json")
    payload = {"season": 2026, "status": "complete"}
    assert season.write_if_changed(payload) is True
    assert season.write_if_changed(payload) is False
    assert season.write_if_changed({**payload, "status": "postseason"}) is True


@pytest.fixture
def site(tmp_path, monkeypatch):
    """Point every file update() touches at a temp dir: tests never write docs/data."""
    monkeypatch.setattr(season, "SITE_DATA", tmp_path)
    monkeypatch.setattr(season, "SEASON_JSON", tmp_path / "season.json")
    monkeypatch.setattr(season, "PRED_LOG", tmp_path / "predictions_log.csv")
    return tmp_path


@pytest.mark.parametrize("regular, statuses", [
    ([game(1, "2026-09-28", "Scheduled", "Preview")], ["final"]),  # games left
    ([game(1, "2026-09-27", "Final")], ["final", "pending"]),      # one still to score
])
def test_update_keeps_the_pipeline_running_while_there_is_work(
        regular, statuses, site, monkeypatch):
    monkeypatch.setattr(season, "_games", lambda *a, **k: regular)
    monkeypatch.setattr(season, "load_log", lambda: pd.DataFrame(
        {"game_id": range(len(statuses)), "status": statuses}))
    assert season.update(2026) == "regular"
    assert '"status": "regular"' in (site / "season.json").read_text(encoding="utf-8")


def test_update_freezes_once_everything_is_scored(site, monkeypatch):
    tmp_path = site
    calls = []

    def fake_games(_season, game_type, hydrate=None):
        calls.append(game_type)
        return [game(1, "2026-09-27", "Final")] if game_type == "R" else BRACKET

    monkeypatch.setattr(season, "_games", fake_games)
    monkeypatch.setattr(season, "load_log", lambda: pd.DataFrame(
        {"game_id": [1], "status": ["final"]}))
    odds = tmp_path / "odds.json"
    odds.write_text(pd.Series(ODDS).to_json(orient="records"), encoding="utf-8")
    monkeypatch.setattr(season, "ODDS_JSON", odds)
    assert season.update(2026) == "complete"
    assert calls == ["R", "F,D,L,W"]
    assert (tmp_path / "season.json").exists()
