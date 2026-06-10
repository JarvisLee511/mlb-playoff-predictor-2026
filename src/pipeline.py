"""Reusable pipeline stages, shared by run_pipeline.py and daily_update.py."""
import pandas as pd

from src.config import CURRENT_SEASON, DATA_PROCESSED, DATA_RAW, OUTPUTS
from src.features import build_features, current_team_snapshot
from src.models.elo import run_elo


def refresh_data() -> None:
    from src.data import fetch, gamelogs, pitchers

    fetch.main()
    gamelogs.update_current()
    pitchers.update_current()


def build_ratings_and_features() -> None:
    print("\nRunning Elo ratings...")
    games = pd.read_csv(DATA_RAW / "games.csv")
    games, ratings = run_elo(games)

    teams = pd.read_csv(DATA_RAW / "teams.csv")
    elo_df = (
        pd.DataFrame([{"team_id": t, "elo": e} for t, e in ratings.items()])
        .merge(teams, on="team_id")
        .sort_values("elo", ascending=False)
    )
    elo_df.to_csv(OUTPUTS / "elo_current.csv", index=False)
    print(elo_df[["team_name", "elo"]].head(5).to_string(index=False))

    print("\nBuilding features...")
    from src.data import gamelogs as gl
    from src.data import pitchers as pt
    from src.features import (
        LEAGUE_FIP,
        PARK_HFA_PRIOR,
        build_bullpen_snapshots,
        build_pitcher_snapshots,
        bullpen_fatigue,
    )
    from src.predictions import today_et

    logs = gl.load()
    probables, pitcher_logs = pt.load()
    feats = build_features(games, gamelogs=logs, probables=probables, pitcher_logs=pitcher_logs)
    feats.to_csv(DATA_PROCESSED / "features.csv", index=False)

    snapshot = current_team_snapshot(
        feats, ratings, gamelogs=logs, season=CURRENT_SEASON
    ).merge(teams, on="team_id")

    # current bullpen quality, 3-day workload, and park HFA for upcoming games
    bp_snaps, bp_daily = build_bullpen_snapshots(logs, pitcher_logs)
    cur_bp = (
        bp_snaps[bp_snaps["season"] == CURRENT_SEASON]
        .sort_values("date").groupby("team_id").tail(1)[["team_id", "bp_fip"]]
    )
    snapshot = snapshot.merge(cur_bp, on="team_id", how="left")
    snapshot["bp_fip"] = snapshot["bp_fip"].fillna(LEAGUE_FIP)
    today = pd.Timestamp(today_et())
    snapshot["bp_ip3"] = [bullpen_fatigue(t, today, bp_daily) for t in snapshot["team_id"]]
    cur_park = (
        feats.sort_values("date").groupby("home_id").tail(1)[["home_id", "park_hfa"]]
        .rename(columns={"home_id": "team_id"})
    )
    snapshot = snapshot.merge(cur_park, on="team_id", how="left")
    snapshot["park_hfa"] = snapshot["park_hfa"].fillna(PARK_HFA_PRIOR)

    # current-season home/road splits and last-10 form
    cur = games[games["season"] == CURRENT_SEASON].sort_values(["date", "game_id"])
    splits: dict[int, list] = {}   # team -> [home_w, home_n, road_w, road_n]
    recent: dict[int, list] = {}
    for r in cur.itertuples():
        sh = splits.setdefault(r.home_id, [0, 0, 0, 0])
        sh[0] += r.home_win
        sh[1] += 1
        sa = splits.setdefault(r.away_id, [0, 0, 0, 0])
        sa[2] += 1 - r.home_win
        sa[3] += 1
        recent.setdefault(r.home_id, []).append(r.home_win)
        recent.setdefault(r.away_id, []).append(1 - r.home_win)
    z = [0, 0, 0, 0]
    snapshot["home_winpct_s"] = [
        (splits.get(t, z)[0] + 5.4) / (splits.get(t, z)[1] + 10) for t in snapshot["team_id"]
    ]
    snapshot["road_winpct_s"] = [
        (splits.get(t, z)[2] + 4.6) / (splits.get(t, z)[3] + 10) for t in snapshot["team_id"]
    ]
    snapshot["winpct_10"] = [
        sum(recent.get(t, [])[-10:]) / len(recent.get(t, [])[-10:])
        if len(recent.get(t, [])) >= 5 else 0.5
        for t in snapshot["team_id"]
    ]
    snapshot.to_csv(DATA_PROCESSED / "current_team_stats.csv", index=False)

    sp_snaps = build_pitcher_snapshots(pitcher_logs)
    cur_sp = (
        sp_snaps[sp_snaps["season"] == CURRENT_SEASON]
        .sort_values("date").groupby("pitcher_id").tail(1)
    )
    cur_sp[["pitcher_id", "sp_fip", "sp_kbb", "sp_fip5", "last_app"]].to_csv(
        DATA_PROCESSED / "current_pitcher_stats.csv", index=False
    )


def train_models() -> None:
    print("\nTraining models...")
    from src.models import train

    train.main()


def simulate_season() -> None:
    print("\nSimulating season...")
    from src import simulate

    simulate.main()
