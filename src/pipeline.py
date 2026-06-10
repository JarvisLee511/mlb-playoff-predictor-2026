"""Reusable pipeline stages, shared by run_pipeline.py and daily_update.py."""
import pandas as pd

from src.config import CURRENT_SEASON, DATA_PROCESSED, DATA_RAW, OUTPUTS
from src.features import build_features, current_team_snapshot
from src.models.elo import run_elo


def refresh_data() -> None:
    from src.data import fetch, gamelogs

    fetch.main()
    gamelogs.update_current()


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

    logs = gl.load()
    feats = build_features(games, gamelogs=logs)
    feats.to_csv(DATA_PROCESSED / "features.csv", index=False)

    snapshot = current_team_snapshot(
        feats, ratings, gamelogs=logs, season=CURRENT_SEASON
    ).merge(teams, on="team_id")
    snapshot.to_csv(DATA_PROCESSED / "current_team_stats.csv", index=False)


def train_models() -> None:
    print("\nTraining models...")
    from src.models import train

    train.main()


def simulate_season() -> None:
    print("\nSimulating season...")
    from src import simulate

    simulate.main()
