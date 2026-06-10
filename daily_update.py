"""Daily automation entry point — run by GitHub Actions every morning (ET).

Order matters:
1. refresh data           -> yesterday's results land in games.csv
2. score predictions      -> yesterday's pending predictions become finals
3. rebuild Elo + features -> ratings now include yesterday's games
4. retrain models         -> continuous learning on the expanding window
5. simulate season        -> updated playoff odds
6. predict today          -> logged BEFORE games are played (true out-of-sample)
7. export site JSON       -> includes fresh injury/roster transactions
"""
from src.pipeline import (
    build_ratings_and_features,
    refresh_data,
    simulate_season,
    train_models,
)
from src.predictions import predict_today, score_pending
from src.site_export import export_all


def main() -> None:
    refresh_data()
    score_pending()
    build_ratings_and_features()
    train_models()
    simulate_season()
    predict_today()
    export_all()
    print("\nDaily update complete.")


if __name__ == "__main__":
    main()
