"""End-to-end pipeline: fetch data -> Elo -> features -> train -> simulate.

Usage:
    python run_pipeline.py            # everything
    python run_pipeline.py --no-fetch # reuse cached raw data
"""
import argparse

from src.pipeline import (
    build_ratings_and_features,
    refresh_data,
    simulate_season,
    train_models,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-fetch", action="store_true", help="skip the API fetch step")
    args = parser.parse_args()

    if not args.no_fetch:
        refresh_data()
    build_ratings_and_features()
    train_models()
    simulate_season()


if __name__ == "__main__":
    main()
