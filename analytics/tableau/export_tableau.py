"""Export the dbt marts from the DuckDB warehouse to flat CSVs for Tableau Public.

Tableau Public cannot connect to a local DuckDB file or refresh from CI, so we
materialise denormalised extracts it can ingest directly. Run AFTER `dbt build`:

    ../../.venv/Scripts/python.exe export_tableau.py

Writes analytics/tableau/data/*.csv. Then follow README.md to build and publish the
workbook.
"""
from pathlib import Path

import duckdb

HERE = Path(__file__).resolve().parent
WAREHOUSE = HERE.parent / "warehouse" / "mlb.duckdb"
OUT = HERE / "data"
OUT.mkdir(exist_ok=True)

# (filename, SQL) — each becomes one flat CSV.
EXPORTS = {
    "standings": "select * from main_marts.mart_standings order by world_series_rank",
    "model_performance": "select * from main_marts.mart_model_performance",
    "calibration": "select * from main_marts.mart_calibration order by model, bin_index",
    "feature_validation": "select * from main_marts.mart_feature_validation order by max_abs_diff desc",
    # long, tidy per-model scored predictions — ideal for accuracy-over-time in Tableau
    "predictions_scored": """
        with finals as (
            select game_date, game_id, home_name, away_name, home_win,
                   p_home_elo, p_home_lr, p_home_xgb, p_home_ens, p_home_skl
            from main_staging.stg_predictions_log
            where is_final and home_win is not null
        )
        unpivot finals
        on p_home_elo, p_home_lr, p_home_xgb, p_home_ens, p_home_skl
        into name prob_col value p_home
    """,
}


def main() -> None:
    if not WAREHOUSE.exists():
        raise SystemExit(f"Warehouse not found at {WAREHOUSE}. Run `dbt build` first.")
    con = duckdb.connect(str(WAREHOUSE), read_only=True)
    for name, sql in EXPORTS.items():
        dest = OUT / f"{name}.csv"
        con.execute(f"copy ({sql}) to '{dest.as_posix()}' (header, delimiter ',')")
        n = con.execute(f"select count(*) from ({sql})").fetchone()[0]
        print(f"  {name}.csv  <- {n} rows")
    con.close()
    print(f"Done. Extracts in {OUT}")


if __name__ == "__main__":
    main()
