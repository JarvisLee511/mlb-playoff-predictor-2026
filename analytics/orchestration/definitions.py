"""Dagster orchestration for the MLB analytics stack.

Models the whole daily flow as one asset lineage graph:

    mlb_prediction_pipeline   (Python: fetch -> score -> features -> train ->
             |                 simulate -> predict -> export ; wraps daily_update.py)
             v
    dbt sources (data/*.csv, outputs/*)
             |
             v
    staging -> intermediate -> marts   (dbt-duckdb, loaded via dagster-dbt)

Run the UI locally with:

    cd analytics/orchestration
    ../../.venv/Scripts/dagster.exe dev -f definitions.py

Materialize everything from the UI (or `dagster asset materialize --select '*'`).
The dbt assets pass an absolute `mlb_root` so file paths resolve regardless of cwd.
"""
from pathlib import Path

from dagster import (
    AssetExecutionContext,
    AssetKey,
    AssetSpec,
    Definitions,
    MaterializeResult,
    define_asset_job,
    multi_asset,
    ScheduleDefinition,
)
from dagster_dbt import (
    DagsterDbtTranslator,
    DbtCliResource,
    DbtProject,
    dbt_assets,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent                 # mlb-playoff-predictor-2026/
DBT_PROJECT_DIR = HERE.parent / "dbt"          # analytics/dbt

# The dbt sources (data/*.csv, outputs/*) that the Python pipeline produces. Each gets
# its own asset key so the graph shows exactly which file feeds which model.
SOURCE_TABLES = [
    "gamelogs", "pitcher_logs", "probables", "predictions_log",
    "games", "teams", "features", "calibration", "elo_current", "playoff_odds",
]


def source_asset_key(name: str) -> AssetKey:
    return AssetKey(["mlb_pipeline", name])


dbt_project = DbtProject(
    project_dir=DBT_PROJECT_DIR,
    profiles_dir=DBT_PROJECT_DIR,
)
dbt_project.prepare_if_dev()


class MlbDbtTranslator(DagsterDbtTranslator):
    """Give each dbt source a unique key under the ``mlb_pipeline`` prefix so it lines
    up with the outputs of the upstream pipeline multi-asset."""

    def get_asset_key(self, dbt_resource_props):
        if dbt_resource_props["resource_type"] == "source":
            return source_asset_key(dbt_resource_props["name"])
        return super().get_asset_key(dbt_resource_props)


@multi_asset(
    specs=[
        AssetSpec(source_asset_key(name), kinds={"python", "csv"})
        for name in SOURCE_TABLES
    ],
    can_subset=False,
)
def mlb_prediction_pipeline(context: AssetExecutionContext):
    """Runs the full prediction pipeline (daily_update.py): refresh data, score pending
    predictions, rebuild features, retrain, simulate, predict, export. One run produces
    all the CSV/JSON files the dbt layer reads."""
    import subprocess
    import sys

    context.log.info("Running daily_update.py to refresh pipeline outputs...")
    result = subprocess.run(
        [sys.executable, "daily_update.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    context.log.info(result.stdout[-4000:] if result.stdout else "(no stdout)")
    if result.returncode != 0:
        context.log.error(result.stderr[-4000:])
        raise RuntimeError(f"daily_update.py failed (exit {result.returncode})")
    for name in SOURCE_TABLES:
        yield MaterializeResult(asset_key=source_asset_key(name))


@dbt_assets(
    manifest=dbt_project.manifest_path,
    dagster_dbt_translator=MlbDbtTranslator(),
)
def mlb_dbt_models(context: AssetExecutionContext, dbt: DbtCliResource):
    mlb_root = REPO_ROOT.as_posix()
    yield from dbt.cli(
        ["build", "--vars", f"{{mlb_root: {mlb_root}}}"], context=context
    ).stream()


daily_refresh_job = define_asset_job(name="daily_refresh", selection="*")

daily_schedule = ScheduleDefinition(
    job=daily_refresh_job,
    cron_schedule="30 7 * * *",  # mirrors the GitHub Actions daily cron (3:30 AM ET)
    execution_timezone="America/New_York",
)

defs = Definitions(
    assets=[mlb_prediction_pipeline, mlb_dbt_models],
    jobs=[daily_refresh_job],
    schedules=[daily_schedule],
    resources={
        "dbt": DbtCliResource(
            project_dir=DBT_PROJECT_DIR,
            profiles_dir=DBT_PROJECT_DIR,
        ),
    },
)
